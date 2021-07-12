import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score
from loss import ContrastiveLoss


def train(model, num_epochs, dataloader_train):
    labels_emo_map = {'sad': 0, 'ang': 1, 'pok': 2, 'hap': 3}
    labels_gen_map = {'m': 0, 'f': 1}

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('CUDA available')
    else:
        device = 'cpu'
        print('CUDA not available - CPU is used')

    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    loss = ContrastiveLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_acc_list_emo = []
        train_acc_list_gen = []
        train_loss_list = []
        for i_batch, sample_batched in enumerate(dataloader_train):
            # categorical to numeric labels
            labels_emo = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample_batched[1]])
            labels_gen = torch.FloatTensor(
                [float(labels_gen_map[label]) for label in sample_batched[2]])

            spec_features = torch.from_numpy(np.asarray(
                [torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
            labels_emo = torch.from_numpy(np.asarray(
                [torch_tensor.numpy() for torch_tensor in labels_emo]))
            labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in labels_gen]))

            spec_features, labels_emo = spec_features.to(
                device, dtype=torch.float), labels_emo.to(device)
            spec_features.requires_grad = True
            optimizer.zero_grad()
            #spec_features.unsqueeze(1).shape
    
  
            emotion_embedding = model(spec_features.unsqueeze(1))
            
            '''     

            # calculate loss
            emotion_loss = loss(preds_emo, labels_emo.squeeze())
            gender_loss = loss(preds_gender, labels_gen.squeeze())
            total_loss = 1*emotion_loss  # +0.25*gender_loss
            total_loss.backward()
            optimizer.step()

            train_loss_list.append(total_loss.item())

            # extract most likely label
            predictions_emotion = np.argmax(
                preds_emo.detach().cpu().numpy(), axis=1)
            #predictions_gender = np.argmax(preds_gender.detach().cpu().numpy(),axis=1)

            accuracy_emotion = accuracy_score(
                labels_emo.detach().cpu().numpy(), predictions_emotion)
            #accuracy_gender = accuracy_score(labels_gen.detach().cpu().numpy(),predictions_gender)

            train_acc_list_emo.append(accuracy_emotion)
            # train_acc_list_gen.append(accuracy_gender)
            if i_batch % 20 == 0:
                print('Loss {} after {} iteration'.format(
                    np.mean(np.asarray(train_loss_list)), i_batch))

        mean_loss = np.mean(np.asarray(train_loss_list))
        mean_acc_emo = np.mean(np.asarray(train_acc_list_emo))

        print(f'Total training accuracy {mean_acc_emo} after {epoch}')
        '''