import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score
from loss import ContrastiveLoss


def train(model, num_epochs, dataloader_train1, dataloader_train2):
    labels_emo_map = {'sad': 0, 'ang': 1, 'pok': 2, 'hap': 3}
    labels_gen_map = {'m': 0, 'f': 1}

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #    print('CUDA available')
    #else:
    device = 'cpu'
    print('CUDA not available - CPU is used')

    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    loss = ContrastiveLoss(2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_acc_list_emo = []
        train_acc_list_gen = []
        train_loss_list = []
        for i, (sample1, sample2) in enumerate(zip(dataloader_train1, dataloader_train2)):
            # categorical to numeric labels
            label_emo1 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample1[1]])
            #labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample1[2]])
            label_emo2 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample2[1]])
            #labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])
            if label_emo1 == label_emo2:
                label = 0
            else:
                label = 1
    
            spec_features1 = sample1[0]
            spec_features2 = sample2[0]
       
            spec_features1, spec_features2, label_emo1, label_emo2 = spec_features1.to(
                device, dtype=torch.float), spec_features2.to(
                device, dtype=torch.float), label_emo1.to(device), label_emo2.to(device)
            
            
            spec_features1.requires_grad = True
            spec_features2.requires_grad = True

            # clear the gradients
            optimizer.zero_grad()

            # model output
            emotion_embedding1, emotion_embedding2 = model(spec_features1, spec_features2)
            #print(f'emotion embedding1: {emotion_embedding1}')
            #print(f'emotion embedding2: {emotion_embedding1}')

            emotion_embedding1 = torch.unsqueeze(emotion_embedding1, 0)
            emotion_embedding2 = torch.unsqueeze(emotion_embedding2, 0)
          
            # calculate loss
            _loss = loss(emotion_embedding1, emotion_embedding2, label)

            _loss.backward()
            # update model weights
            optimizer.step()

            train_loss_list.append(_loss.item())

            if i % 40 == 0:
                print('Loss {} after {} iterations'.format(
                    np.mean(np.asarray(train_loss_list)), i))

        mean_loss = np.mean(np.asarray(train_loss_list))
     
