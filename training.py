import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score
from loss import ContrastiveLoss, ContrastiveLossCosine
from model import CNN_BLSTM_SELF_ATTN
from sklearn.metrics import f1_score
import torch.nn.functional as F
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train(model, num_epochs, dataloader_train1, dataloader_train2, support, query, path):
    # continue training with model
    # model.load_state_dict(torch.load('state_dict_model_meta_singapore_en_1200ep_emb150_batch32.pt'))
    model.train()
    labels_emo_map = {'sad': 0, 'ang': 1, 'hap': 2, 'neu': 3, 'pok': 4}

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('CUDA available')
        model.cuda()
    else:
        device = 'cpu'
        print('CUDA not available - CPU is used')

    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           weight_decay=0.00001, betas=(0.9, 0.98), eps=1e-9)
    # criterion = ContrastiveLoss(4)
    criterion = torch.nn.CosineEmbeddingLoss()
    # Measures the loss given an input tensor x and a labels tensor y (containing 1 or -1)

    # scheduler = ReduceLROnPlateau(
    #    optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    train_loss_list = []
    for epoch in range(num_epochs):
        for i, (sample1, sample2) in enumerate(zip(dataloader_train1, dataloader_train2)):
            # categorical to numeric labels
            label_emo1 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample1[1]])

            label_emo2 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample2[1]])

            targets = list()
            for label1, label2 in zip(label_emo1, label_emo2):
                if label1 == label2:
                    targets.append(1)
                else:
                    targets.append(-1)
            targets = torch.FloatTensor(targets)

            # print(targets)
            spec_features1 = sample1[0]
            spec_features2 = sample2[0]
            # print(spec_features1.shape)

            spec_features1, spec_features2, label_emo1, label_emo2, targets = spec_features1.to(
                device, dtype=torch.float), spec_features2.to(
                device, dtype=torch.float), label_emo1.to(device), label_emo2.to(device), targets.to(device)

            spec_features1.requires_grad = True
            spec_features2.requires_grad = True

            # clear the gradients
            optimizer.zero_grad()

            # model output

            emotion_embedding1, emotion_embedding2 = model(
                spec_features1, spec_features2)  # siamese
            #emotion_embedding1 = model(spec_features1)
            #emotion_embedding2 = model(spec_features2)

            # print(emotion_embedding1)
            #print(f'emotion embedding1: {emotion_embedding1}')
            #print(f'emotion embedding2: {emotion_embedding2}')

            #emotion_embedding1 = torch.unsqueeze(emotion_embedding1, 0)
            #emotion_embedding2 = torch.unsqueeze(emotion_embedding2, 0)

            # calculate loss
            loss = criterion(emotion_embedding1, emotion_embedding2, targets)

            loss.backward()
            # update model weights
            optimizer.step()

            train_loss_list.append(loss.item())

            if i % 500 == 0:
                print('Loss {} after {} iterations'.format(
                    np.mean(np.asarray(train_loss_list)), i))

        mean_loss = np.mean(np.asarray(train_loss_list))
        print('Loss {} after {} epochs'.format(
            np.mean(np.asarray(mean_loss)), epoch))

        torch.save(model.state_dict(), path)
        #evaluate(model, support, query, path)


def evaluate(model, support, query, PATH, confusion_file):
    labels_emo_map = {'sad': 0, 'ang': 1, 'hap': 2, 'neu': 3, 'pok': 4}

    with torch.no_grad():
        model.load_state_dict(torch.load(PATH))
        model.double()
        model.eval()
        y_pred = list()
        y_true = list()
        for i, samp1 in enumerate(query):
            #print(f'query: {samp1}')
            label_emo1 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in samp1[1]])
            y_true.append(int(label_emo1.item()))
            spec_query = samp1[0]

            losses = dict()
            # compare sample in query set with every sample in emotion set and choose emotion with highest similarity
            for samp2 in support:
                label_emo2 = torch.FloatTensor(
                    [float(labels_emo_map[label]) for label in samp2[1]])

                if label_emo1 == label_emo2:
                    label = 1
                else:
                    label = 0
                spec_support = samp2[0]
                emb1, emb2 = model(spec_query, spec_support)

                #emb1 = torch.unsqueeze(emb1, 0)
                #emb2 = torch.unsqueeze(emb2, 0)

                similarity = abs(distance.cosine(emb1, emb2))

                #_loss = loss(emb1, emb2, label)
                losses[int(label_emo2.item())] = similarity
                # print(losses)

            prediction = min(losses, key=losses.get)
            y_pred.append(prediction)

        print(f'predictions: {y_pred}')
        print(f'gold label: {y_true}')
        f_score_classes = f1_score(y_true, y_pred, average=None)
        f_score = f1_score(y_true, y_pred, average='macro')
        print(f'f_score per class: {f_score_classes}')
        print(f'f_score overall: {f_score}')
        x_axis_labels = ['sad', 'ang', 'hap', 'neu', 'sur']  # labels for x-axis
        y_axis_labels = ['sad', 'ang', 'hap', 'neu', 'sur']

        # Get the confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        print(cf_matrix)
        svm = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                          fmt='.2%', cmap='Blues', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        figure = svm.get_figure()
        figure.savefig(confusion_file, dpi=400)
