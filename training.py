import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score
from loss import ContrastiveLoss
from model import CNN_BLSTM_SELF_ATTN
from sklearn.metrics import f1_score
import torch.nn.functional as F
from scipy.spatial import distance


def train(model, num_epochs, dataloader_train1, dataloader_train2):
    model.load_state_dict(torch.load('state_dict_model_CUDA.pt'))
    model.train()
    labels_emo_map = {'sad': 0, 'ang': 1, 'pok': 2, 'hap': 3, 'neu': 4}
    model = model 
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

    train_loss_list = []
    for epoch in range(num_epochs):
        for i, (sample1, sample2) in enumerate(zip(dataloader_train1, dataloader_train2)):
            # categorical to numeric labels
            label_emo1 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample1[1]])
            
            label_emo2 = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample2[1]])
            
            if label_emo1 == label_emo2:
                label = 1
            else:
                label = 0
     
            #print(label_emo1, label_emo2)

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
            #print(f'emotion embedding2: {emotion_embedding2}')

            emotion_embedding1 = torch.unsqueeze(emotion_embedding1, 0)
            emotion_embedding2 = torch.unsqueeze(emotion_embedding2, 0)
          
            # calculate loss
            _loss = loss(emotion_embedding1, emotion_embedding2, label)

            _loss.backward()
            # update model weights
            optimizer.step()

            train_loss_list.append(_loss.item())

            if i % 500 == 0:
                print('Loss {} after {} iterations'.format(
                    np.mean(np.asarray(train_loss_list)), i))
         

        mean_loss = np.mean(np.asarray(train_loss_list))
        print('Loss {} after {} epochs'.format(
                    np.mean(np.asarray(mean_loss)), epoch))
        
    
        PATH = "state_dict_model_CUDA.pt"
        torch.save(model.state_dict(), PATH)

def evaluate(model, support, query, PATH):
    labels_emo_map = {'sad': 0, 'ang': 1, 'pok': 2, 'hap': 3, 'neu': 4}
    #labels_gen_map = {'m': 0, 'f': 1}
    with torch.no_grad():
            model.load_state_dict(torch.load(PATH))
            model.double()
            model.eval()
            loss = ContrastiveLoss(2)
            y_pred = list()
            y_true = list()
            for i, samp1 in enumerate(query):
                #print(f'query: {samp1}')
                label_emo1 = torch.FloatTensor([float(labels_emo_map[label]) for label in samp1[1]])
                y_true.append(int(label_emo1.item()))
                spec_query = samp1[0]
        
                losses = dict()
                # compare sample in query set with every sample in emotion set and choose emotion with highest similarity 
                for samp2 in support:
                    label_emo2 = torch.FloatTensor([float(labels_emo_map[label]) for label in samp2[1]])
                    
                    if label_emo1 == label_emo2:
                        label = 1
                    else:
                        label = 0
                    spec_support = samp2[0]
                    emb1, emb2 = model(spec_query, spec_support)
                    #print(f'support: {samp2[1]}')

                    #emb1 = torch.unsqueeze(emb1, 0)
                    #emb2 = torch.unsqueeze(emb2, 0)
            
          
                    similarity = abs(distance.euclidean(emb1, emb2))
                   
                    #_loss = loss(emb1, emb2, label)
                    losses[int(label_emo2.item())] = similarity
                    print(losses)
              
                #print(losses)
                prediction = min(losses, key=losses.get)
                y_pred.append(prediction)
   
            print(f'predictions: {y_pred}')
            print(f'gold label: {y_true}')
            score = f1_score(y_true, y_pred, average='macro')
            print(f'f_score: {score}')
          






#labels_gen_map = {'m': 0, 'f': 1}
#labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample1[2]])
#labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in samp1[2]])
#labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])
#labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])