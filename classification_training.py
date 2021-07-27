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
from dataset import EmotionDataset, create_train_test
from classification_model import EmotionClassificationNet




def train(model, num_epochs, dataloader_train):
    #model.load_state_dict(torch.load('state_dict_model_CUDA.pt')) # continue training with model 
    model.train()
    labels_emo_map = {'sad': 0, 'ang': 1, 'pok': 2, 'hap': 3, 'neu': 4}
    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #    print('CUDA available')
    #else:
    device = 'cpu'
    print('CUDA not available - CPU is used')

    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
   
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    train_loss_list = []
    for epoch in range(num_epochs):
        for i, sample in enumerate(dataloader_train):
            # categorical to numeric labels
            label_emo = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in sample[1]])
            print(label_emo)
           
            spec_features = sample[0]
           
       
            spec_features, label_emo= spec_features.to(
                device, dtype=torch.float), label_emo.to(device)
            
            
            spec_features.requires_grad = True
          
            # clear the gradients
            optimizer.zero_grad()

            # model output
            prediction = model(spec_features)
            print(f'prediction: {prediction}')
            prediction = prediction.type(torch.LongTensor)
    
            #prediction = torch.argmax(prediction)
            print(prediction)
        
            prediction = torch.unsqueeze(prediction, 0)
            #label_emo = torch.unsqueeze(label_emo, 0)
       
   
          
            # calculate loss
            loss = criterion(prediction, label_emo)

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
          

dataset = EmotionDataset('data/pavoque/pavoque_all_500_dur_7_5_norm_0to1.json')
evaluation, train1, train2 = create_train_test(dataset)
num_classes = 5
model = EmotionClassificationNet(1,64,2,8,60,num_classes,2,26)

print(train(model, 1, train1))


#labels_gen_map = {'m': 0, 'f': 1}
#labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample1[2]])
#labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in samp1[2]])
#labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])
#labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])