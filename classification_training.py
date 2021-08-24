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
from dataset import EmotionDataset, create_classification_set, create_train_test
from classification_model import EmotionClassificationNet
import matplotlib.pyplot as plt


def classification_training(model, num_epochs, dataloader_train, PATH):
    # model.load_state_dict(torch.load('state_dict_model_classification_iemocap.pt')) # continue training with model
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
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    train_loss_list = []
    for epoch in range(num_epochs):
        for i, sample in enumerate(dataloader_train):
            # categorical to numeric labels
            label_emo = torch.Tensor(
                [int(labels_emo_map[label]) for label in sample[1]])
            # print(type(label_emo))
            label_emo = label_emo.type(torch.LongTensor)
            label_emo =  label_emo.to(device)
            spec_features = sample[0]
            spec_features = spec_features.to(device, dtype=torch.float)
#            spec_features1, spec_features2, label_emo1, label_emo2 = spec_features1.to(
#                device, dtype=torch.float), spec_features2.to(
#                device, dtype=torch.float), label_emo1.to(device), label_emo2.to(device)
            spec_features.requires_grad = True

            # clear the gradients
            optimizer.zero_grad()

            # model output
            prediction = model(spec_features)

            # calculate loss
            loss = criterion(prediction, label_emo)
            #print(f'loss: {loss}')

            loss.backward()
            # for name, param in model.named_parameters():
            #    print(name, param)
            # update model weights
            optimizer.step()

            train_loss_list.append(loss.item())

            if i % 500 == 0:
                print('Loss {} after {} iterations'.format(
                    np.mean(np.asarray(train_loss_list)), i))

                mean_loss = np.mean(np.asarray(train_loss_list))
                plt.plot(mean_loss)
                plt.savefig('classification_loss.png')

        print('Loss {} after {} epochs'.format(
            np.mean(np.asarray(mean_loss)), epoch))

        #PATH = "state_dict_model_classification_iemocap_500ep_20_08_21_11_28am.pt"

        torch.save(model.state_dict(), PATH)


def evaluate(model, test, PATH):
    labels_emo_map = {'sad': 0, 'ang': 1, 'hap': 2, 'neu': 3, 'pok': 4}

    with torch.no_grad():
        model.load_state_dict(torch.load(PATH))
        model.double()
        model.eval()
        #loss = ContrastiveLoss(2)
        y_pred = list()
        y_true = list()
        soft = torch.nn.Softmax(dim=None)

        for i, samp in enumerate(test):
            #print(f'query: {samp1}')
            label_emo = torch.FloatTensor(
                [float(labels_emo_map[label]) for label in samp[1]])
            y_true.append(label_emo.item())
            #y_true.append(label_emo)
            spec_test = samp[0]

            prediction = model(spec_test)
            print(prediction)
            prediction = torch.argmax(prediction)
            # print(type(prediction))
            print(prediction)

            y_pred.append(prediction)

        print(f'predictions: {y_pred}')
        print(f'gold label: {y_true}')
        score = f1_score(y_true, y_pred, average='macro')
        print(f'f_score: {score}')

        x_axis_labels = ['sad', 'ang', 'hap', 'neu'] 
        y_axis_labels = ['sad', 'ang', 'hap', 'neu']

        #Get the confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        print(cf_matrix)
        svm = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        figure = svm.get_figure()    
        figure.savefig(confusion_file, dpi=400)


model_path = 'state_dict_model_classification_iemocap_300ep_batch8.pt'
dataset = EmotionDataset(
    '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2021/student_directories/Lyonel_Behringer/advanced-ml/iemocap_across_500_dur_4_spectrograms.json')
train, test = create_classification_set(dataset)
num_classes = 4
model = EmotionClassificationNet(26, 64, 2, 8, 20, num_classes, 26)
# current best params --> EmotionClassificationNet(26, 1, 2, 10, 20, num_classes, 26)
print(classification_training(model, 300, train, model_path))
print(evaluate(model, test, model_path))

#labels_gen_map = {'m': 0, 'f': 1}
#labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample1[2]])
#labels_gen1 = torch.FloatTensor([float(labels_gen_map[label]) for label in samp1[2]])
#labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])
#labels_gen2 = torch.FloatTensor([float(labels_gen_map[label]) for label in sample2[2]])
