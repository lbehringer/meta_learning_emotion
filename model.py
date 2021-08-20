"""
Created on Wed Mar  4 20:08:19 2020
@modified from: https://github.com/KrishnaDN/speech-emotion-recognition-using-self-attention.git 
CNN: RELU activation function added
Output layer: returns emotion embedding of defined size
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class CNN_BLSTM_SELF_ATTN(torch.nn.Module):
    def __init__(self, input_spec_size, cnn_filter_size, num_layers_lstm, num_heads_self_attn, hidden_size_lstm, num_emo_classes, num_gender_class, embedding_size, n_mels):
        super(CNN_BLSTM_SELF_ATTN, self).__init__()

        self.input_spec_size = input_spec_size
        self.cnn_filter_size = cnn_filter_size
        self.num_layers_lstm = num_layers_lstm
        self.num_heads_self_attn = num_heads_self_attn
        self.hidden_size_lstm = hidden_size_lstm
        self.num_emo_classes = num_emo_classes
        self.num_gender_class = num_gender_class
        self.embedding_size = embedding_size
        self.n_mels = n_mels

        self.cnn_layer1 = nn.Sequential(nn.Conv1d(
            self.input_spec_size, self.cnn_filter_size, kernel_size=3, stride=1), nn.MaxPool1d(2), nn.ReLU(inplace=True))
        self.cnn_layer2 = nn.Sequential(nn.Conv1d(
            self.cnn_filter_size, self.cnn_filter_size, kernel_size=3, stride=1),  nn.MaxPool1d(2), nn.ReLU(inplace=True))

        ###
        self.embedding = nn.Linear(self.n_mels, 1)
        self.lstm = nn.LSTM(input_size=self.cnn_filter_size, hidden_size=self.hidden_size_lstm,
                            num_layers=self.num_layers_lstm, bidirectional=True, dropout=0.5, batch_first=True)
        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size_lstm*2, dim_feedforward=5, nhead=self.num_heads_self_attn)
        self.gender_layer = nn.Linear(
            self.hidden_size_lstm*4, self.num_gender_class)  # , nn.ReLU(inplace=True))#, #nn.Linear(self.n_mels, self.embedding_size))
        self.emotion_layer = nn.Sequential(nn.Linear(
            self.hidden_size_lstm*4, self.embedding_size), nn.ReLU(inplace=True))

    def forward_once(self, input):  # input shape = (batch_size, channels, spec_rows, spec_columns)
        #inputs = input.permute(1,0,2)
        #out = self.cnn_layer1(inputs)
        out = self.cnn_layer1(input) # comment this instead of one line above
        out = self.cnn_layer2(out)
        out = out.permute(0, 2, 1)
        out, (final_hidden_state, final_cell_state) = self.lstm(out)
        out = self.encoder_layer(out)
        mean = torch.mean(out, 1)
        std = torch.std(out, 1)
        stat = torch.cat((mean, std), 1)
        #pred_gender = self.gender_layer(stat)
        embedding_emo = self.emotion_layer(stat)
        embedding_emo = embedding_emo.permute(1, 0)
        embedding_emo = self.embedding(embedding_emo)
        embedding_emo = torch.flatten(embedding_emo)

        return embedding_emo


    def forward(self, input1, input2):
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        # out = self.out(torch.abs(output1-output2))
        # return out.view(out.size())
        return embedding1, embedding2




class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 64@48*48
            nn.Conv1d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool1d(2),   # 128@21*21
            nn.Conv1d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool1d(2), # 128@9*9
            nn.Conv1d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(3328, 1)

    def forward_once(self, x):
        x = x.permute(1,0,2)
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        #print(x)
        #x = self.liner(x)
        x = self.out(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out1 = torch.flatten(out1)
        out2 = torch.flatten(out2)
        #dis = torch.abs(out1 - out2)
        #out = self.out(dis)
        #  return self.sigmoid(out)
        return out1, out2

'''
# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
'''
