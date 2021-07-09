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
            self.input_spec_size, self.cnn_filter_size, kernel_size=3, stride=1), nn.MaxPool1d(3), nn.ReLU(inplace=True))
        self.cnn_layer2 = nn.Sequential(nn.Conv1d(
            self.cnn_filter_size, self.cnn_filter_size, kernel_size=3, stride=1),  nn.MaxPool1d(3), nn.ReLU(inplace=True))

        ###
        self.embedding = nn.Linear(self.n_mels, 1)
        self.soft = nn.Softmax()
        self.lstm = nn.LSTM(input_size=self.cnn_filter_size, hidden_size=self.hidden_size_lstm,
                            num_layers=self.num_layers_lstm, bidirectional=True, dropout=0.5, batch_first=True)
        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size_lstm*2, dim_feedforward=512, nhead=self.num_heads_self_attn)
        self.gender_layer = nn.Linear(
            self.hidden_size_lstm*4, self.num_gender_class)  # , nn.ReLU(inplace=True))#, #nn.Linear(self.n_mels, self.embedding_size))
        self.emotion_layer = nn.Sequential(nn.Linear(
            self.hidden_size_lstm*4, self.embedding_size), nn.ReLU(inplace=True))

    def forward(self, inputs):
        out = self.cnn_layer1(inputs)
        out = self.cnn_layer2(out)
        #h_0 = Variable(torch.randn(self.num_layers_lstm,inputs.shape[0],self.hidden_size_lstm ))
        #c_0 = Variable(torch.randn(self.num_layers_lstm,inputs.shape[0],self.hidden_size_lstm ))

        out = out.permute(0, 2, 1)
        out, (final_hidden_state, final_cell_state) = self.lstm(out)
        out = self.encoder_layer(out)
        mean = torch.mean(out, 1)
        std = torch.std(out, 1)
        stat = torch.cat((mean, std), 1)
        pred_gender = self.gender_layer(stat)
        pred_emo = self.emotion_layer(stat)
        pred_emo = pred_emo.permute(1, 0)
        pred_emo = self.embedding(pred_emo)
        return pred_emo, pred_gender

    '''
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
    '''
