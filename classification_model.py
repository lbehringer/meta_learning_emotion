import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class EmotionClassificationNet(torch.nn.Module):
    def __init__(self, input_spec_size, cnn_filter_size, num_layers_lstm, num_heads_self_attn, hidden_size_lstm, num_emo_classes, num_gender_class,  n_mels):
        super(EmotionClassificationNet, self).__init__()

        self.input_spec_size = input_spec_size
        self.cnn_filter_size = cnn_filter_size
        self.num_layers_lstm = num_layers_lstm
        self.num_heads_self_attn = num_heads_self_attn
        self.hidden_size_lstm = hidden_size_lstm
        self.num_emo_classes = num_emo_classes
        self.num_gender_class = num_gender_class
        self.n_mels = n_mels

        self.cnn_layer1 = nn.Sequential(nn.Conv1d(
            self.input_spec_size, self.cnn_filter_size, kernel_size=3, stride=1), nn.MaxPool1d(3), nn.ReLU(inplace=True))
        self.cnn_layer2 = nn.Sequential(nn.Conv1d(
            self.cnn_filter_size, self.cnn_filter_size, kernel_size=3, stride=1),  nn.MaxPool1d(3), nn.ReLU(inplace=True))

        ###
        self.out = nn.Sequential(nn.Linear(self.n_mels, 1))  # , nn.Softmax(0))
        self.lstm = nn.LSTM(input_size=self.cnn_filter_size, hidden_size=self.hidden_size_lstm,
                            num_layers=self.num_layers_lstm, bidirectional=True, dropout=0.5, batch_first=True)
        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size_lstm*2, dim_feedforward=512, nhead=self.num_heads_self_attn)
        self.gender_layer = nn.Linear(
            self.hidden_size_lstm*4, self.num_gender_class)  # , nn.ReLU(inplace=True))#, #nn.Linear(self.n_mels, self.embedding_size))
        self.emotion_layer = nn.Sequential(nn.Linear(
            self.hidden_size_lstm*4, self.num_emo_classes), nn.ReLU(inplace=True))

    def forward(self, input):  # input shape = (batch_size, channels, spec_rows, spec_columns)
        inputs = input.permute(1, 0, 2)
        out = self.cnn_layer1(inputs)
        out = self.cnn_layer2(out)
        out = out.permute(0, 2, 1)
        out, (final_hidden_state, final_cell_state) = self.lstm(out)
        out = self.encoder_layer(out)
        mean = torch.mean(out, 1)
        std = torch.std(out, 1)
        stat = torch.cat((mean, std), 1)
        #pred_gender = self.gender_layer(stat)
        out = self.emotion_layer(stat)
        out = out.permute(1, 0)
        out = self.out(out)
        out = torch.flatten(out)

        return out
