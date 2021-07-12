import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataset import random_split


class EmotionDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_json(file_name)
        df = df.T

        # name columns
        df.columns = ['features', 'emotion_label', 'gender_label']
        # convert all features from list to np.array
        df["features"] = df["features"].apply(lambda x: np.array(x))

        self.spectrogram = df["features"]
        self.emotion_label = df["emotion_label"]
        self.gender_label = df["gender_label"]

    def __len__(self):
        return len(self.emotion_label)  # access with len(name_of_dataset)

    def __getitem__(self, idx):
        # access with name_of_dataset[idx]
        return self.spectrogram[idx], self.emotion_label[idx], self.gender_label[idx]


def create_train_test(dataset):
    # train/test ratio
    split_train = int(len(dataset) * 2/3)
    split_test = len(dataset) - split_train

    # split dataset in train and test
    train, test = random_split(dataset=dataset, lengths=[
                               split_train, split_test])

    # load train and test data to batches and shuffle
    train_dataloader = torch.utils.data.DataLoader(train,
                                                   batch_size=32,
                                                   shuffle=True)

    test_dataloader =  torch.utils.data.DataLoader(test,batch_size=32,shuffle=False)

    return train_dataloader, test_dataloader


##dataset = EmotionDataset('data/pavoque/sad.json')
#print(dataset[0])
