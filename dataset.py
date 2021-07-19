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
        df.columns = ['features', 'emotion_label']
        # convert all features from list to np.array
        df["features"] = df["features"].apply(lambda x: np.array(x))

        self.spectrogram = df["features"]
        self.emotion_label = df["emotion_label"]
        #self.gender_label = df["gender_label"]

    def __len__(self):
        return len(self.emotion_label)  # access with len(name_of_dataset)

    def __getitem__(self, idx):
        # access with name_of_dataset[idx]
        return self.spectrogram[idx], self.emotion_label[idx]


def create_train_test(dataset):
    # train/test ratio
    split_train = int(len(dataset) * 9/10)
    split_test = len(dataset) - split_train

   

    # split dataset in train and test
    train, test = random_split(dataset=dataset, lengths=[
                               split_train, split_test])
    
    split1 = int(len(train) * 1/2)
    split2 = len(train) - split1
    
    train1, train2 = random_split(dataset=train, lengths=[
                               split1, split2])

    # load train and test data to batches and shuffle
    query_dataloader = torch.utils.data.DataLoader(test,
                                                   batch_size=1,
                                                   shuffle=False)
    
    train_dataloader1 = torch.utils.data.DataLoader(train,
                                                   batch_size=1,
                                                   shuffle=True)
    train_dataloader2 = torch.utils.data.DataLoader(train,
                                                   batch_size=1,
                                                   shuffle=True)
   

    return query_dataloader, train_dataloader1, train_dataloader2



#dataset = EmotionDataset('data/pavoque/pavoque_across_500.json')
#print(dataset[0])
#print(create_train_test(dataset))
