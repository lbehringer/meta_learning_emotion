import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataset import random_split


iemocap_across = '/mount/arbeitsdaten/studenten1/team-lab-phonetics/2021/student_directories/Lyonel_Behringer/advanced-ml/iemocap_across_500_dur_4_spectrograms.json'
iemocap_across_support = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/iemocap/iemocap_across_502_dur_4_support.json'
pavoque_across = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/pavoque/pavoque_across_500_dur_4_preemph_norm_0to1.json'
pavoque_across_support = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/pavoque/pavoque_across_500_dur_4_preemph_support_norm_0to1.json'
pavoque_all = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/pavoque/pavoque_all_500_dur_4_preemph_norm_0to1.json'
pavoque_all_support = ""
singapore_emo_en = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/merged_en.json'
singapore_emo_zh = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/merged_zh.json'
singapore_emo_en_support = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/support_merged_en.json'
singapore_emo_zh_support = '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/support_merged_zh.json'


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


def create_classification_set(dataset):
    split_train = int(len(dataset) * 9/10)
    split_test = len(dataset) - split_train

    train, test = random_split(dataset=dataset, lengths=[
                               split_train, split_test])
    test_dataloader = torch.utils.data.DataLoader(test,
                                                  batch_size=1,
                                                  shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(train,
                                                   batch_size=16,
                                                   shuffle=True)

    return train_dataloader, test_dataloader


def create_train_test(dataset):
    # train/test ratio
    split_train = int(len(dataset) * 9/10)
    split_test = len(dataset) - split_train

    # split dataset in train and test
    train, test = random_split(dataset=dataset, lengths=[
                               split_train, split_test])

    # load train and test data to batches and shuffle
    query_dataloader = torch.utils.data.DataLoader(test,
                                                   batch_size=1,
                                                   shuffle=False)

    train_dataloader1 = torch.utils.data.DataLoader(train,
                                                    batch_size=32,
                                                    shuffle=True)
    train_dataloader2 = torch.utils.data.DataLoader(train,
                                                    batch_size=32,
                                                    shuffle=True)

    return query_dataloader, train_dataloader1, train_dataloader2
