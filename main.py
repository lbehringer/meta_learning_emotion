import argparse
import numpy as np
from input_features import audio_to_spectrogram, load_dvector, concatenate_features
from model import CNN_BLSTM_SELF_ATTN
from dataset import EmotionDataset, create_train_test
import torch
from torch.utils.data.dataset import random_split


def main(args):
    # load data, split into train & test
    dataset = EmotionDataset(
        '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/data/pavoque/ang_wav.csv')
    train_dataloader, test_dataloader = create_train_test(dataset)
    for item in train_dataloader:
        print(item)

    model = CNN_BLSTM_SELF_ATTN(args.input_spec_size, args.cnn_filter_size, args.num_layers_lstm,
                                args.num_heads_self_attn, args.hidden_size_lstm, args.num_emo_classes, args.num_gender_class, args.embedding_size, args.n_mels)

    # TBD (training, evaluation)

    # test
    # features = audio_to_spectrogram(
    #    '/mount/arbeitsdaten/studenten1/advanced_ml/dengelva/meta_learning_emotion/Emotional_Speech_Dataset_Singapore/0001/Angry/0001_000657.wav', args.offset, args.duration, args.n_mels)
    #preds_emo, preds_gender = model(features.unsqueeze(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0, required=False,
                        type=int, help="offset in seconds")
    parser.add_argument("--duration", default=2, required=False,
                        type=int, help="window size in seconds")
    parser.add_argument("--n_mels", default=26, required=False,
                        type=int, help="number of filterbanks")
    parser.add_argument("--input_spec_size", default=1, required=False,  # in_channels ?
                        type=int, help="number of convolution filters")
    parser.add_argument("--cnn_filter_size", default=64, required=False,
                        type=int, help="number of convolution filters")
    parser.add_argument("--num_layers_lstm", default=2,
                        required=False, type=int, help="number of lstm layers")
    parser.add_argument("--hidden_size_lstm", default=60, required=False,
                        type=int, help="number of LSTM hidden units")
    parser.add_argument("--num_heads_self_attn", default=8,
                        required=False, type=int, help="number of attention heads")
    parser.add_argument("--num_emo_classes", default=4,
                        required=False, type=int, help="emotion classes")
    parser.add_argument("--num_gender_class", default=2,
                        required=False, type=int, help="gender classes --> m & f")
    parser.add_argument("--embedding_size", default=50,
                        required=False, type=int, help="embedding size for emotion embeddings")

    args = parser.parse_args()

    main(args)
