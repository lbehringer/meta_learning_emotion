import argparse
import numpy as np
from input_features import audio_to_spectrogram, load_dvector, concatenate_features
from model import CNN_BLSTM_SELF_ATTN, Siamese
from dataset import EmotionDataset, create_train_test
import torch
from torch.utils.data.dataset import random_split
from training import train, evaluate
from embeddings2json import get_embeddings


def main(args):
    # load dataset, split into train & test
    dataset = EmotionDataset('data/pavoque/pavoque_across_500_dur_7_5.json')
    support_data = EmotionDataset('data/pavoque/support_0.json')

    query_set, train1, train2 = create_train_test(dataset)
    support_set = torch.utils.data.DataLoader(support_data,batch_size=1,
                                                   shuffle= False)

   

    model = CNN_BLSTM_SELF_ATTN(args.input_spec_size, args.cnn_filter_size, args.num_layers_lstm,
                                args.num_heads_self_attn, args.hidden_size_lstm, args.num_emo_classes, args.num_gender_class, args.embedding_size, args.n_mels)

    #model.cuda()

    # train & evaluate model
    if args.train:
        train(model, args.num_epochs, train1, train2)

    if args.evaluate:
        evaluate(model, support_set, query_set, "state_dict_model_7_5.pt")

    # get embeddings
    if args.embeddings2file:
        get_embeddings(model, "state_dict_model_7_5.pt", query_set)


    # TBD visualization (t-sne, PCA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0, required=False,
                        type=int, help="offset in seconds")
    parser.add_argument("--duration", default=7.5, required=False,
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
    parser.add_argument("--embedding_size", default=150,
                        required=False, type=int, help="embedding size for emotion embeddings")
    parser.add_argument("--num_epochs", default=120,
                        required=False, type=int, help="num_epochs")
    parser.add_argument("--embeddings2file", default=False,
                        required=False, type=bool, help="write embeddings to file?")
    parser.add_argument("--train", default=False,
                        required=False, type=bool, help="train model?")
    parser.add_argument("--evaluate", default=True,
                        required=False, type=bool, help="evaluate model?")

    args = parser.parse_args()

    main(args)
