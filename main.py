import argparse
import numpy as np
from model import CNN_BLSTM_SELF_ATTN, Siamese
from dataset import *
import torch
from classification_model import EmotionClassificationNet
from torch.utils.data.dataset import random_split
from training import train, evaluate
from embeddings2json import write_embeddings
from tsne import tsne


def main(args):
    torch.manual_seed(99)
    # load dataset, split into train & test
    dataset = EmotionDataset(pavoque_all)
    support_data = EmotionDataset(pavoque_all_support)

    query_set, train1, train2 = create_train_test(dataset)
    support_set = torch.utils.data.DataLoader(support_data, batch_size=1,
                                              shuffle=False)

    # choose model
    model = CNN_BLSTM_SELF_ATTN(args.input_spec_size, args.cnn_number_filters, args.num_layers_lstm,
                                args.num_heads_self_attn, args.hidden_size_lstm, args.num_emo_classes, args.embedding_size, args.n_mels)
    #model = Siamese()
    #model = EmotionClassificationNet(26, 64, 2, 8, 20, 4, 26)

    # train & evaluate model
    trained_model_name = "state_dict_model_" + args.file_name + ".pt"
    if args.train:
        train(model, args.num_epochs, train1, train2,
              query_set, support_set, trained_model_name)

    if args.evaluate:
        confusion_file = "confusion_" + args.file_name + ".png"
        evaluate(model, support_set, query_set,
                 trained_model_name, confusion_file)

    # get embeddings & visualize with tsne
    embeddings_file = "embeddings_" + args.file_name + ".json"
    tsne_plot_file = "tsne_plot_" + args.file_name + ".png"
    if args.embeddings2file:
        # , map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(trained_model_name))
        write_embeddings(model, query_set, embeddings_file)
        tsne(embeddings_file, tsne_plot_file, args.embedding_size, 15, 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0, required=False,
                        type=int, help="offset in seconds")
    parser.add_argument("--duration", default=4, required=False,
                        type=int, help="window size in seconds")
    parser.add_argument("--n_mels", default=26, required=False,
                        type=int, help="number of filterbanks")
    parser.add_argument("--input_spec_size", default=26, required=False,  # in_channels
                        type=int, help="number of spectrogram features")
    parser.add_argument("--cnn_number_filters", default=64, required=False,
                        type=int, help="number of convolution filters")
    parser.add_argument("--num_layers_lstm", default=2,
                        required=False, type=int, help="number of lstm layers")
    parser.add_argument("--hidden_size_lstm", default=40, required=False,
                        type=int, help="number of LSTM hidden units")
    parser.add_argument("--num_heads_self_attn", default=8,
                        required=False, type=int, help="number of attention heads")
    parser.add_argument("--num_emo_classes", default=4,
                        required=False, type=int, help="emotion classes")
    parser.add_argument("--num_gender_class", default=2,
                        required=False, type=int, help="gender classes --> m & f")
    parser.add_argument("--embedding_size", default=150,
                        required=False, type=int, help="embedding size for emotion embeddings")
    parser.add_argument("--num_epochs", default=1200,
                        required=False, type=int, help="num_epochs")
    parser.add_argument("--embeddings2file", default=True,
                        required=False, type=bool, help="write embeddings to file?")
    parser.add_argument("--train", default=True,
                        required=False, type=bool, help="train model?")
    parser.add_argument("--evaluate", default=True,
                        required=False, type=bool, help="evaluate model?")
    parser.add_argument("--file_name", default="meta_pavoque_all_1200ep_emb150_batch32",
                        required=False, type=str, help="file name for model, embedding and plot file")

    args = parser.parse_args()

    main(args)
