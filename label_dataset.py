import numpy
import numpy as np
import pandas as pd
import os
import csv
import json
import argparse
import sys
from input_features import audio_to_spectrogram
numpy.set_printoptions(threshold=sys.maxsize)


def write_emo_csv_wav(csv_file, dir, target):
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['utterance'] + ['emotion'])
        for file in os.listdir(dir):
            writer.writerow([file] + [target])


def write_emo_csv(csv_file, dir, spectrograms, target, deleted_files):
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['utterance'] + ['emotion'])
        num_files = len(os.listdir(dir))
        for i in range(num_files-deleted_files):
            writer.writerow([[spectrograms[i]]] + [target])


def create_json(spectrograms, emotion):
    data = {}
    # for each utt, append dict with "features": <features>, "target": <target>
    for i in range(len(spectrograms)):
        data[emotion + '_' + str(i)] = {"features": spectrograms[i].tolist(), "target": emotion, "gender": "m"}

    out_file = 'data/pavoque/' + emotion + '.json'
    with open(out_file, 'w') as out:
        json.dump(data, out)



def get_spectrograms(emotion):
    spectrograms = []
    deleted_files = 0
    for file in os.listdir('data/pavoque/' + emotion):
        try:
            spectrograms.append(audio_to_spectrogram(
                    'data/pavoque/' + emotion + '/' + file,
                    args.offset, args.duration, args.n_mels).numpy())
        except ValueError as e:
            deleted_files += 1
            with open("data/pavoque/logs/log_" + emotion + ".log", "w") as f:
                f.writelines(file + ": " + str(e))
    return spectrograms, deleted_files


#write_emo_csv('data/pavoque/sad.csv', 'data/pavoque/sad', 'sad')
#df_ang = pd.read_csv('data/pavoque/hap.csv')
#print(df_ang.head())

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

    emotion = 'sad'
    spectrograms, deleted_files = get_spectrograms(emotion)
    #write_emo_csv('data/pavoque/' + emotion + '.csv', 'data/pavoque/' + emotion + '/', spectrograms, emotion,
    # deleted_files)
    create_json(spectrograms, emotion)


    #print(spectrograms[0:5])

    #
    # with open('data/pavoque/ang.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['utterance'] + ['emotion'])
    #     #num_files = len(os.listdir('data/pavoque/ang'))
    #     #for i in range(num_files):
    #     writer.writerow([spectrograms] + ['ang'])