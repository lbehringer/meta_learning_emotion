import numpy as np
import jsonmerge
import os
import csv
import json
import argparse
import sys
from input_features import audio_to_spectrogram
from input_features import get_spectrograms
np.set_printoptions(threshold=sys.maxsize)


# def write_emo_csv_wav(csv_file, dir, target):
#     with open(csv_file, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['utterance'] + ['emotion'])
#         for file in os.listdir(dir):
#             writer.writerow([file] + [target])
#
#
# def write_emo_csv(csv_file, dir, spectrograms, target, deleted_files):
#     with open(csv_file, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['utterance'] + ['emotion'])
#         num_files = len(os.listdir(dir))
#         for i in range(num_files-deleted_files):
#             writer.writerow([[spectrograms[i]]] + [target])


def create_json(spectrograms, emotion):
    data = {}
    # for each utt, append dict with "features": <features>, "target": <target>
    for i in range(len(spectrograms)):
        if i == max_retrieved_samples:
            break
        else:
            data[emotion + '_' + str(i)] = {"features": spectrograms[i].tolist(), "target": emotion, "gender": "m"}

    out_file = 'data/pavoque/7_digit/' + emotion + filename_suffix + '_norm_0to1.json'
    with open(out_file, 'w') as out:
        json.dump(data, out)


def merge_json():
    # with open("data/pavoque/7_digit/pok" + filename_suffix + "_norm_0to1.json", 'r') as f:
    #     emo = json.load(f)
    # with open("data/pavoque/7_digit/pavoque_across" + filename_suffix + "_norm_0to1.json", 'r') as f:
    #     merged = json.load(f)
    #     merged = jsonmerge.merge(emo, merged)
    # with open("data/pavoque/7_digit/pavoque_all" + filename_suffix + "_norm_0to1.json", 'w+') as f:
    #     json.dump(merged, f)

    with open("data/iemocap/iemocap_sad" + filename_suffix + ".json", 'r') as f:
        emo = json.load(f)
    with open("data/iemocap/iemocap_across" + filename_suffix + ".json", 'r') as f:
        merged = json.load(f)
        merged = jsonmerge.merge(emo, merged)
    with open("data/iemocap/iemocap_across" + filename_suffix + ".json", 'w+') as f:
        json.dump(merged, f)


def get_all_spectrograms(emotion):
    spectrograms = []
    deleted_files = 0
    i = 0
    for file in os.listdir('data/pavoque/7_digit/' + emotion):
        try:
            if i == max_retrieved_samples:
                break
            spectrograms.append(audio_to_spectrogram(
                    'data/pavoque/7_digit/' + emotion + '/' + file,
                    args.offset, args.duration, args.n_mels).numpy())
            # spectrograms.append(get_spectrograms('data/pavoque/7_digit/' + emotion + '/' + file,
            #                                      ref_db=20, max_db=100, n_fft=2048, sr=16000,
            #                                      n_mels=80, preemphasis=.97, duration=4)[0])
            i += 1
        except ValueError as e:
            deleted_files += 1
            with open("data/pavoque/logs/log_" + emotion + ".log", "a") as f:
                f.writelines(file + ": " + str(e))
    return spectrograms, deleted_files


def get_support_set_spectrograms(emotion):
    spectrograms = []
    deleted_files = 0
    i = 0
    for file in os.listdir('data/pavoque/7_digit/' + emotion):
        i += 1
        if i > max_retrieved_samples:
            try:
                if i > max_retrieved_samples + 1:
                    break
                spectrograms.append(audio_to_spectrogram(
                        'data/pavoque/7_digit/' + emotion + '/' + file,
                        args.offset, args.duration, args.n_mels).numpy())
                # spectrograms.append(get_spectrograms('data/pavoque/7_digit/' + emotion + '/' + file,
                #                                      ref_db=20, max_db=100, n_fft=2048, sr=16000,
                #                                      n_mels=80, preemphasis=.97, duration=4)[0])
            except ValueError as e:
                deleted_files += 1
                with open("data/pavoque/logs/log_" + emotion + ".log", "a") as f:
                    f.writelines(file + ": " + str(e))
    return spectrograms, deleted_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0, required=False,
                        type=int, help="offset in seconds")
    parser.add_argument("--duration", default=4, required=False,
                        type=int, help="window size in seconds")
    parser.add_argument("--n_mels", default=26, required=False,
                        type=int, help="number of filterbanks")

    args = parser.parse_args()

    max_retrieved_samples = 500
    emotion = 'sad'
    dur = '_dur_' + str(args.duration)
    filename_suffix = '_' + str(max_retrieved_samples) + dur + '_spectrograms'
    #spectrograms, deleted_files = get_all_spectrograms(emotion)
    #spectrograms, deleted_files = get_support_set_spectrograms(emotion)
    #write_emo_csv('data/pavoque/' + emotion + '.csv', 'data/pavoque/' + emotion + '/', spectrograms, emotion,
    # deleted_files)
    #create_json(spectrograms, emotion)
    merge_json()


    #print(spectrograms[0:5])

