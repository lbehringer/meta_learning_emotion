import os
import json
import jsonmerge
import argparse
from input_features import audio_to_spectrogram
from label_dataset import create_json


def create_iemocap_label_spectrograms_json():
    """This creates one separate file with spectrograms for each emotion"""

    with open('data/iemocap/path2utt.txt') as json_file:
        path2utt = json.load(json_file)
    with open('data/iemocap/utt2label.txt') as json_file:
        utt2label = json.load(json_file)
    with open('data/iemocap/utt2path.txt') as json_file:
        utt2path = json.load(json_file)



    # get list of IEMOCAP_labels
    label_list = os.listdir('IEMOCAP_labels')
    print(label_list)
    # iterate over all emotions one by one, open respective file containing the relevant utterances
    for label in label_list:
        spectrograms = dict()
        deleted_files = 0
        i = 0
        max_num = 500

        with open(os.path.join('IEMOCAP_labels/', label), 'r') as f:
            lines = f.readlines()
        file_list = list()  # this contains all utt id's, e.g. "Ses01F_impro01_F012\n" (WITHOUT ".wav" at the end)
        for line in lines:
            line = line.rstrip('\n') + '.wav'  # remove \n, add file format
            file_list.append(utt2path[line])  # for each utt id, add respective filepath to file_list
        for filepath in file_list:
            file = path2utt[filepath]  # get file (utt id + ".wav")
            gender = file[-8].lower()  # get "m" or "f" from utt id, e.g. get "f" from "Ses01F_impro01_F012.wav"
            try:
                single_spectrogram = dict()
                single_spectrogram['features'] = audio_to_spectrogram(
                    filepath,
                    args.offset, args.duration, args.n_mels).numpy().tolist()
                single_spectrogram['target'] = label
                single_spectrogram['gender'] = gender
                spectrograms[file] = single_spectrogram

                i += 1
                if i >= max_num:
                    break
                # spectrograms.append(audio_to_spectrogram(
                #        filepath,
                #        args.offset, args.duration, args.n_mels).numpy())
            except KeyError as e:
                deleted_files += 1
                with open("data/iemocap/logs/log_" + file + ".log", "a") as f:
                    f.writelines(file + ": " + str(e))
            except ValueError as e:
                deleted_files += 1
                with open("data/iemocap/logs/log_" + file + ".log", "a") as f:
                    f.writelines(file + ": " + str(e))

        out_file = 'data/iemocap/iemocap_' + label + '_' + str(max_num) + '_dur_' + str(args.duration) + '_spectrograms.json'
        with open(out_file, 'w') as out:
            json.dump(spectrograms, out)


def get_iemocap_spectrograms():
    """This iterates over the iemocap wav files (not in label-specific order!), creates a spectrogram file and returns
    the spectrograms dict"""
    with open('data/iemocap/path2utt.txt') as json_file:
        path2utt = json.load(json_file)
    with open('data/iemocap/utt2label.txt') as json_file:
        utt2label = json.load(json_file)

    spectrograms = dict()
    deleted_files = 0
    i = 0
    num_dirs = len(os.listdir('data/iemocap/wav'))

    for dir in os.listdir('data/iemocap/wav/'):
        if os.path.isdir('data/iemocap/wav/' + dir):
            dir_path = 'data/iemocap/wav/' + dir + '/'
            for file in os.listdir(dir_path):
                try:
                    filepath = dir_path + file
                    # get label from path2utt --> utt2label
                    utt = path2utt[filepath]
                    label = utt2label[utt]
                    try:
                        single_spectrogram = dict()
                        single_spectrogram['features'] = audio_to_spectrogram(
                                filepath,
                                args.offset, args.duration, args.n_mels).numpy().tolist()
                        single_spectrogram['target'] = label
                        spectrograms[file] = single_spectrogram

                        i += 1
                        if i >= 2000:
                            break
                        #spectrograms.append(audio_to_spectrogram(
                        #        filepath,
                        #        args.offset, args.duration, args.n_mels).numpy())
                    except KeyError as e:
                        deleted_files += 1
                        with open("data/iemocap/logs/log_" + file + ".log", "a") as f:
                            f.writelines(file + ": " + str(e))
                    except ValueError as e:
                        deleted_files += 1
                        with open("data/iemocap/logs/log_" + file + ".log", "a") as f:
                            f.writelines(file + ": " + str(e))
                except KeyError as e:
                    with open("data/iemocap/logs/log_" + file + ".log", "a") as f:
                        f.writelines(file + ": " + str(e))

    out_file = 'data/iemocap/iemocap_2k_spectrograms.json'
    with open(out_file, 'w') as out:
        json.dump(spectrograms, out)
    return spectrograms


def create_iemocap_json(spectrograms):
    data = {}
    # for each utt, append dict with "features": <features>, "target": <target>
    i = 0
    for sample in (spectrograms):
        data[sample] = 'test'
        print(spectrograms[sample]['spectrogram'])
        print(spectrograms[sample]['label'])
        i += 1
        if i == 2000:
            break
    #     else:
    #         data[emotion + '_' + str(i)] = {"features": spectrograms[i].tolist(), "target": emotion, "gender": "m"}
    #
    # out_file = 'data/pavoque/7_digit/' + emotion + filename_suffix + '.json'
    # with open(out_file, 'w') as out:
    #     json.dump(data, out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0, required=False,
                        type=int, help="offset in seconds")
    parser.add_argument("--duration", default=4, required=False,
                        type=int, help="window size in seconds")
    parser.add_argument("--n_mels", default=26, required=False,
                        type=int, help="number of filterbanks")
    parser.add_argument("--embedding_size", default=50,
                        required=False, type=int, help="embedding size for emotion embeddings")

    args = parser.parse_args()

    #emotion = 'pok'
    #filename_suffix = '_500_dur_7_5'
    #spectrograms = get_iemocap_spectrograms()
    #print(spectrograms)
    #write_emo_csv('data/pavoque/' + emotion + '.csv', 'data/pavoque/' + emotion + '/', spectrograms, emotion,
    # deleted_files)
    #create_iemocap_json(spectrograms)
    #merge_json()
    create_iemocap_label_spectrograms_json()
