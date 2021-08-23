import os
import json
import jsonmerge
import argparse
from input_features import audio_to_spectrogram
from label_dataset import create_json


def create_singapore_spectrograms_json():
    """This creates one separate file with spectrograms for each emotion"""
    languages = ['en', 'zh']
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    genders = ['f', 'm']
    for language in languages:
        for emotion in emotions:
            label = emotion[:3].lower()
            max_num = 500  # this should be divisible by 10
            max_num_per_gender = int(max_num/2)  # this should be divisible by 5
            spectrograms = dict()
            deleted_files = 0
            num_created_specs = 0
            for gender in genders:
                path = os.path.join("data/ESD_Singapore/", language, gender)
                speakers = os.listdir(path)
                num_speakers = len(speakers)
                max_num_per_speaker = max_num_per_gender / num_speakers  # e.g. for max_num=500 --> 50 per speaker
                speaker_id = 0
                speaker_emo_subpath = os.path.join(speakers[speaker_id], emotion)
                utt_list = os.listdir(os.path.join(path, speaker_emo_subpath))
                for i in range(max_num_per_gender):
                    if i != 0 and i % max_num_per_speaker == 0:
                        speaker_id += 1
                        speaker_emo_subpath = os.path.join(speakers[speaker_id], emotion)
                        utt_list = os.listdir(os.path.join(path, speaker_emo_subpath))
                        print("moving to speaker {}, starting with file {}".format(speaker_id, utt_list[i]))
                    utt = utt_list[i]
                    filepath = os.path.join(path, speaker_emo_subpath, utt)
                    try:
                        single_spectrogram = dict()
                        single_spectrogram['features'] = audio_to_spectrogram(
                            filepath,
                            args.offset, args.duration, args.n_mels).numpy().tolist()
                        single_spectrogram['target'] = label
                        single_spectrogram['gender'] = gender
                        spectrograms[utt] = single_spectrogram

                        num_created_specs += 1
                        if num_created_specs >= max_num:
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
            print(spectrograms)
            out_file = os.path.join('data', 'ESD_Singapore', 'spectrograms', language,
                                    emotion + '_' + str(max_num) + '_dur_' + str(args.duration) + '.json')
            with open(out_file, 'w') as out:
                json.dump(spectrograms, out)


def create_singapore_support_spectrograms_json():
    """This creates one separate support-set file with one spectrogram for each emotion"""
    languages = ['en', 'zh']
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    gender = 'm'
    for language in languages:
        for emotion in emotions:
            path = os.path.join("data/ESD_Singapore/", language, gender)
            label = emotion[:3].lower()
            spectrograms = dict()
            deleted_files = 0
            utt_index_to_be_retrieved = 250
            speakers = os.listdir(path)
            speaker_id = 0
            speaker_emo_subpath = os.path.join(speakers[speaker_id], emotion)
            utt_list = os.listdir(os.path.join(path, speaker_emo_subpath))
            utt = utt_list[utt_index_to_be_retrieved]
            filepath = os.path.join(path, speaker_emo_subpath, utt)
            try:
                single_spectrogram = dict()
                single_spectrogram['features'] = audio_to_spectrogram(
                    filepath,
                    args.offset, args.duration, args.n_mels).numpy().tolist()
                single_spectrogram['target'] = label
                single_spectrogram['gender'] = gender
                spectrograms[utt] = single_spectrogram

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
            print(spectrograms)
            out_file = os.path.join('data', 'ESD_Singapore', 'spectrograms', language,
                                    'support_' + emotion + '_dur_' + str(args.duration) + '.json')
            with open(out_file, 'w') as out:
                json.dump(spectrograms, out)


def merge_json(input_list, out):
    for i in range(len(input_list)):
        if not os.path.exists(out):
            print(input_list[i])
            with open(input_list[i], 'r') as f:
                specs_1 = json.load(f)
            with open(input_list[i+1], 'r') as f:
                specs_2 = json.load(f)
                merged = jsonmerge.merge(specs_1, specs_2)
            with open(out, 'w+') as f:
                json.dump(merged, f)
        else:
            with open(input_list[i], 'r') as f:
                specs_1 = json.load(f)
            with open(out, 'r') as f:
                specs_2 = json.load(f)
                merged = jsonmerge.merge(specs_1, specs_2)
            with open(out, 'w+') as f:
                json.dump(merged, f)


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

    path_en = os.path.join('data', 'ESD_Singapore', 'spectrograms', 'en')
    path_zh = os.path.join('data', 'ESD_Singapore', 'spectrograms', 'zh')
    file_list_en = os.listdir(os.path.join('data', 'ESD_Singapore', 'spectrograms', 'en'))
    file_list_zh = os.listdir(os.path.join('data', 'ESD_Singapore', 'spectrograms', 'zh'))
    input_list_en = list()
    input_list_zh = list()
    for file in file_list_en:
        # if "support" in file:
        #     input_list_en.append(os.path.join(path_en, file))
        input_list_en.append(os.path.join(path_en, file))
    for file in file_list_zh:
        # if "support" in file:
        #     input_list_zh.append(os.path.join(path_zh, file))
        input_list_zh.append(os.path.join(path_zh, file))

    outfile_en = '_support_merged_en.json'
    outfile_zh = '_support_merged_zh.json'
    outpath_en = os.path.join('data', 'ESD_Singapore', outfile_en)
    outpath_zh = os.path.join('data', 'ESD_Singapore', outfile_zh)

    #create_singapore_spectrograms_json()
    #create_singapore_support_spectrograms_json()
    #merge_json(input_list_en, outpath_en)
    #merge_json(input_list_zh, outpath_zh)
