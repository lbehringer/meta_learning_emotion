import argparse
from input_features import audio_to_spectrogram, load_dvector, concatenate_features


def main(args):
    # example: feature concatenation 
    print(concatenate_features(load_dvector('test.wav'),
                           audio_to_spectrogram('test.wav', args.offset, args.duration, args.n_mels)))

    # TBD (dataloading & conversion, model training, evaluation)


# possible arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=1, required=False,
                        type=int, help="offset in seconds")
    parser.add_argument("--duration", default=4, required=False,
                        type=int, help="window size in seconds")
    parser.add_argument("--n_mels", default=4, required=False, type=int, help="number of filterbanks -> hyperparameter")

    args = parser.parse_args()

    main(args)
