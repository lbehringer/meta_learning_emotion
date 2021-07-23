import librosa
import torch
import torchaudio
import numpy as np


# convert audio to spectrogram and normalize
def audio_to_spectrogram(path_audio, offset, duration, n_mels):
    signal = librosa.load(path_audio, sr=16000, offset=offset, duration=duration)[
        0]  # , offset=args.offset, duration=args.duration)

    mel_spectrogram = librosa.feature.melspectrogram(
        signal, n_fft=800, hop_length=400, n_mels=n_mels, fmax=8000, win_length=800)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # normalise log mel spectrogram -> think about which norm makes the most sense
    norm_spec = librosa.util.normalize(log_mel_spectrogram, norm=2)
    norm_spec = torch.from_numpy(norm_spec)
    print(norm_spec.shape)
    return norm_spec


# load speaker embedding
def load_dvector(path_audio):
    wav2mel = torch.jit.load("data_dvector/wav2mel.pt")
    dvector = torch.jit.load("data_dvector/dvector-step250000.pt").eval()
    wav_tensor, sample_rate = torchaudio.load(path_audio)
    # shape: (frames, mel_dim)
    mel_tensor = wav2mel(wav_tensor, sample_rate)
    emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
    emb_tensor = emb_tensor.detach().numpy()

    return emb_tensor


#  concatenate normalised log mel spectrogram with speaker embedding
def concatenate_features(d_vector, norm_spec):
    d_vector = d_vector.copy()
    pad = norm_spec.shape[1]
    d_vector.resize(pad)  # pad d_vector with zeros
    # concatenate d_vector as a row to the normalized log spec matrix
    input_matrix = numpy.vstack([norm_spec, d_vector])

    return input_matrix, input_matrix.shape


def get_spectrograms(fpath, ref_db, max_db, n_fft, sr, n_mels, preemphasis):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # num = np.random.randn()
    # if num < .2:
    #     y, sr = librosa.load(fpath, sr=hp.sr)
    # else:
    #     if num < .4:
    #         tempo = 1.1
    #     elif num < .6:
    #         tempo = 1.2
    #     elif num < .8:
    #         tempo = 0.9
    #     else:
    #         tempo = 0.8
    #     cmd = "ffmpeg -i {} -y ar {} -hide_banner -loglevel panic -ac 1 -filter:a atempo={} -vn temp.wav".format(fpath, hp.sr, tempo)
    #     os.system(cmd)
    #     y, sr = librosa.load('temp.wav', sr=hp.sr)

    # Loading sound file
    y, sr = librosa.load(fpath, sr=sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=int(sr*0.0125),
                          win_length=int(sr*0.0125))

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


# print(get_spectrograms('Emotional_Speech_Dataset_Singapore/0001/Angry/0001_000351.wav',
    #     ref_db=20, max_db=100, n_fft=2048, sr=16000, n_mels=80, preemphasis=.97))
