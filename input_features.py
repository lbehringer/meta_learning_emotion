import librosa
import torch 
import torchaudio
import numpy


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

