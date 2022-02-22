import numpy as np
from scipy.io.wavfile import read
import torch
import librosa

def griffinlim_reconstruction(spectrogram, n_iter=50, window='hann',
                              n_fft=1024, hop_length=256):
    if hop_length == -1:
        hop_length = n_fft // 4
    # Implementation from Librosa github issue #434
    # https://github.com/librosa/librosa/issues/434

    # use random phase
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    # use mix phase
    # angles = np.exp(1j * phase)

    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(
            full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(
            inverse, n_fft=n_fft, hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return np.clip(inverse, -1., 1.)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()

    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate



def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
