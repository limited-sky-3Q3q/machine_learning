import numpy as np
import torch
import torch.nn as nn
import librosa
import os
from librosa.filters import mel as librosa_mel_fn
from config import _C as config
import argparse
from multiprocessing import Pool
from functools import partial
from glob import glob

class Audio2Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.sampling_rate = config.sampling_rate
        self.n_mel_channels = config.n_mel_channels
        self.mel_fmin = config.mel_fmin
        self.mel_fmax = config.mel_fmax
        window = torch.hann_window(self.win_length).float()
        mel_basis = librosa_mel_fn(
            self.sampling_rate, self.n_fft, self.n_mel_channels, self.mel_fmin, self.mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
       
    def forward(self, audio):
        audio = audio.squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        mel_spectrogram = torch.log(torch.clamp(mel_output, min=1e-5))
        return mel_spectrogram.squeeze()

audio2mel = Audio2Mel()

def save_spectrogram(filename, save_dir):
    audio, _ = librosa.load(filename, sr=config.sampling_rate)
    audio = torch.FloatTensor(audio.astype(np.float32))           
    audio = torch.autograd.Variable(audio, requires_grad=False)
    mel_spectrogram = audio2mel(audio.unsqueeze(0))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename.split(os.sep)[-1].replace('.wav', '.npy'))
    np.save(save_path, mel_spectrogram.data.cpu().numpy())


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument('-i', '--input_dir', type=str)
    paser.add_argument('-o', '--output_dir', type=str)
    paser.add_argument('-n', '--num_worker', type=int)
    args = paser.parse_args()

    files = glob(os.path.join(args.input_dir, "*.wav"))
    with Pool(args.num_worker) as p:
        p.map(partial(save_spectrogram, save_dir=args.output_dir), files)