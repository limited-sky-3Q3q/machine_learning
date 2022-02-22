import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import librosa
import os
from librosa.filters import mel as librosa_mel_fn
from config import _C as config
from text import text_to_sequence


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


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads text, mel-spectrogram pairs
        2) normalizes text and converts them to sequences of one-hot vectors
    """
    def __init__(self, split_path):
        with open(split_path, encoding='utf-8') as f:
            self.audiopaths_and_text = [line.strip().split('|') for line in f]
        f.close()
        self.audio2mel = Audio2Mel()

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audio_id, text = audiopath_and_text
        spec_path = os.path.join(config.mel_spectrogram_root, 
                        audio_id.replace('.wav', '.npy'))
        text = self.get_text(text)
        mel_spectrogram = self.get_mel_spectrogram(spec_path)
        return (text, mel_spectrogram, audio_id.replace('.wav', ''))

    def get_mel_spectrogram(self, filename):
        mel_spectrogram = np.load(filename).astype(np.float32)
        mel_spectrogram = torch.FloatTensor(mel_spectrogram)
        return mel_spectrogram

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, config.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and audio
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
        audio_ids = [x[2] for x in batch]
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, audio_ids