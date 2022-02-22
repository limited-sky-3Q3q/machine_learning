import argparse
import os
import torch
import numpy as np
from config import _C as config
from text import text_to_sequence
from model import Tacotron2
import librosa
from utils import griffinlim_reconstruction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        config.merge_from_file(args.config_file)
 
    config.merge_from_list(args.opts)
    config.freeze()

    model = Tacotron2().cuda()
    ckpt = torch.load(config.checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt["state_dict"])
    mel_basis = librosa.filters.mel(sr=config.sampling_rate, n_fft=config.n_fft, 
                                n_mels=config.n_mel_channels, 
                                fmin=config.mel_fmin, fmax=config.mel_fmax)

    text = 'no economic panacea, which could simply revive over-night the heavy industries and the trades dependent upon them'
    sequence = np.array(text_to_sequence(text, config.text_cleaners))[None, :]
    sequence = torch.from_numpy(sequence).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    melspec = torch.exp(mel_outputs[0]).data.cpu().numpy()
    linear_spec = np.matmul(np.linalg.pinv(mel_basis), melspec)
    audio = griffinlim_reconstruction(linear_spec)
    librosa.output.write_wav("test.wav", audio, sr=22050)
