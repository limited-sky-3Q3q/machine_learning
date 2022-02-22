import argparse
import json
import os
import os.path as P
import torch
import random
import time
import librosa
import numpy as np
from tensorboardX import SummaryWriter

from config import _C as config
from Recorder import Recorder
from contextlib import redirect_stdout

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from loss_function import Tacotron2Loss
from dataset import TextAudioLoader, TextMelCollate
from utils import griffinlim_reconstruction

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict( checkpoint_dict['state_dict'])
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    torch.save({'state_dict': model.state_dict(),
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def plot_results(mel_basis, audio_ids, y_pred, y, epoch):
    save_dir = P.join(config.save_dir, 'result', f'epoch_{epoch:06d}')
    os.makedirs(save_dir, exist_ok=True)
    for yi in range(len(y_pred[0])):
        audio_gt, sr = librosa.load(P.join(config.audio_root, audio_ids[yi]+'.wav'), sr=None)
        melspec = torch.exp(y_pred[0][yi]).data.cpu().numpy()
        linear_spec = np.matmul(np.linalg.pinv(mel_basis), melspec)
        audio_pred = griffinlim_reconstruction(linear_spec)
        librosa.output.write_wav(P.join(save_dir, audio_ids[yi]+'_gt.wav'), audio_gt, sr=22050)
        librosa.output.write_wav(P.join(save_dir, audio_ids[yi]+'_pred.wav'), audio_pred, sr=22050)

        plt.figure(figsize=(12, 9))
        
        ax1 = plt.subplot(312)
        ax1.plot(audio_pred)
        ax1.set_title('audio pred')

        ax2 = plt.subplot(311, sharey=ax1, sharex=ax1)
        ax2.plot(audio_gt)
        ax2.set_title('audio ground_truth')

        plt.subplot(325)
        plt.imshow(y[0][yi].data.cpu().numpy(), origin='lower', aspect='auto')
        plt.title('mel-spectrogram ground_truth')

        plt.subplot(326)
        plt.imshow(y_pred[0][yi].data.cpu().numpy(), origin='lower', aspect='auto')
        plt.title('mel-spectrogram pred')

        plt.margins(0, 0)
        plt.tight_layout()
        plt.savefig(P.join(save_dir, audio_ids[yi]+'.jpg'))
        plt.close()

def train(num_gpus, rank, group_name):

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, config.dist_backend, config.dist_url)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = Tacotron2Loss()
    model = Tacotron2().cuda()

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
       model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if config.checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(config.checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = TextAudioLoader(config.train_files)
    valset = TextAudioLoader(config.val_files)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    val_sampler = DistributedSampler(valset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    collate_fn = TextMelCollate(config.n_frames_per_step)
    shuffle = False if num_gpus > 1 else True
    train_loader = DataLoader(trainset, num_workers=4, shuffle=shuffle,
                              batch_size=config.batch_size,
                              sampler=train_sampler,
                              pin_memory=False,
                              drop_last=True, 
                              collate_fn=collate_fn)
    val_loader = DataLoader(valset, num_workers=4, shuffle=False,
                              batch_size=1,
                              sampler=val_sampler,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=collate_fn)
    if rank == 0:
        testset = TextAudioLoader(config.test_files)
        test_loader = DataLoader(testset, num_workers=1, shuffle=False,
                                batch_size=1,
                                pin_memory=False,
                                drop_last=True)
    # Get shared output_directory ready
    if rank == 0:
        log_dir = P.join(config.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)
    
    mel_basis = librosa.filters.mel(sr=config.sampling_rate, n_fft=config.n_fft, 
                                n_mels=config.n_mel_channels, 
                                fmin=config.mel_fmin, fmax=config.mel_fmax)

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, config.epochs):
        reduced_loss_mean = []
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            x, y, audio_ids = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            reduced_loss_mean.append(reduced_loss)

            if config.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
                    
            print("epoch:{} iter:{} loss:{:.6f} time:{:0.4f}".format(
                epoch,i, reduced_loss, time.time() - start_time))
            iteration += 1
            start_time = time.time()

        if (epoch % config.save_epoch == 0):
            if rank == 0:
                checkpoint_path = P.join(config.save_dir, 'model_{:08d}'.format(iteration))
                save_checkpoint(model, optimizer, config.learning_rate, iteration,
                                checkpoint_path)

        if rank == 0 and reduced_loss_mean:
            if not (reduced_loss_mean is np.nan or reduced_loss_mean is np.inf):
                logger.add_scalar('loss/training', np.mean(reduced_loss_mean), epoch)

        if epoch % config.val_epoch == 0:
            if rank == 0 and epoch == 0:
                checkpoint_path = P.join(config.save_dir, 'model_{:08d}'.format(iteration))
                save_checkpoint(model, optimizer, config.learning_rate, iteration, checkpoint_path)
 
            model.eval()
            reduced_loss_mean = []
            for _, batch in enumerate(val_loader):
                x, y, audio_ids = model.parse_batch(batch)                
                with torch.no_grad():
                    y_pred = model(x)
                loss = criterion(y_pred, y)
                reduced_loss = loss.item()
                reduced_loss_mean.append(reduced_loss)
                plot_results(mel_basis, audio_ids, y_pred, y, epoch)

            model.train()
            if rank == 0 and reduced_loss_mean:
                if not (reduced_loss_mean is np.nan or reduced_loss_mean is np.inf):
                    logger.add_scalar('loss/val', np.mean(reduced_loss_mean), epoch)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        config.merge_from_file(args.config_file)
 
    config.merge_from_list(args.opts)
    config.freeze()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    if args.rank == 0:
        recorder = Recorder(config.save_dir, config.exclude_dirs)
        with open(P.join(config.save_dir, 'opts.yml'), 'w') as f:
            with redirect_stdout(f): 
                print(config.dump())
    
    if num_gpus == 1 or args.rank == 0:
        recorder.tee_stdout(P.join(config.save_dir, 'GPU_0.log'))
        
    setup_seed(config.seed)
    train(num_gpus, args.rank, args.group_name)
