from yacs.config import CfgNode as CN
from text import symbols

_C = CN()
_C.save_dir = 'ckpt'
_C.exclude_dirs = ['ckpt', 'data', 'dataset']
_C.checkpoint_path = './ckpt/v2/model_00002950'
_C.val_files = "filelists/ljs_audio_text_val_filelist.txt"
_C.train_files = "filelists/ljs_audio_text_train_filelist.txt"
_C.test_files = "filelists/ljs_audio_text_test_filelist.txt"
_C.audio_root = 'data/LJSpeech-1.1/wavs'
_C.mel_spectrogram_root = 'data/LJSpeech-1.1-spectrogram'
_C.val_epoch = 10
_C.save_epoch = 20

################################
# Experiment Parameters        #
################################
_C.epochs=500
_C.iters_per_checkpoint=5000
_C.seed=1234
_C.dynamic_loss_scaling=True
_C.fp16_run=False
_C.distributed_run=False
_C.dist_backend="nccl"
_C.dist_url="tcp://localhost:54321"
_C.cudnn_enabled=True
_C.cudnn_benchmark=False
_C.ignore_layers=['embedding.weight']
_C.text_cleaners=['english_cleaners']

################################
# Audio Parameters             #
################################
_C.max_wav_value=32768.0
_C.sampling_rate=22050
_C.filter_length=1024
_C.hop_length=256
_C.win_length=1024
_C.n_mel_channels=80
_C.mel_fmin=0.0
_C.n_fft=1024
_C.mel_fmax=11025.0

################################
# Model Parameters             #
################################
_C.n_symbols=len(symbols)
_C.symbols_embedding_dim=512

# Encoder parameters
_C.encoder_kernel_size=5
_C.encoder_n_convolutions=3
_C.encoder_embedding_dim=512

# Decoder parameters
_C.n_frames_per_step=1  # currently only 1 is supported
_C.decoder_rnn_dim=1024
_C.prenet_dim=256
_C.max_decoder_steps=1000
_C.gate_threshold=0.5
_C.p_attention_dropout=0.1
_C.p_decoder_dropout=0.1

# Attention parameters
_C.attention_rnn_dim=1024
_C.attention_dim=128

# Location Layer parameters
_C.attention_location_n_filters=32
_C.attention_location_kernel_size=31

# Mel-post processing network parameters
_C.postnet_embedding_dim=512
_C.postnet_kernel_size=5
_C.postnet_n_convolutions=5

################################
# Optimization Hyperparameters #
################################
_C.use_saved_learning_rate=False
_C.learning_rate=1e-3
_C.weight_decay=1e-6
_C.grad_clip_thresh=1.0
_C.batch_size=64
_C.mask_padding=True  # set model's padded outputs to padded values
