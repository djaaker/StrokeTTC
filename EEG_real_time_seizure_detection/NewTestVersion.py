import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from itertools import groupby
import math
import time
from builder.utils.metrics import Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import mne

# Custom modules and utilities (required external Python files)
from builder.utils.lars import LARC  # Optimizer utility (external file needed)
from builder.data.v8dataPreprocessTest import get_data_preprocessed  # Data preprocessing function (external file needed)
from builder.models.detector_models.commented_resnet_lstm import CNN2D_LSTM_V8_4  # Import ResNetLSTM model directly (external file needed)
from builder.utils.logger import Logger  # Logger utility to track training and validation metrics (external file needed)
from builder.utils.utils import set_seeds, set_devices  # Utility functions for setting seeds and devices (external file needed)
from builder.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts  # Custom learning rate scheduler (external file needed)
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle  # Custom learning rate scheduler (external file needed)
from builder.trainer import get_trainer  # Trainer utility for training steps (external file needed)
#from builder.trainer import *  # Additional trainer utilities (external file needed)

# Setting CUDA device order to ensure consistent GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# List to store test results for each seed
list_of_test_results_per_seed = []

# Define result class

# Manually set arguments here instead of using argparse
class Args:
    seed_list = [0]
    seed = 0
    project_name = "test_project"
    #checkpoint = False
    epochs = 10
    batch_size = 32
    optim = 'adam'
    lr_scheduler = "Single"
    lr_init = 1e-3
    lr_max = 4e-3
    t_0 = 5
    t_mult = 2
    t_up = 1
    gamma = 0.5
    momentum = 0.9
    weight_decay = 1e-6
    task_type = 'binary'
    log_iter = 10
    best = True
    last = False
    test_type = "test"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")  # GPU device number to use

    binary_target_groups = 2
    output_dim = 2

    # Manually set paths here instead of using a YAML config file
    data_path = 'THIS SHOULD BE WHATEVER LOCAL FILE WE WANT TO USE' # Set the data path directly
    dir_root = os.getcwd()  # Set the root directory as the current working directory
    dir_result = os.path.join(dir_root, 'results')  # Set the result directory directly
    EEGFile = "/Users/aprib/OneDrive/Documents/BME489/aaaaaaaa_s001_t000.edf"
    EDFFile = mne.io.read_raw_edf(EEGFile).get_data()
    #outputs = model(EDFFile)
    # Check if the results directory exists, and create it if it doesn't
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    reset = False  # Set reset flag to False to avoid overwriting existing results

    num_layers = 2
    hidden_dim = 512  # Number of features in the hidden state of the LSTM
    dropout = 0.1  # Dropout rate for regularization
    num_channel = 20  # Number of data channels (e.g., EEG channels)
    sincnet_bandnum = 20  # SincNet configuration
    enc_model = 'raw'  # Encoder model for feature extraction

    window_shift_label = 1
    window_size_label = 4
    requirement_target = None
    sincnet_kernel_size = 81
    seed = 0
    project_name = "test_project"
    checkpoint = True
    epochs = 10
    batch_size = 32
    optim = 'adam'
    lr_scheduler = "Single"
    lr_init = 1e-3
    lr_max = 4e-3
    t_0 = 5
    t_mult = 2
    t_up = 1
    gamma = 0.5
    momentum = 0.9
    weight_decay = 1e-6
    task_type = 'binary'
    log_iter = 10
    best = True
    last = False
    test_type = "test"
    binary_target_groups = 2
    output_dim = 2
    data_path = '/path/to/data_directory/data_path'
    dir_root = os.getcwd()
    dir_result = '/path/to/results_directory'
    num_layers = 2
    dropout = 0.1
    num_channel =  32 # Number of data channels (e.g., EEG channels)
    sincnet_bandnum = 20 # SincNet configuration
    enc_model = "sincnet"
    window_size = 1
    window_size_sig = 1 #added 12/2
    sincnet_kernel_size = 81
    sincnet_layer_num = 1
    cnn_channel_sizes = [20, 10, 10]
    sincnet_stride = 2
    sincnet_input_normalize = "none"
    window_shift_label = 1
    window_size_label = 1
    requirement_target = None
    feature_sample_rate = 256
    ignore_model_speed = False
    target_channels = 40
    target_points = 5000
    segment_length = 500
    max_time_points = 100000


def initialize_model(args, device):
    # Create the model
    model = CNN2D_LSTM_V8_4(args, device).to(device)  # Directly initialize ResNetLSTM and move to the appropriate device (CPU, GPU, or MPS)
    return model

def load_checkpoint(args, model, device, logger, seed_num):
    # Load checkpoint if specified
    if args.checkpoint:
        if args.last:
            ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best_{}.pth'.format(str(seed_num))
        elif args.best:
            ckpt_path = "/Users/aprib/Downloads/best_model_weights.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)  # Load model checkpoint from file
        model.load_state_dict(checkpoint, strict=False)  # Load saved model state
        model.eval()
        print('loaded model')
        evaluator = Evaluator(args)
        evaluator.reset()
        result_list = []
        iteration = 0
        start_epoch = 1
        #logger.best_auc = checkpoint['score']  # Set best AUC score from checkpoint
        #start_epoch = checkpoint['epoch']  # Set starting epoch from checkpoint
        del checkpoint
    else:
        logger.best_auc = 0
        start_epoch = 1
    return model, start_epoch

def set_optimizer(args, model):
    # Set up the optimizer based on specified argument
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'adam_lars':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)  # LARS wrapper for adaptive learning rate scaling
    elif args.optim == 'sgd_lars':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    elif args.optim == 'adamw_lars':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    return optimizer

def set_scheduler(args, optimizer, one_epoch_iter_num):
    # Set up learning rate scheduler
    if args.lr_scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_0 * one_epoch_iter_num, T_mult=args.t_mult, eta_max=args.lr_max, T_up=args.t_up * one_epoch_iter_num, gamma=args.gamma)  # Custom cosine annealing scheduler with warmup (external file needed)
    elif args.lr_scheduler == "Single":
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), epochs=args.epochs, steps_per_epoch=one_epoch_iter_num, div_factor=math.sqrt(args.batch_size))  # Alternative scheduler (external file needed)
    return scheduler

def main():
    args = Args()

    # Set the device (MPS for Apple silicon, CUDA for Nvidia GPUs, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define result classes for validation and test results
    #save_valid_results = experiment_results_validation(args)
    #save_test_results = experiment_results(args)

    # Loop through each seed to train and evaluate the model
    for seed_num in args.seed_list:
        # Set the seed for reproducibility
        args.seed = seed_num
        set_seeds(args)

        # Initialize the logger to track training and validation metrics
        logger = Logger(args)  # Logger instance to save metrics
        logger.evaluator.best_auc = 0

        # Load preprocessed data (train, validation, test)
        train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)  # Data loaders for training, validation, and testing
        
        # Initialize model
        model = initialize_model(args, device)

        # Load checkpoint if available
        model, start_epoch = load_checkpoint(args, model, device, logger, seed_num)

        # Set up optimizer and scheduler
        optimizer = set_optimizer(args, model)
        one_epoch_iter_num = len(train_loader)  # Total number of iterations per epoch
        scheduler = set_scheduler(args, optimizer, one_epoch_iter_num)


        #logger.test_result_only()  # Log only validation results

        

def test_model(args, model, test_loader, device, logger, iteration, scheduler, optimizer):
    model.eval()  # Set model to evaluation mode
    test_iteration = 0
    logger.test_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = batch  # Unpack validation batch
            test_x, test_y = test_x.to(device), test_y.to(device)  # Move data to appropriate device (CPU, GPU, or MPS)
            model, test_loss = get_trainer(args, iteration, test_x, test_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, nn.CrossEntropyLoss(reduction='none'), signal_name_list, flow_type=args.test_type)  # Perform validation step
            logger.test_loss += np.mean(test_loss)
            test_iteration += 1
        
        # logger.log_val_loss(test_iteration, iteration)  # Log validation loss
        # logger.add_validation_logs(iteration)  # Add validation metrics to log
        # logger.save(model, optimizer, iteration, iteration)  # Save model checkpoint


if __name__ == "__main__":
    main()