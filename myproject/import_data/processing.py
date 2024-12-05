# import_data/processing.py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas 
import ipywidgets
import mne
import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from botocore.exceptions import NoCredentialsError
from mne import Epochs, compute_covariance, find_events, make_ad_hoc_cov
from mne.datasets import sample
from mne.preprocessing import annotate_movement, compute_average_dev_head_t, annotate_muscle_zscore, find_eog_events
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf, read_raw_edf
from mne.viz import plot_alignment, set_3d_view
from mne.simulation import (
    add_ecg,
    add_eog,
    add_noise,
    simulate_raw,
    simulate_sparse_stc,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassCalibrationError  # For calibration metrics
import torch.nn.functional as F
from django.conf import settings


# EDFInputPath = 'EDFFiles'
# EDFOutputPath = 'OutputFiles'

def preprocess_eeg(file_path):
    # Placeholder function to simulate preprocessing
    print(f"Preprocessing the EEG file at: {file_path}")
    # processed_data = "processed_data_placeholder"  # Simulated processed data


    def AllEDFProcess(EDFFolder):
        # if not os.path.exists(EDFOutputPath):
        #     os.makedirs(EDFOutputPath)
    
        # for FileName in os.listdir(EDFFolder):
        #   if FileName.endswith('.edf'):
        #      EDFFilePath = os.path.join(EDFFolder, FileName)
        processed_data, PSD_data, EEG_image = EDFProcess(EDFFolder)
        return processed_data, PSD_data


    def EDFProcess(EDFFilePath):
        RawEEGDataFile = mne.io.read_raw_edf(EDFFilePath, preload=True)
        RawEEGDataFile.interpolate_bads();

        BPEEGDataFile = BPFilter(RawEEGDataFile)

        # OutputFileName = f"filtered_{os.path.splitext(os.path.basename(EDFFilePath))[0]}.fif"
        # OutputFile = os.path.join(EDFOutputPath, OutputFileName)
        # BPEEGDataFile.save(OutputFile, overwrite=True)

        EEG_image = RawEEGDataFile
        #EEG_image = 'peepeepoopoo'
        

        ADRatioDF = AlphaDeltaProcess(BPEEGDataFile)
    
        # PSDOutputFileName = f"PSD_{os.path.splitext(os.path.basename(EDFFilePath))[0]}.csv"
        # PSDOutputFile = os.path.join(EDFOutputPath, PSDOutputFileName)
        # ADRatioDF.to_csv(PSDOutputFile, index=False)

        #print(f"Finished and saved file {EDFFilePath} to {OutputFile}")
        #print(f"Finished and saved PSD data to {PSDOutputFile}")
        return BPEEGDataFile, ADRatioDF, EEG_image

    def BPFilter(RawEEGDataFile):
        BPEEGDataFile = RawEEGDataFile.copy().filter(l_freq=0.5, h_freq=40.0, fir_design='firwin')
        return BPEEGDataFile


    ## ALPHA DELTA PSD ANALYSIS AND DATA FRAMING ##
    def AlphaDeltaProcess(EEGFile):
        AlphaComp = EEGFile.compute_psd(method='welch', fmin=8, fmax=12, tmin=None, tmax=None, picks='eeg', exclude=(), proj=False, remove_dc=True, reject_by_annotation=True, n_jobs=1, verbose=None)
        AlphaPSD, AlphaFreq = AlphaComp.get_data(return_freqs=True)
        #display(AlphaComp)
        DeltaComp = EEGFile.compute_psd(method='welch', fmin=0.5, fmax=4, tmin=None, tmax=None, picks='eeg', exclude=(), proj=False, remove_dc=True, reject_by_annotation=True, n_jobs=1, verbose=None)
        DeltaPSD, DeltaFreq = DeltaComp.get_data(return_freqs=True)
        #DeltaComp.plot()
        #raw_csd = mne.preprocessing.compute_current_source_density(RawEEGDataFile);

        ChanLab = EEGFile.ch_names

        AlphaMean = AlphaPSD.mean(axis=1)
        DeltaMean = DeltaPSD.mean(axis=1)

        AlDeRat = AlphaMean / DeltaMean

        PSDRatDF = pandas.DataFrame({'Channel': ChanLab,'Alpha Power': AlphaMean,'Delta Power': DeltaMean,'Alpha/Delta Ratio': AlDeRat})

        #display(PSDRatDF)
    
        return PSDRatDF



    processed_data, PSD_data = AllEDFProcess(file_path)

    return processed_data, PSD_data

def run_ml_model(single_file_path):
    class Args:
        seed_list = [0, 1004, 911, 2021, 119]
        seed = 10
        project_name = "test_project"
        checkpoint = False
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
        device = 0  # GPU device number to use

        binary_target_groups = 2
        output_dim = 4

        # Manually set paths here instead of using a YAML config file
        data_path = '/Volumes/SDCARD/v2.0.3/edf' # Set the data path directly
        dir_root = os.getcwd()  # Set the root directory as the current working directory
        dir_result = os.path.join(dir_root, 'results')  # Set the result directory directly

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


    class BasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()

            # First convolutional layer
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)

            # Second convolutional layer
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            # Shortcut for downsampling
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )

        def forward(self, x):
            # Forward pass
            out = F.relu(self.bn1(self.conv1(x)))  # First conv layer + batch norm + ReLU
            out = self.bn2(self.conv2(out))        # Second conv layer + batch norm
            out += self.shortcut(x)               # Add residual connection
            out = F.relu(out)                     # ReLU activation
            return out

    class CNN2D_LSTM_V8_4(nn.Module):
        def __init__(self, args, device):
            super(CNN2D_LSTM_V8_4, self).__init__()
            self.args = args

            # Model parameters
            self.num_layers = args.num_layers  # Number of LSTM layers
            self.hidden_dim = 256  # Features in the LSTM hidden state
            self.dropout = args.dropout  # Dropout rate
            self.num_data_channel = args.num_channel  # Data channels (e.g., EEG channels)
            self.sincnet_bandnum = args.sincnet_bandnum  # SincNet configuration
            self.feature_extractor = args.enc_model  # Feature extraction method
            self.in_planes = 1  # Input planes for ResNet

            # Activation functions
            activation = 'relu'  # Default activation function
            self.activations = nn.ModuleDict({
                'lrelu': nn.LeakyReLU(),
                'prelu': nn.PReLU(),
                'relu': nn.ReLU(inplace=True),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'leaky_relu': nn.LeakyReLU(0.2),
                'elu': nn.ELU()
            })

            # Initialize hidden state for LSTM
            self.hidden = (
                torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)
            )

            # Define ResNet layers using the BasicBlock
            self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

            # Adaptive average pooling
            self.agvpool = nn.AdaptiveAvgPool2d((1, 1))

            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout
            )

            # Fully connected classifier
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
                nn.BatchNorm1d(64),
                self.activations[activation],
                nn.Linear(in_features=64, out_features=args.output_dim, bias=True)
            )

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride1 in strides:
                layers.append(block(self.in_planes, planes, stride1))
                self.in_planes = planes
            return nn.Sequential(*layers)

        def forward(self, x):
            batch_size = x.size(0)

            # Permute input
            x = x.permute(0, 2, 1).unsqueeze(1)

            # Pass through ResNet layers
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            # Apply adaptive pooling
            x = self.agvpool(x)
            x = x.view(x.size(0), -1)

            # Prepare for LSTM
            x = x.unsqueeze(1)

            # Initialize LSTM hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)

            # LSTM forward pass
            output, hidden = self.lstm(x, hidden)
            output = output[:, -1, :]

            # Classification
            output = self.classifier(output)
            return output

        def init_state(self, device):
            self.hidden = (
                torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)
            )

    # Function to calculate metrics
    def calculate_metrics(labels, predictions, probabilities, num_classes):
        # Calculate accuracy
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # Calculate confusion matrix
        cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=list(range(num_classes)))

        # Calculate precision, recall, FPR, FNR
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Precision, Recall, FPR, FNR per class
        precision = TP / (TP + FP + 1e-8)  # Added epsilon to avoid division by zero
        recall = TP / (TP + FN + 1e-8)
        FPR = FP / (FP + TN + 1e-8)
        FNR = FN / (FN + TP + 1e-8)

        # Average metrics
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_FPR = np.mean(FPR)
        avg_FNR = np.mean(FNR)

        # Compute calibration error
        calibration_error = ece_criterion(probabilities, labels)

        metrics = {
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_FPR': avg_FPR,
            'avg_FNR': avg_FNR,
            'ECE': calibration_error.item()
        }

        return metrics

    # Function to log metrics
    def log_metrics(metrics, epoch, phase):
        logging.info(f'{phase} Metrics after Epoch {epoch}:')
        logging.info(f"Accuracy: {metrics['accuracy']:.2f}%")
        logging.info(f"Precision: {metrics['avg_precision']:.4f}")
        logging.info(f"Recall: {metrics['avg_recall']:.4f}")
        logging.info(f"False Positive Rate: {metrics['avg_FPR']:.4f}")
        logging.info(f"False Negative Rate: {metrics['avg_FNR']:.4f}")
        logging.info(f"Expected Calibration Error: {metrics['ECE']:.4f}")

        # Write to TensorBoard
        writer.add_scalar(f'{phase} accuracy', metrics['accuracy'], epoch)
        writer.add_scalar(f'{phase} precision', metrics['avg_precision'], epoch)
        writer.add_scalar(f'{phase} recall', metrics['avg_recall'], epoch)
        writer.add_scalar(f'{phase} FPR', metrics['avg_FPR'], epoch)
        writer.add_scalar(f'{phase} FNR', metrics['avg_FNR'], epoch)
        writer.add_scalar(f'{phase} ECE', metrics['ECE'], epoch)

    # Function to evaluate the model
    def evaluate_model(model, data_loader, device, temperature, num_classes):
        model.eval()
        with torch.no_grad():
            all_labels = []
            all_predictions = []
            all_probs = []
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # Apply temperature scaling
                scaled_outputs = outputs / temperature

                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(scaled_outputs, dim=1)

                # Get predicted labels
                _, predicted = torch.max(probabilities.data, 1)

                all_labels.append(labels)
                all_predictions.append(predicted)
                all_probs.append(probabilities)

            # Concatenate all tensors
            all_labels = torch.cat(all_labels)
            all_predictions = torch.cat(all_predictions)
            all_probs = torch.cat(all_probs)

            # Calculate metrics
            metrics = calculate_metrics(all_labels, all_predictions, all_probs, num_classes)

        return metrics

    # Function to test on a single data point
    def test_single_data_point(model, data_point, label, device, temperature, class_names=None):
        model.eval()
        target_channels = 40
        target_points = 5000
        segment_length = 500
        max_time_points = 100000
        
        with torch.no_grad():
            # single_file_input = data_point.to(device).unsqueeze(0)  # Add batch dimension
            data = mne.io.read_raw_edf(data_point, preload=True).get_data()

            # Limit the number of time points if they exceed max_time_points
            if data.shape[1] > max_time_points:
                data = data[:, :max_time_points]

            # Pad or trim channels to match target_channels
            if data.shape[0] < target_channels:
                padding = np.zeros((target_channels - data.shape[0], data.shape[1]))
                data = np.vstack((data, padding))
            elif data.shape[0] > target_channels:
                data = data[:target_channels, :]

            # Randomly select a segment
            if data.shape[1] > segment_length:
                start = np.random.randint(0, max(1, data.shape[1] - segment_length))
                end = start + segment_length
                segment = data[:, start:end]
            else:
                segment = data

            # Interpolate or compress to match target_points
            if segment.shape[1] != target_points:
                segment = np.array([np.interp(np.linspace(0, 1, target_points),
                                            np.linspace(0, 1, segment.shape[1]), channel)
                                    for channel in segment])
                                    
            # Return tensor
            data = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).dtype)
            
            output = model(data)
            # Apply temperature scaling
            scaled_output = output / temperature

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(scaled_output, dim=1)
            probabilities = probabilities.cpu().numpy()[0]  # Convert to numpy array

            # Get predicted label
            predicted_label = np.argmax(probabilities)
            actual_label = label

            # Calculate accuracy (1 if correct, 0 if incorrect)
            is_correct = int(predicted_label == actual_label)
            accuracy = 100 * is_correct

            logging.info('Single Data Point Test:')
            if class_names:
                logging.info(f'Actual Label: {class_names[actual_label]}')
                logging.info(f'Predicted Label: {class_names[predicted_label]}')
            else:
                logging.info(f'Actual Label: {actual_label}')
                logging.info(f'Predicted Label: {predicted_label}')
            logging.info(f'Accuracy: {accuracy}%')
            logging.info('Class Probabilities:')
            for i, prob in enumerate(probabilities):
                class_label = class_names[i] if class_names else i
                logging.info(f'Class {class_label}: {prob*100:.2f}%')
                classProbs[i] = prob

        
            # Plot the probabilities as a bar graph
            fig, ax = plt.subplots()
            classes = class_names if class_names else list(range(len(probabilities)))
            ax.bar(classes, probabilities)
            ax.set_xlabel('Class')
            ax.set_ylabel('Probability')
            ax.set_title('Class Probabilities')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Add the figure to TensorBoard
            writer.add_figure('Class Probabilities', fig)

            output_file = os.path.join(output_folder, 'bar_graph_ci.png')
            prob_png = fig.savefig(output_file)

    ### CHANGE THIS FOR CUSTOM FILES AND MODEL PATHS ###
    model_path = r'C:\Users\aprib\OneDrive\Documents\BME489\3int\StrokeTTC\myproject\full_model.pth'
    output_folder = r'C:\Users\aprib\OneDrive\Documents\BME489\3int\StrokeTTC\myproject\static'

    # List to store the probabilities of each class
    classProbs = [0,0,0,0]
    classNames = ['Healthy', 'Epilepsy', 'Stroke', 'Concussion']

    args = Args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info("Using MPS device (Apple Silicon GPU).")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using CUDA device.")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU.")

    # Hyperparameters
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    temperature = 2.0  # Temperature parameter for temperature scaling
    # checkpoint_dir = './checkpoints'
    # os.makedirs(checkpoint_dir, exist_ok=True)

    model = CNN2D_LSTM_V8_4(args,device).to(device)  # Initialize the model
    model = torch.load(model_path, map_location=device)  # Load the state dictionary
    model = model.float()  # Convert model to float
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Calibration metric
    ece_criterion = MulticlassCalibrationError(num_classes=10, n_bins=15).to(device)

    # TensorBoard writer
    writer = SummaryWriter('runs/simple_cnn')

    # Test on a single data point
    # Let's take the first image from the test set
    single_label = 0
    predicted_label, prob_png = test_single_data_point(model, single_file_path, single_label, device, temperature)

    logging.info("Training and evaluation completed.")

    # Close the TensorBoard writer
    writer.close()
    return predicted_label, prob_png
