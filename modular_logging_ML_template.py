#this is going to be our main.py file that imports all the good preprocessing stuff and data input stuff and the machine learning stuff

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
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter('runs/simple_cnn')

# Image preprocessing modules
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST dataset
full_train_dataset = torchvision.datasets.MNIST(root='./data',
                                                train=True,
                                                transform=transform,
                                                download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform)

# Split train_dataset into train and validation sets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Simple CNN model (you can replace this with any other model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 10)  # Adjusted for MNIST image size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = self.fc1(x)
        return x

# Define a basic block for ResNet. This block will be reused in the construction of ResNet layers.
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        # Input:
        # - in_planes: Number of input channels
        # - planes: Number of output channels after convolution
        # - stride: Stride used in convolution to control output size
        super(BasicBlock, self).__init__()
        
        # Define the first convolutional layer, followed by batch normalization.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Define the second convolutional layer, with stride fixed at 1, followed by batch normalization.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # If stride is greater than 1, downsample input to match the shape for addition.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # Forward pass of the BasicBlock.
        # Input: x - input tensor
        # Output: out - output tensor after applying convolutions and adding residual connection
        out = F.relu(self.bn1(self.conv1(x)))  # First conv layer, followed by batch norm and ReLU activation
        out = self.bn2(self.conv2(out))        # Second conv layer, followed by batch norm
        out += self.shortcut(x)                # Add residual (shortcut) connection to maintain information flow
        out = F.relu(out)                      # Apply ReLU activation again to introduce non-linearity
        return out

# Define a CNN2D-LSTM model for EEG signal classification
class CNN2D_LSTM_V8_4(nn.Module):
    def __init__(self, args, device):
        super(CNN2D_LSTM_V8_4, self).__init__()      
        self.args = args

        # Set model parameters
        self.num_layers = args.num_layers  # Number of LSTM layers
        self.hidden_dim = 256  # Number of features in the hidden state of the LSTM
        self.dropout = args.dropout  # Dropout rate for regularization
        self.num_data_channel = args.num_channel  # Number of data channels (e.g., EEG channels)
        self.sincnet_bandnum = args.sincnet_bandnum  # SincNet configuration
        self.feature_extractor = args.enc_model  # Feature extraction method
        self.in_planes = 1  # Initial number of input planes for ResNet (matching the number of input channels)

        # Activation functions
        activation = 'relu'  # Use ReLU as the default activation function
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['leaky_relu', nn.LeakyReLU(0.2)],
            ['elu', nn.ELU()]
        ])

        # Create a new variable for the hidden state, necessary to calculate the gradients
        self.hidden = (
            (torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device),
             torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device))
        )

        # Define the ResNet layers using the BasicBlock
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # First ResNet layer with 64 output channels
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # Second ResNet layer with 128 output channels
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # Third ResNet layer with 256 output channels

        # Adaptive average pooling to reduce spatial dimensions to (1, 1)
        self.agvpool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layer for temporal sequence learning
        self.lstm = nn.LSTM(
            input_size=256,  # Input size matches the output of the ResNet layers
            hidden_size=self.hidden_dim,  # Number of features in LSTM hidden state
            num_layers=args.num_layers,  # Number of LSTM layers
            batch_first=True,  # Input and output tensors are provided as (batch, seq, feature)
            dropout=args.dropout  # Dropout for regularization
        )

        # Fully connected classifier layer for outputting class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),  # Linear layer to reduce feature dimension
            nn.BatchNorm1d(64),  # Batch normalization layer
            self.activations[activation],  # Activation function
            nn.Linear(in_features=64, out_features=args.output_dim, bias=True),  # Final linear layer for classification
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        # Create a ResNet layer with multiple blocks.
        # Input:
        # - block: Block type (BasicBlock)
        # - planes: Number of output channels for this layer
        # - num_blocks: Number of blocks in this layer
        # - stride: Stride for the first block
        strides = [stride] + [1] * (num_blocks - 1)  # Set stride for the first block, others have stride of 1
        layers = []
        for stride1 in strides:
            layers.append(block(self.in_planes, planes, stride1))  # Append blocks to the layer
            self.in_planes = planes  # Update input channel size for the next block
        return nn.Sequential(*layers)  # Return the complete layer as a sequential model

    def forward(self, x):
        # Forward pass of the CNN2D-LSTM model.
        # Input: x - input tensor of shape (batch_size, channels, sequence_length)
        # Output: output - output tensor with class scores

        batch_size = x.size(0)  # Get the batch size from the input tensor

        # Permute the input to (batch_size, sequence_length, channels)
        x = x.permute(0, 2, 1)  # You may need to adjust this permutation based on your input shape

        # Reshape to add channel dimension (e.g., for convolutional layers)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, sequence_length, channels)

        # Pass through ResNet layers
        x = self.layer1(x)  # Pass through first ResNet layer
        x = self.layer2(x)  # Pass through second ResNet layer
        x = self.layer3(x)  # Pass through third ResNet layer

        # Apply adaptive average pooling to reduce spatial size to (1, 1)
        x = self.agvpool(x)  # Shape: (batch_size, features, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, features)

        # Prepare data for LSTM
        # Assuming that the LSTM expects (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # Add a time dimension to make it compatible with LSTM (batch_size, seq_len=1, features)

        # Initialize the hidden state for the LSTM with the correct batch size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        hidden = (h0, c0)

        # LSTM forward pass
        output, hidden = self.lstm(x, hidden)  # Apply LSTM to learn temporal dependencies
        output = output[:, -1, :]  # Take the output from the last time step of the sequence

        # Classification through the fully connected layer
        output = self.classifier(output)  # Classify using the fully connected layer

        return output

    def init_state(self, device):
        # Initialize the hidden state for the LSTM
        # Input: device - The device (CPU or GPU) where the hidden state should be allocated
        # Output: Initializes the hidden state with zeros
        self.hidden = (
            (torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device),
             torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device))
        )

# Manually set arguments here instead of using argparse
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
    
    
args = Args()

'''#model = SimpleCNN().to(device)
model_path = 'full_model.pth'
model = CNN2D_LSTM_V8_4(args, device).to(device)  # Initialize the model
#model = SimpleCNN().to(device)  # Initialize the model
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the state dictionary '''

import torch

model_path = 'full_model.pth'
import mne

# Load the entire model object
model = torch.load(model_path, map_location=device)
edfFile = '/Volumes/SDCARD/v2.0.3/edf/eval/aaaaaqld/s002_2014/01_tcp_ar/aaaaaqld_s002_t001.edf' # path the the edf file
input = mne.io.read_raw_edf(edfFile, preload=True).get_data()

output = model(edfFile)  # Forward pass

# show the classificaiton output
print(output)

# Set the model to evaluation mode if you are going to use it for inference
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Calibration metric
ece_criterion = MulticlassCalibrationError(num_classes=10, n_bins=15).to(device)

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
    with torch.no_grad():
        single_image_input = data_point.to(device).unsqueeze(0)  # Add batch dimension
        output = model(single_image_input)
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

        # Visualize the input image and prediction in TensorBoard
        # Unnormalize the image for visualization
        img = data_point.numpy() * 0.3081 + 0.1307  # Unnormalize

        # Add image and prediction to TensorBoard
        writer.add_image('Single Test Image', img, dataformats='CHW')
        writer.add_text('Single Test Prediction',
                        f'Actual Label: {actual_label}, Predicted Label: {predicted_label}, Accuracy: {accuracy}%')

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

# Training loop
total_steps = len(train_loader)
logging.info("Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss
        if (i+1) % 100 == 0:
            logging.info(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
            # Write to TensorBoard
            writer.add_scalar('training loss', loss.item(), epoch * total_steps + i)

    # Validation
    val_metrics = evaluate_model(model, val_loader, device, temperature, num_classes=10)
    log_metrics(val_metrics, epoch, phase='Validation')

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f'Model checkpoint saved to {checkpoint_path}')

# Testing
test_metrics = evaluate_model(model, test_loader, device, temperature, num_classes=10)
log_metrics(test_metrics, num_epochs, phase='Test')

# Test on a single data point
# Let's take the first image from the test set
single_image, single_label = test_dataset[0]
test_single_data_point(model, single_image, single_label, device, temperature)

logging.info("Training and evaluation completed.")

# Close the TensorBoard writer
writer.close()
