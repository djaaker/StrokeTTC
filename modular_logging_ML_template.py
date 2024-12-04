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

model = SimpleCNN().to(device)
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
