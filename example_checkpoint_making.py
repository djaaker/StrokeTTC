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

# Simple CNN model
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

# Training loop
total_steps = len(train_loader)
logging.info("Starting training...")
for epoch in range(num_epochs):
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
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
            # Write to TensorBoard
            writer.add_scalar('training loss', loss.item(), epoch * total_steps + i)

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Calculate accuracy
        accuracy = 100 * correct / total
        logging.info(f'Validation Accuracy after Epoch {epoch+1}: {accuracy:.2f}%')
        writer.add_scalar('validation accuracy', accuracy, epoch+1)

        # Calculate precision, recall, FPR, FNR
        cm = confusion_matrix(all_labels, all_predictions)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Precision, Recall, FPR, FNR per class
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)

        # Handle divide by zero
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        FPR = np.nan_to_num(FPR)
        FNR = np.nan_to_num(FNR)

        # Average metrics
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_FPR = np.mean(FPR)
        avg_FNR = np.mean(FNR)

        logging.info(f'Validation Precision after Epoch {epoch+1}: {avg_precision:.4f}')
        logging.info(f'Validation Recall after Epoch {epoch+1}: {avg_recall:.4f}')
        logging.info(f'Validation False Positive Rate after Epoch {epoch+1}: {avg_FPR:.4f}')
        logging.info(f'Validation False Negative Rate after Epoch {epoch+1}: {avg_FNR:.4f}')

        # Write to TensorBoard
        writer.add_scalar('validation precision', avg_precision, epoch+1)
        writer.add_scalar('validation recall', avg_recall, epoch+1)
        writer.add_scalar('validation FPR', avg_FPR, epoch+1)
        writer.add_scalar('validation FNR', avg_FNR, epoch+1)

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f'Model checkpoint saved to {checkpoint_path}')

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    logging.info(f'Test Accuracy: {accuracy:.2f}%')

    # Calculate precision, recall, FPR, FNR
    cm = confusion_matrix(all_labels, all_predictions)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Precision, Recall, FPR, FNR per class
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)

    # Handle divide by zero
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    FPR = np.nan_to_num(FPR)
    FNR = np.nan_to_num(FNR)

    # Average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_FPR = np.mean(FPR)
    avg_FNR = np.mean(FNR)

    logging.info(f'Test Precision: {avg_precision:.4f}')
    logging.info(f'Test Recall: {avg_recall:.4f}')
    logging.info(f'Test False Positive Rate: {avg_FPR:.4f}')
    logging.info(f'Test False Negative Rate: {avg_FNR:.4f}')

    # Write to TensorBoard
    writer.add_scalar('test accuracy', accuracy)
    writer.add_scalar('test precision', avg_precision)
    writer.add_scalar('test recall', avg_recall)
    writer.add_scalar('test FPR', avg_FPR)
    writer.add_scalar('test FNR', avg_FNR)

# Test on a single data point
# Let's take the first image from the test set
single_image, single_label = test_dataset[0]
model.eval()
with torch.no_grad():
    single_image_input = single_image.to(device).unsqueeze(0)  # Add batch dimension
    output = model(single_image_input)
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.cpu().numpy()[0]  # Convert to numpy array

    # Get predicted label
    _, predicted = torch.max(output.data, 1)
    predicted_label = predicted.item()
    actual_label = single_label

    # Calculate accuracy (1 if correct, 0 if incorrect)
    is_correct = int(predicted_label == actual_label)
    accuracy = 100 * is_correct

    logging.info('Single Data Point Test:')
    logging.info(f'Actual Label: {actual_label}')
    logging.info(f'Predicted Label: {predicted_label}')
    logging.info(f'Accuracy: {accuracy}%')
    logging.info('Class Probabilities:')
    for i, prob in enumerate(probabilities):
        logging.info(f'Class {i}: {prob*100:.2f}%')

    # Visualize the input image and prediction in TensorBoard
    # Unnormalize the image for visualization
    img = single_image.numpy() * 0.3081 + 0.1307  # Unnormalize

    # Add image and prediction to TensorBoard
    writer.add_image('Single Test Image', img, dataformats='CHW')
    writer.add_text('Single Test Prediction',
                    f'Actual Label: {actual_label}, Predicted Label: {predicted_label}, Accuracy: {accuracy}%')

    # Plot the probabilities as a bar graph
    fig, ax = plt.subplots()
    classes = list(range(10))
    ax.bar(classes, probabilities)
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_title('Class Probabilities')
    plt.xticks(classes)
    plt.tight_layout()

    # Add the figure to TensorBoard
    writer.add_figure('Class Probabilities', fig)

    # Optionally, save the figure as an image file
    # plt.savefig('class_probabilities.png')

logging.info("Training and evaluation completed.")

# Close the TensorBoard writer
writer.close()
