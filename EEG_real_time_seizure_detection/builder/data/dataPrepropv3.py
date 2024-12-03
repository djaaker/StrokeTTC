import os
import glob
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

class DetectorDataset(Dataset):
    def __init__(self, data_paths, labels, args):
        self.data_paths = data_paths
        self.labels = labels
        self.args = args
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        label = self.labels[index]

        # Load EEG data
        raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        data = raw.get_data()

        # Preprocess data
        # Limit time points
        if data.shape[1] > self.args['max_time_points']:
            data = data[:, :self.args['max_time_points']]

        # Pad or trim channels
        if data.shape[0] < self.args['target_channels']:
            padding = np.zeros((self.args['target_channels'] - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))
        elif data.shape[0] > self.args['target_channels']:
            data = data[:self.args['target_channels'], :]

        # Random segment
        if data.shape[1] > self.args['segment_length']:
            start = np.random.randint(0, data.shape[1] - self.args['segment_length'])
            data = data[:, start:start + self.args['segment_length']]

        # Interpolate to target points
        if data.shape[1] != self.args['target_points']:
            data = np.array([np.interp(np.linspace(0, 1, self.args['target_points']),
                                       np.linspace(0, 1, data.shape[1]), channel)
                             for channel in data])

        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)

        return data, label

def get_data_loaders(ml_data, args):
    # Split data into train, validation, and test sets
    train_df, test_df = train_test_split(ml_data, test_size=0.2, stratify=ml_data['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'], random_state=42)  # 0.25 x 0.8 = 0.2

    # Create datasets
    train_dataset = DetectorDataset(train_df['file'].tolist(), train_df['label'].tolist(), args)
    val_dataset = DetectorDataset(val_df['file'].tolist(), val_df['label'].tolist(), args)
    test_dataset = DetectorDataset(test_df['file'].tolist(), test_df['label'].tolist(), args)

    # Compute class weights for imbalance
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_df['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# Paths
trainPath = "/Volumes/SDCARD/v2.0.3/edf/train"
evalPath = "/Volumes/SDCARD/v2.0.3/edf/eval"
labelPath = "EEG_real_time_seizure_detection/DiseaseLabels.csv"

# Read labels
label_frame = pd.read_csv(labelPath)

# Collect subject paths
all_subjects = []
for path in [trainPath, evalPath]:
    subjects = os.listdir(path)
    all_subjects.extend([os.path.join(path, subj) for subj in subjects])

# Match subjects with labels
names = []
labels = []

for subj_path in all_subjects:
    subj_name = os.path.basename(subj_path)
    label_row = label_frame[label_frame['name'] == subj_name]
    if not label_row.empty:
        names.append(subj_path)
        labels.append(label_row.iloc[0]['label'])
    else:
        print(f"No label found for {subj_name}")

# Collect EDF files and labels
edf_files = []
edf_labels = []

for name, label in zip(names, labels):
    subject_edf_files = glob.glob(os.path.join(name, '**', '*.edf'), recursive=True)
    edf_files.extend(subject_edf_files)
    edf_labels.extend([label] * len(subject_edf_files))

# Remove duplicates
edf_files, unique_indices = np.unique(edf_files, return_index=True)
edf_labels = np.array(edf_labels)[unique_indices].tolist()

# Create DataFrame and balance dataset
ml_data = pd.DataFrame({'file': edf_files, 'label': edf_labels})
ml_data = ml_data.sample(frac=1).reset_index(drop=True)
ml_data = ml_data.groupby('label').head(30)
ml_data['label'] = ml_data['label'].replace(3, 1)

# Args
args = {
    'batch_size': 32,
    'target_channels': 40,
    'target_points': 5000,
    'segment_length': 500,
    'max_time_points': 100000,
}

# Data Loaders
train_loader, val_loader, test_loader = get_data_loaders(ml_data, args)
