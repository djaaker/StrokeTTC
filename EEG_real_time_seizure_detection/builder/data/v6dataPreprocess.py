import os
import glob
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Paths
trainPath = r"/Volumes/SDCARD/v2.0.3/edf/train"
evalPath = r"/Volumes/SDCARD/v2.0.3/edf/eval"
labelPath = r"EEG_real_time_seizure_detection/DiseaseLabels.csv"

# Read labels
labelFrame = pd.read_csv(labelPath)
print('This is label frame:\n', labelFrame)

labels = []
names = []
tests = []
noLabelCnt = 0

# Combine train and eval paths
for split in enumerate([os.listdir(trainPath), os.listdir(evalPath)]):
    test = split[0]
    subNames = split[1]

    path = trainPath if test == 0 else evalPath

    for sub in subNames:
        if len(labelFrame.loc[labelFrame['name'] == sub]) == 0:
            print("No label found for", sub)
            noLabelCnt += 1
        else:
            labels.append(labelFrame.loc[labelFrame['name'] == sub, 'label'].values[0])
            names.append(os.path.join(path, sub))  # Store the full path
            tests.append(test)

df = pd.DataFrame({'name': names, 'label': labels, 'test': tests})

# Set up args
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
    output_dim = 2

    target_channels = 40
    target_points = 5000
    segment_length = 500
    max_time_points = 100000

    samples_per_class = 200  # Adjust as needed
    verbose = True  # Set to True to enable verbose logging

args = Args()

# Collect EDF files and labels
healthyEdfs = []
seizEdfs = []

for subject in names:
    # Extract subject name from the full path
    subject_name = os.path.basename(subject)

    # Find the label corresponding to the subject from the DataFrame
    subject_label_row = df[df['name'].str.endswith(subject_name)]
    
    if subject_label_row.empty:
        print(f"No label found for subject {subject_name}")
        continue  # Skip this subject if no label is found

    subject_label = subject_label_row['label'].values[0]

    # Only include labels 0 and 3
    if subject_label == 0 or subject_label == 3:
        # Recursively find all .edf files under the subject's directory
        edf_files_in_subject = glob.glob(os.path.join(subject, '**', '*.edf'), recursive=True)
        
        if subject_label == 0:
            healthyEdfs.extend(edf_files_in_subject)
        elif subject_label == 3:
            seizEdfs.extend(edf_files_in_subject)
        
        # Optionally print information for debugging
        if args.verbose:
            print(f"Collected {len(edf_files_in_subject)} edf files for subject {subject_name} (Label: {subject_label})")
            if len(edf_files_in_subject) > 0:
                print("Sample edf files:")
                for f in edf_files_in_subject[:3]:
                    print(f)
    else:
        # Ignore other labels (1 and 2)
        continue

# Remove duplicates
healthyEdfs = list(set(healthyEdfs))
seizEdfs = list(set(seizEdfs))
edfFiles = healthyEdfs + seizEdfs

# Assign labels
edfLabels = [0] * len(healthyEdfs) + [1] * len(seizEdfs)

# Create DataFrame and shuffle
mlData = pd.DataFrame({'file': edfFiles, 'label': edfLabels})
mlData = mlData.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Adjust samples_per_class if needed
min_samples = min(len(healthyEdfs), len(seizEdfs))
if args.samples_per_class > min_samples:
    args.samples_per_class = min_samples
    print(f"Adjusted samples_per_class to {args.samples_per_class} due to limited data.")

# Balance the dataset
mlData = mlData.groupby('label', group_keys=False).apply(lambda x: x.sample(args.samples_per_class, random_state=args.seed)).reset_index(drop=True)

# Display counts
print("Healthy count:", len(mlData[mlData['label'] == 0]))
print("Seizure count:", len(mlData[mlData['label'] == 1]))
print("Total samples:", len(mlData))
# Check label distribution in labelFrame
label_counts = labelFrame['label'].value_counts()
print("Label counts in labelFrame:")
print(label_counts)


# Preprocessing functions
def EDFProcess(EDFFilePath):
    # Get raw file
    RawEEGDataFile = mne.io.read_raw_edf(EDFFilePath, preload=True, verbose=False)
    RawEEGDataFile.interpolate_bads(verbose=False)

    # Bandpass raw file
    BPEEGDataFile = BPFilter(RawEEGDataFile)

    # AD ratio raw file
    ADRatioDF = AlphaDeltaProcess(BPEEGDataFile)

    # Return processed file and AD ratio dataframe
    return BPEEGDataFile, ADRatioDF

# Bandpass Filtering
def BPFilter(RawEEGDataFile):
    BPEEGDataFile = RawEEGDataFile.copy().filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
    return BPEEGDataFile

# Alpha-Delta PSD Analysis and Data Framing
def AlphaDeltaProcess(EEGFile):
    AlphaComp = EEGFile.compute_psd(method='welch', fmin=8, fmax=12, tmin=None, tmax=None, picks='eeg',
                                    exclude=(), proj=False, remove_dc=True, reject_by_annotation=True,
                                    n_jobs=1, verbose='CRITICAL')
    AlphaPSD, _ = AlphaComp.get_data(return_freqs=True)

    DeltaComp = EEGFile.compute_psd(method='welch', fmin=0.5, fmax=4, tmin=None, tmax=None, picks='eeg',
                                    exclude=(), proj=False, remove_dc=True, reject_by_annotation=True,
                                    n_jobs=1, verbose='CRITICAL')
    DeltaPSD, _ = DeltaComp.get_data(return_freqs=True)

    ChanLab = EEGFile.ch_names

    AlphaMean = AlphaPSD.mean(axis=1)
    DeltaMean = DeltaPSD.mean(axis=1)

    AlDeRat = AlphaMean / DeltaMean

    PSDRatDF = pd.DataFrame({'Channel': ChanLab, 'Alpha Power': AlphaMean, 'Delta Power': DeltaMean,
                             'Alpha/Delta Ratio': AlDeRat})

    return PSDRatDF

# Detector_Dataset Class with integrated preprocessing
class Detector_Dataset(Dataset):
    def __init__(self, data_paths, labels, args, augment=None, data_type=None):
        self.data_paths = data_paths  # Paths to the raw data files
        self.labels = labels  # Corresponding labels for each data file
        self.args = args
        self.augment = augment
        self.data_type = data_type

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        label = self.labels[index]

        try:
            # Optionally print the data path being processed
            if self.args.verbose and index % 10 == 0:
                print(f"Processing file {index}: {data_path}")

            # Process the EDF file using the preprocessing functions
            BPEEGDataFile, _ = EDFProcess(data_path)

            # Get the preprocessed data
            data = BPEEGDataFile.get_data()

            # Limit the number of time points if they exceed max_time_points
            if data.shape[1] > self.args.max_time_points:
                data = data[:, :self.args.max_time_points]

            # Pad or trim channels to match target_channels
            if data.shape[0] < self.args.target_channels:
                padding = np.zeros((self.args.target_channels - data.shape[0], data.shape[1]))
                data = np.vstack((data, padding))
            elif data.shape[0] > self.args.target_channels:
                data = data[:self.args.target_channels, :]

            # Randomly select a segment
            if data.shape[1] > self.args.segment_length:
                start = np.random.randint(0, max(1, data.shape[1] - self.args.segment_length))
                end = start + self.args.segment_length
                segment = data[:, start:end]
            else:
                segment = data

            # Interpolate or compress each segment to match target_points
            if segment.shape[1] != self.args.target_points:
                segment = np.array([np.interp(np.linspace(0, 1, self.args.target_points),
                                              np.linspace(0, 1, segment.shape[1]), channel)
                                    for channel in segment])

            # Convert to tensor
            segment = torch.tensor(np.expand_dims(segment, axis=0), dtype=torch.float32)

            return segment, label, os.path.basename(data_path).split(".")[0]
        except Exception as e:
            print(f"Error processing file {data_path}: {e}")
            # Return a zero tensor if there's an error
            segment = torch.zeros((1, self.args.target_channels, self.args.target_points), dtype=torch.float32)
            return segment, label, os.path.basename(data_path).split(".")[0]

# Adjusted get_data_preprocessed function
def get_data_preprocessed(args, mlData):
    print("Preparing data for binary detector...")

    # Extract file paths and labels from mlData
    edf_files = mlData['file'].tolist()
    edf_labels = mlData['label'].tolist()

    # Encode labels to ensure they are zero-based indices
    le = LabelEncoder()
    edf_labels = le.fit_transform(edf_labels)

    # Check unique labels
    print("Unique labels in dataset:", le.classes_)

    # Use StratifiedShuffleSplit to split data while ensuring both classes are present
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    for train_index, test_index in sss1.split(edf_files, edf_labels):
        train_files_temp = [edf_files[i] for i in train_index]
        train_labels_temp = [edf_labels[i] for i in train_index]
        test_files = [edf_files[i] for i in test_index]
        test_labels = [edf_labels[i] for i in test_index]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    for train_index, val_index in sss2.split(train_files_temp, train_labels_temp):
        train_files = [train_files_temp[i] for i in train_index]
        train_labels = [train_labels_temp[i] for i in train_index]
        val_files = [train_files_temp[i] for i in val_index]
        val_labels = [train_labels_temp[i] for i in val_index]

    # Check class distributions
    print("Train labels distribution:", np.unique(train_labels, return_counts=True))
    print("Validation labels distribution:", np.unique(val_labels, return_counts=True))
    print("Test labels distribution:", np.unique(test_labels, return_counts=True))

    # Verify that training set contains more than one class
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training set contains only one class after splitting. Adjust the splitting strategy or random seed.")

    # Use WeightedRandomSampler for imbalanced classes
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    # Create datasets
    train_dataset = Detector_Dataset(train_files, train_labels, args)
    val_dataset = Detector_Dataset(val_files, val_labels, args)
    test_dataset = Detector_Dataset(test_files, test_labels, args)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=1, pin_memory=True)

    print("Number of training data:", len(train_dataset))
    print("Number of validation data:", len(val_dataset))
    print("Number of test data:", len(test_dataset))

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

# Now, call get_data_preprocessed with args and mlData
train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args, mlData)

# Set mne logging level to suppress excessive output
mne.set_log_level(verbose='WARNING')

# You can proceed with your training function using these data loaders
# Example of a training loop (adjust according to your model and requirements)
"""
model = YourModelClass()  # Define your model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)

for epoch in range(args.epochs):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_iter == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
"""
