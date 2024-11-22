import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets
import mne
import os
# import boto3
# import sagemaker
# from sagemaker import get_execution_role
# from botocore.exceptions import NoCredentialsError
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
from torch.utils.data import Dataset, DataLoader


EDFInputPath = r'/Volumes/SDCARD/v2.0.3/edf/train'
EDFOutputPath = 'OutputFiles'

# import a csv file with the patient ID and the label
df = pd.read_csv(r'EEG_real_time_seizure_detection/DiseaseLabels.csv')

# make the voices go away
mne.set_log_level(verbose=False)

# run through all of the files present in the folder
def AllEDFProcess(EDFFolder):
    # if not os.path.exists(EDFOutputPath):
    #     os.makedirs(EDFOutputPath)
    
    # for FileName in os.listdir(EDFFolder):
    #     if FileName.endswith('.edf'):
    #         EDFFilePath = os.path.join(EDFFolder, FileName)
    #         EDFProcess(EDFFilePath)
    # EDFFiles = list_edf_files_from_s3(BucketName, EDFFolder)
    DataFiles = []
    PSD_Array = []
    display(EDFFolder)
    processed_data_array = []
    # for all of the files in the local folder, find the ones that have .edf endings and then create the overall directory to get them, add the name to a list
    # for FileName in os.listdir(EDFFolder):
    #       if FileName.endswith('.edf'):
    #          EDFFilePath = os.path.join(EDFFolder, FileName)
    #          #processed_data, PSD_data = EDFProcess(EDFFilePath)
    #          DataFiles.append(FileName)
    #          #processed_data_array.append(processed_data)
    #          #PSD_Array.append(PSD_data)
    #          #return processed_data, PSD_data, EEG_image, DataFiles
    for root, dirs, files in os.walk(EDFFolder):
        for file in files:
            if file.endswith('.edf'):
                EDFFilePath = os.path.join(root, file)
                DataFiles.append(EDFFilePath)
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    #display(DataFiles)
    # display(root)
    # display(dirs)
    # display(files)
    BPEEGDataFiles = []
    label = []
    # run through each file individually
    for f in DataFiles:
        print('file ran')
        display(f)
        # split the file parts by the underscores
        FileParts = f.split('/')
        #display(FileParts)
        #display(FileParts)
        # find the patient ID (this could change based on your file structure)
        patient_id = FileParts[6]
        #print(patient_id)
        # check if one of the dataframe names matches the patient ID
        label_row = df[df['name'] == patient_id]

        # if it does (i.e. if the row isn't empty)
        if not label_row.empty:
            # Add the label name
            label = label_row.iloc[0]['label']
            # create the complete file pathway name
            ff = os.path.join(EDFFolder, f)
            # run the edf process
            BPEEGDataFile, ADRatioDF = EDFProcess(ff)
            # add the processed data and the label to a list
            BPEEGDataFiles.append((BPEEGDataFile.get_data(), label))
            # add the AD ratio data to a list
            PSD_Array.append(ADRatioDF)
            
            print(f"Label for Patient ID '{patient_id}': {label}")
        else:
            print(f"No label found for Patient ID '{patient_id}'")
    
    return BPEEGDataFiles, ADRatioDF
                
# Process the edf files
def EDFProcess(EDFFilePath):
    # get raw file
    RawEEGDataFile = mne.io.read_raw_edf(EDFFilePath, preload=True, verbose=False)
    RawEEGDataFile.interpolate_bads(verbose=False)

    

    # bandpass raw file
    BPEEGDataFile = BPFilter(RawEEGDataFile)
    
    
    # AD ratio raw file
    ADRatioDF = AlphaDeltaProcess(BPEEGDataFile)
    
    # return processed file and AD ratio dataframe
    return BPEEGDataFile, ADRatioDF

# Bandpass Filtering
def BPFilter(RawEEGDataFile):
    BPEEGDataFile = RawEEGDataFile.copy().filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
    return BPEEGDataFile

## ALPHA DELTA PSD ANALYSIS AND DATA FRAMING ##
def AlphaDeltaProcess(EEGFile):
    AlphaComp = EEGFile.compute_psd(method='welch', fmin=8, fmax=12, tmin=None, tmax=None, picks='eeg', exclude=(), proj=False, remove_dc=True, reject_by_annotation=True, n_jobs=1, verbose='CRITICAL')
    AlphaPSD, AlphaFreq = AlphaComp.get_data(return_freqs=True)
    #display(AlphaComp)
    DeltaComp = EEGFile.compute_psd(method='welch', fmin=0.5, fmax=4, tmin=None, tmax=None, picks='eeg', exclude=(), proj=False, remove_dc=True, reject_by_annotation=True, n_jobs=1, verbose='CRITICAL')
    DeltaPSD, DeltaFreq = DeltaComp.get_data(return_freqs=True)
    #DeltaComp.plot()
    #raw_csd = mne.preprocessing.compute_current_source_density(RawEEGDataFile);

    ChanLab = EEGFile.ch_names

    AlphaMean = AlphaPSD.mean(axis=1)
    DeltaMean = DeltaPSD.mean(axis=1)

    AlDeRat = AlphaMean / DeltaMean

    PSDRatDF = pd.DataFrame({'Channel': ChanLab,'Alpha Power': AlphaMean,'Delta Power': DeltaMean,'Alpha/Delta Ratio': AlDeRat})

    #display(PSDRatDF)
    return PSDRatDF

# Run the function
BPEEGDataFiles, ADRatioDF = AllEDFProcess(EDFInputPath)

class Detector_Dataset(Dataset):
    def __init__(self, data_paths, labels, args):
        self.data_paths = data_paths  # Paths to the preprocessed data files
        self.labels = labels  # Corresponding labels for each data file
        self.args = args  # Configuration arguments (e.g., window size, eeg_type)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        # Load preprocessed EEG data from .edf file
        data_path = self.data_paths[index]
        raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)  # Load the EDF file
        signals = raw.get_data()  # Extract the raw EEG signals as a NumPy array
        label = self.labels[index]

        # Depending on the configuration, apply bipolar or unipolar processing
        # if self.args.eeg_type == "bipolar":
        #     signals = bipolar_signals_func(signals)  # Apply bipolar processing
        #     signals = torch.tensor(signals)  # Convert to tensor
        # elif self.args.eeg_type == "uni_bipolar":
        #     bipolar_signals = bipolar_signals_func(signals)
        #     signals = torch.cat((torch.tensor(signals), torch.tensor(bipolar_signals)))  # Combine unipolar and bipolar signals
        # else:
        signals = torch.tensor(signals)

        return signals, label, data_path.split("/")[-1].split(".")[0]

def eeg_binary_collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # Sort batch by sequence length in descending order
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    seq_lengths = torch.IntTensor([len(sample[0]) for sample in batch])
    target_lengths = torch.IntTensor([1 for sample in batch])  # Assuming binary labels, can adjust based on actual use
    batch_size = len(batch)

    # Determine maximum sequence length in the batch
    max_seq_length = seq_lengths.max().item()
    eeg_type_size = len(batch[0][0])  # Number of EEG channels (e.g., 20 for bipolar, 21 for unipolar+bipolar)

    # Prepare zero-padded tensors for EEG sequences and targets
    seqs = torch.zeros(batch_size, max_seq_length, eeg_type_size)
    targets = torch.zeros(batch_size).to(torch.long)
    signal_name_list = []

    for i in range(batch_size):
        sample = batch[i]
        signals, label, signal_name = sample
        seq_length = len(signals)

        # Copy the EEG signals into the corresponding batch tensor, with padding where needed
        seqs[i, :seq_length] = signals.permute(1, 0)  # Adjusting dimension to match expected batch format
        targets[i] = label
        signal_name_list.append(signal_name)

    return seqs, targets, seq_lengths, target_lengths, [], signal_name_list

args = {
    'batch_size': 32,
    'eeg_type': 'unipolar',
    'window_size': 4,
    'window_shift': 1,
    'sample_rate': 256,
    'augmentation': False
}

# Set the columns to the signal and label data
display(BPEEGDataFiles)
SignalData = [row[0] for row in BPEEGDataFiles]
LabelData = [row[1] for row in BPEEGDataFiles]

Detector_Dataset(SignalData, LabelData, args)
#for batch in DataLoader:
 #   sequences, targets, seq_lengths, target_lengths, aug_list, signal_name_list = batch

# Assuming you have the following:
# - preprocessed_data_paths: List of paths to the preprocessed EEG data files.
# - labels: Corresponding labels for each preprocessed file.
# - args: Configuration arguments (contains settings like eeg_type, batch_size, etc.).


def get_data_preprocessed(args, mode="train"):
   
    print("Preparing data for bianry detector...")
    train_data_path = args.data_path + "/dataset-tuh_task-binary_datatype-train_v6"
    # dev_data_path = args.data_path + "/dataset-tuh_task-binary_datatype-dev_v6"
    dev_data_path = args.data_path + "/dataset-tuh_task-binary_noslice_datatype-dev_v6"
    train_dir = search_walk({"path": train_data_path, "extension": ".pkl"})
    dev_dir = search_walk({"path": dev_data_path, "extension": ".pkl"})
    random.shuffle(train_dir)
    random.shuffle(dev_dir)

    aug_train = ["0"] * len(train_dir)
    if args.augmentation == True:
        train_dir += train_dir
        aug_train = ["1"] * len(train_dir)

    half_dev_num = int(len(dev_dir) // 2)
    val_dir = dev_dir[:half_dev_num] 
    test_dir = dev_dir[half_dev_num:]
    aug_val = ["0"] * len(val_dir)
    aug_test = ["0"] * len(test_dir)


    train_data = Detector_Dataset(args, data_pkls=train_dir, augment=aug_train, data_type="training dataset")
    class_sample_count = np.unique(train_data._type_list, return_counts=True)[1]
    weight = 1. / class_sample_count
    ########## Change Dataloader Sampler Rate for each class Here ##########
    # abnor_nor_ratio = len(class_sample_count)-1
    # weight[0] = weight[0] * abnor_nor_ratio
    if args.binary_sampler_type == "6types":
        patT_idx = (train_data.type_type).index("patT")
        patF_idx = (train_data.type_type).index("patF")
        # weight[patT_idx] = weight[patT_idx] * 2
        # weight[patF_idx] = weight[patF_idx] * 2
    elif args.binary_sampler_type == "30types":
        patT_idx = (train_data.type_type).index("0_patT")
        patF_idx = (train_data.type_type).index("0_patF")
        # weight[patT_idx] = weight[patT_idx] * 14
        # weight[patF_idx] = weight[patF_idx] * 14
        weight[patT_idx] = weight[patT_idx] * 7
        weight[patF_idx] = weight[patF_idx] * 7
    else:
        print("No control on sampler rate")
    ########################################################################
    samples_weight = weight[train_data._type_list]
    # print("samples_weight: ", samples_weight)
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    val_data = Detector_Dataset(args, data_pkls=val_dir, augment=aug_val, data_type="validation dataset")
    test_data = Detector_Dataset(args, data_pkls=test_dir, augment=aug_test, data_type="test dataset")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, sampler=sampler, collate_fn=eeg_binary_collate_fn)     
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, collate_fn=eeg_binary_collate_fn)               
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, collate_fn=eeg_binary_collate_fn)  


    info_dir = train_data_path + "/preprocess_info.infopkl"
    with open(info_dir, 'rb') as _f:
        data_info = pkl.load(_f)

        args.disease_labels = data_info["disease_labels"]
        args.disease_labels_inv = data_info["disease_labels_inv"]
        
        args.sample_rate = data_info["sample_rate"]
        args.feature_sample_rate = data_info["feature_sample_rate"]
        
        args.disease_type = data_info["disease_type"]

        args.target_dictionary = data_info["target_dictionary"]
        args.selected_diseases = data_info["selected_diseases"]
        
        args.window_size_label = args.feature_sample_rate * args.window_size
        args.window_shift_label = args.feature_sample_rate * args.window_shift
        args.window_size_sig = args.sample_rate * args.window_size
        args.window_shift_sig = args.sample_rate * args.window_shift
        
        args.fsr_sr_ratio = (args.sample_rate // args.feature_sample_rate)


    with open(train_dir[0], 'rb') as _f:
        data_pkl = pkl.load(_f)
        signals = data_pkl['RAW_DATA'][0]

    if args.eeg_type == "bipolar":
        args.num_channel = 20
    elif args.eeg_type == "uni_bipolar":
        args.num_channel = 20 + signals.size(0)
    else:
        args.num_channel = signals.size(0)
    
    ############################################################
    print("Number of training data: ", len(train_dir))
    print("Number of validation data: ", len(val_dir))
    print("Number of test data: ", len(test_dir))
    print("Selected seizures are: ", args.seiz_classes)
    print("Selected task type is: ", args.task_type)
    if args.task_type == "binary":
        print("Selected binary group is: ", args.num_to_seizure)
        print("Selected sampler type: ", args.binary_sampler_type)
        print("Max number of normal slices per patient: ", str(args.dev_bckg_num))
    print("label_sample_rate: ", args.feature_sample_rate)
    print("raw signal sample_rate: ", args.sample_rate)
    print("Augmentation: ", args.augmentation)     
    
    return train_loader, val_loader, test_loader, len(train_data._data_list), len(val_data._data_list), len(test_data._data_list)