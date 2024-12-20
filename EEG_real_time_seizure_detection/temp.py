import matplotlib.pyplot as plt
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

s3 = boto3.client('s3')
BucketName = 'seniordesignt6'
EDFInputPath = 'tuh_eeg/v2.0.1/edf/'
EDFOutputPath = 'OutputFiles'

# Temporary processing directory
temp_dir = '/home/sagemaker-user'

# def find_all_edf_files(bucket_name, prefix=''):
#     s3 = boto3.client('s3')
    
    
    
    
#     return edf_files

def list_edf_files_from_s3(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    # Use paginator to handle large number of objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    edf_files = []
    
    for page in paginator.paginate(Bucket=BucketName, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Check if the object key ends with .edf
                if key.endswith('.edf'):
                    edf_files.append(key)
    # files = [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith('.edf')]
    return edf_files

def download_file_from_s3(bucket, s3_key, local_path):
    try:
        s3.download_file(bucket, s3_key, local_path)
        print(f"Downloaded {s3_key} to {local_path}")
    except NoCredentialsError:
        print("Credentials not available")

def upload_file_to_s3(local_path, bucket, s3_key):
    try:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except NoCredentialsError:
        print("Credentials not available")


def AllEDFProcess(EDFFolder):
    # if not os.path.exists(EDFOutputPath):
    #     os.makedirs(EDFOutputPath)
    
    # for FileName in os.listdir(EDFFolder):
    #     if FileName.endswith('.edf'):
    #         EDFFilePath = os.path.join(EDFFolder, FileName)
    #         EDFProcess(EDFFilePath)
    EDFFiles = list_edf_files_from_s3(BucketName, EDFFolder)
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    print(len(EDFFiles))

    for s3_key in EDFFiles:
        print('file ran')
        local_file = os.path.join(temp_dir, os.path.basename(s3_key))
        download_file_from_s3(BucketName, s3_key, local_file)
        BPEEGDataFile, ADRatioDF = EDFProcess(local_file)
            
            

def EDFProcess(EDFFilePath):
    RawEEGDataFile = mne.io.read_raw_edf(EDFFilePath, preload=True)
    RawEEGDataFile.interpolate_bads()

    print('it works')

    
    BPEEGDataFile = BPFilter(RawEEGDataFile)
    
    # EOGDataFile = find_eog_events(BPEEGDataFile, ch_name=["EEG FP1-REF", "EEG FZ-REF"])
    # EOGOnset = EOGDataFile[:, 0] / BPEEGDataFile.info["sfreq"] - 0.25
    # EOGDuration = [0.5] * len(EOGDataFile)
    # EOGDescr = ["bad blink"] * len(EOGDataFile)
    # blink_annot = mne.Annotations(EOGOnset, EOGDuration, EOGDescr, orig_time=BPEEGDataFile.info["meas_date"])
    # BPEEGDataFile.set_annotations(blink_annot)

    #OutputFileName = f"filtered_{os.path.splitext(os.path.basename(EDFFilePath))[0]}.fif"
    # OutputFile = os.path.join(EDFOutputPath, OutputFileName)
    #OutputLocalPath = os.path.join(temp_dir, OutputFileName)
   # BPEEGDataFile.save(OutputFile, overwrite=True)
    # RawEEGDataFile.plot(duration=200, start=100)
    # BPEEGDataFile.plot(duration=200, start=100)
    
    ADRatioDF = AlphaDeltaProcess(BPEEGDataFile)
    
    #PSDOutputFileName = f"PSD_{os.path.splitext(os.path.basename(EDFFilePath))[0]}.csv"
    # PSDOutputFile = os.path.join(EDFOutputPath, PSDOutputFileName)
    #PSDOutputLocalPath = os.path.join(temp_dir, PSDOutputFileName)
    #ADRatioDF.to_csv(PSDOutputLocalPath, index=False)

    ###NEW####
    # Upload filtered data and analysis results to S3
    #upload_file_to_s3(OutputLocalPath, bucket_name, f"{EDFOutputPath}/{OutputFileName}")
    #upload_file_to_s3(PSDOutputLocalPath, bucket_name, f"{EDFOutputPath}/{PSDOutputFileName}")

    #print(f"Finished and saved file {EDFFilePath} to {OutputFile}")
    #print(f"Finished and saved PSD data to {PSDOutputFile}")
    return BPEEGDataFile, ADRatioDF

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



AllEDFProcess(EDFInputPath)

# RawEEGDataFile = mne.io.read_raw_edf("aaaaaaaa_s001_t000.edf", preload=True)
# raw_data = RawEEGDataFile.get_data()
# info = RawEEGDataFile.info
# channels = RawEEGDataFile.ch_names

# RawEEGDataFile = RawEEGDataFile.pick(picks=["eeg", "eog", "ecg", "stim"]).load_data();
# RawEEGDataFile.interpolate_bads();
# RawEEGDataFile.get_data().shape


#display(raw_data)
#display(info)

# data_path = sample.data_path()
# subjects_dir = data_path / "sample"
# meg_path = data_path / "MEG" / "sample"
# raw_fname = meg_path / "sample_audvis_raw.fif"
# fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

# Load real data as the template
#RawEEGDataFile = mne.io.read_raw_fif(raw_fname, preload=True);
#RawEEGDataFile.set_eeg_reference(projection=True);
# RawEEGDataFile = RawEEGDataFile.pick(picks=["eeg", "eog", "ecg", "stim"]).load_data();
# RawEEGDataFile.interpolate_bads();
# RawEEGDataFile.get_data().shape
# EEGChannelNum = input("Please input the number of channels:")
#Freq = float(input("Please input sample refresh rate in Hz:"))
# RawEEGDataFile = input("Please input the EEG data file name:")



#raw_csd.plot();

#raw.compute_psd().plot(picks="data", exclude="bads", amplitude=False);

#raw_csd.compute_psd().plot(picks="data", exclude="bads", amplitude=False);
