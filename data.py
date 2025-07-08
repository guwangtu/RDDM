import torch
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
import sklearn.preprocessing as skp
from torch.utils.data import Dataset, DataLoader

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ecg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)

def get_WESAD_RAW(
    DATA_PATH = "WESAD",   
    ):
    ecg_list=[]
    ppg_list=[]
    for folder_name in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder_name)
        #print(folder_name)
        if "."in folder_name:
            continue
        if os.path.isdir(folder_path):
            # 加载BVP.csv的第一列
            bvp_path = os.path.join(folder_path, 'BVP.csv')
            if os.path.exists(bvp_path):
                bvp_df = pd.read_csv(bvp_path)
                bvp_data = bvp_df.iloc[:, 0].values[1:]  # 第一列数据不要第一个
                #print(bvp_data.shape) 
                ppg_list.append(bvp_data)
            # 加载对应的pkl文件中的"ecg"
            pkl_path = os.path.join(folder_path, f"{folder_name}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f, encoding='latin1')
                ecg_data = pkl_data['signal']['chest']['ECG']
                if ecg_data is not None: 
                    ecg_data = np.array(ecg_data).squeeze()
                    #print(ecg_data.shape)    
                    ecg_list.append(ecg_data)
    print(len(ecg_list), len(ppg_list))
    ppg_train_list, ppg_test_list, ecg_train_list, ecg_test_list = train_test_split(ppg_list, ecg_list, test_size=0.2, random_state=42)
    
    ecg_train = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
    ppg_train = np.nan_to_num(np.concatenate(ppg_train_list).astype("float32"))
    ecg_test = np.nan_to_num(np.concatenate(ecg_test_list).astype("float32"))
    ppg_test = np.nan_to_num(np.concatenate(ppg_test_list).astype("float32"))

    dataset_train = ECGDataset(
        skp.minmax_scale(ecg_train, (-1, 1), axis=1),
        skp.minmax_scale(ppg_train, (-1, 1), axis=1)
    )
    dataset_test = ECGDataset(
        skp.minmax_scale(ecg_test, (-1, 1), axis=1),
        skp.minmax_scale(ppg_test, (-1, 1), axis=1)
    )
    return dataset_train, dataset_test

def get_datasets(
    DATA_PATH = "../../ingenuity_NAS/21ds94_nas/21ds94_mount/AAAI24/datasets/", 
    datasets=["BIDMC", "CAPNO", "DALIA", "MIMIC-AFib", "WESAD"],
    window_size=4,
    ):

    ecg_train_list = []
    ppg_train_list = []
    ecg_test_list = []
    ppg_test_list = []
    
    for dataset in datasets:

        ecg_train = np.load(DATA_PATH + dataset + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_train = np.load(DATA_PATH + dataset + f"/ppg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        
        ecg_test = np.load(DATA_PATH + dataset + f"/ecg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_test = np.load(DATA_PATH + dataset + f"/ppg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)

        ecg_train_list.append(ecg_train)
        ppg_train_list.append(ppg_train)
        ecg_test_list.append(ecg_test)
        ppg_test_list.append(ppg_test)

    ecg_train = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
    ppg_train = np.nan_to_num(np.concatenate(ppg_train_list).astype("float32"))

    ecg_test = np.nan_to_num(np.concatenate(ecg_test_list).astype("float32"))
    ppg_test = np.nan_to_num(np.concatenate(ppg_test_list).astype("float32"))

    dataset_train = ECGDataset(
        skp.minmax_scale(ecg_train, (-1, 1), axis=1),
        skp.minmax_scale(ppg_train, (-1, 1), axis=1)
    )
    dataset_test = ECGDataset(
        skp.minmax_scale(ecg_test, (-1, 1), axis=1),
        skp.minmax_scale(ppg_test, (-1, 1), axis=1)
    )

    return dataset_train, dataset_test