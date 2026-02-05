import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from src.data_loader import MITBIHDataLoader
from src.preprocessing import ECGPreprocessor
from src.windowing import WindowGenerator
from src.labeling import Labeler

class ECGDataset(Dataset):
    def __init__(self, data_dir: str, patient_ids: List[str] = None, window_seconds: float = 5.0, step_seconds: float = 2.5, fs: float = 360.0):
        """
        PyTorch Dataset for ECG Abnormality Detection.
        
        Args:
            data_dir: Path to MIT-BIH data.
            patient_ids: List of patient IDs to include (for train/test split). If None, loads all.
            window_seconds: Length of window.
            step_seconds: Step size for sliding window.
            fs: Sampling rate.
        """
        self.loader = MITBIHDataLoader(data_dir)
        self.preprocessor = ECGPreprocessor(fs=fs)
        self.window_gen = WindowGenerator(fs=fs, window_seconds=window_seconds, step_seconds=step_seconds)
        self.labeler = Labeler()
        
        self.windows = []
        self.labels = []
        
        # Download data if not exists (checked in loader)
        # self.loader.download_data() 
        
        all_records = self.loader.get_record_list()
        if not all_records:
            print("No records found. Attempting download...")
            self.loader.download_data()
            all_records = self.loader.get_record_list()
            
        if patient_ids:
            # Filter records
            self.records = [r for r in all_records if r in patient_ids]
        else:
            self.records = all_records
            
        self._prepare_data()

    def _prepare_data(self):
        print(f"Preparing data for {len(self.records)} records...")
        for record_id in tqdm(self.records):
            try:
                # Load
                signals, fields, annotation = self.loader.load_record(record_id)
                
                # Preprocess
                # signals shape: (samples, channels)
                processed_signals = self.preprocessor.process(signals)
                
                # Windowing values
                # shape: (n_windows, window_size, channels)
                windows = self.window_gen.segment_signal(processed_signals)
                
                # Windowing indices for labeling
                window_indices = self.window_gen.segment_metadata(processed_signals.shape[0])
                
                # Labeling
                labels = self.labeler.get_labels(annotation, window_indices)
                
                if len(windows) > 0:
                    self.windows.append(windows)
                    self.labels.append(labels)
                    
            except Exception as e:
                print(f"Error processing record {record_id}: {e}")
                
        if len(self.windows) > 0:
            self.windows = np.concatenate(self.windows, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            self.windows = np.array([])
            self.labels = np.array([])
            
        print(f"Data prepared. Total windows: {len(self.windows)}")
        
        # Check for data imbalance
        if len(self.labels) > 0:
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"Label distribution: {dict(zip(unique, counts))}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Window shape: (window_size, channels) -> (1800, 2)
        # PyTorch Conv1d expects (channels, window_size)
        
        window = self.windows[idx]
        label = self.labels[idx]
        
        # Transpose to (Channels, Length)
        window = window.transpose(1, 0)
        
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
