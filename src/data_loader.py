import wfdb
import os
import numpy as np
from typing import Tuple, Optional

class MITBIHDataLoader:
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory where the MIT-BIH dataset will be stored/loaded from.
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_data(self):
        """
        Download the MIT-BIH Arrhythmia Database from PhysioNet.
        """
        print(f"Downloading MIT-BIH Arrhythmia Database to {self.data_dir}...")
        try:
            wfdb.dl_database('mitdb', self.data_dir)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading database: {e}")

    def load_record(self, record_id: str) -> Tuple[np.ndarray, dict, Optional[wfdb.Annotation]]:
        """
        Load a specific ECG record and its annotation.

        Args:
            record_id: The ID of the record to load (e.g., '100').

        Returns:
            signals: The ECG signals as a numpy array.
            fields: A dictionary containing signal metadata.
            annotation: The annotation object if available, else None.
        """
        record_path = os.path.join(self.data_dir, record_id)
        
        try:
            record = wfdb.rdrecord(record_path)
            signals = record.p_signal
            fields = record.__dict__
            
            try:
                annotation = wfdb.rdann(record_path, 'atr')
            except FileNotFoundError:
                print(f"Warning: Annotation file not found for record {record_id}")
                annotation = None
                
            return signals, fields, annotation
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Record {record_id} not found in {self.data_dir}. Please run download_data() first.")

    def get_record_list(self) -> list:
        """
        Get list of all record IDs in the local dataset directory.
        Assumption: file names are like '100.dat', '100.hea', etc.
        """
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.hea')]
        record_ids = [f.split('.')[0] for f in files]
        return sorted(list(set(record_ids)))

if __name__ == "__main__":
    # Test stub
    loader = MITBIHDataLoader("data/raw")
    # loader.download_data() # Uncomment to download
    # if loader.get_record_list():
    #     sig, fields, ann = loader.load_record(loader.get_record_list()[0])
    #     print(f"Loaded record: {fields['record_name']}, Signal shape: {sig.shape}")
