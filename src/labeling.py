import numpy as np
import wfdb

class Labeler:
    def __init__(self):
        # MIT-BIH Beat Classifications
        # N: Normal
        # S: Supraventricular ectopic beat
        # V: Ventricular ectopic beat
        # F: Fusion beat
        # Q: Unknown beat
        
        # We classify 'N' as Normal (0), and everything else relevant as Abnormal (1)
        # Typically Normal includes: 'N', 'L', 'R', 'e', 'j'
        # Abnormal includes: 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'
        
        self.normal_codes = ['N', 'L', 'R', 'e', 'j']
        self.abnormal_codes = ['A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']

    def is_abnormal(self, symbol: str) -> bool:
        return symbol in self.abnormal_codes

    def get_labels(self, annotation: wfdb.Annotation, window_indices: list) -> np.ndarray:
        """
        Generate binary labels for each window.
        
        Args:
            annotation: WFDB Annotation object containing beat samples and symbols.
            window_indices: List of (start, end) sample indices for each window.
            
        Returns:
            labels: Numpy array of shape (num_windows,) with 0 (Normal) or 1 (Abnormal).
        """
        if annotation is None:
            # If no annotation, return all zeros or handle as error. 
            # Ideally we shouldn't train on unannotated data if we need labels.
            return np.zeros(len(window_indices))

        sample_indices = annotation.sample
        symbols = annotation.symbol
        
        labels = []
        
        # Optimize: Sort samples (usually already sorted)
        # For each window, find beats falling inside [start, end)
        
        for start, end in window_indices:
            # Find indices of beats within this window
            # using searchsorted is faster than iterating
            idx_start = np.searchsorted(sample_indices, start)
            idx_end = np.searchsorted(sample_indices, end)
            
            window_symbols = symbols[idx_start:idx_end]
            
            # Check if ANY abnormal beat is present
            is_window_abnormal = any(self.is_abnormal(sym) for sym in window_symbols)
            
            labels.append(1 if is_window_abnormal else 0)
            
        return np.array(labels)
