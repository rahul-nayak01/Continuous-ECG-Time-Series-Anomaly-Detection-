import numpy as np
from scipy.signal import butter, filtfilt

class ECGPreprocessor:
    def __init__(self, fs: float = 360.0):
        """
        Initialize the preprocessor.
        
        Args:
            fs: Sampling frequency of the ECG signal (default 360Hz for MIT-BIH).
        """
        self.fs = fs

    def bandpass_filter(self, signal: np.ndarray, lowcut: float = 0.5, highcut: float = 50.0, order: int = 5) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to the signal.
        
        Args:
            signal: Input ECG signal (1D or 2D array [samples, channels]).
            lowcut: Lower cutoff frequency (Hz).
            highcut: Upper cutoff frequency (Hz).
            order: Order of the filter.
            
        Returns:
            Filtered signal.
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter along the first axis (time)
        # If signal is 2D (samples, channels), filtfilt defaults to axis=-1, 
        # but typically ECG is (samples, channels). 
        # Let's ensure we filter along time axis (usually axis 0 if shape is (N, ch)).
        
        is_1d = signal.ndim == 1
        if is_1d:
             y = filtfilt(b, a, signal)
        else:
             # Assuming shape (samples, channels), filter along axis 0
             y = filtfilt(b, a, signal, axis=0)
             
        return y

    def normalize(self, signal: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize the signal.
        
        Args:
            signal: Input signal.
            method: 'minmax' or 'zscore'.
            
        Returns:
            Normalized signal.
        """
        if method == 'minmax':
            # Min-Max scaling to [0, 1] per channel
            min_val = np.min(signal, axis=0)
            max_val = np.max(signal, axis=0)
            # Avoid division by zero
            denom = max_val - min_val
            denom[denom == 0] = 1.0 
            return (signal - min_val) / denom
            
        elif method == 'zscore':
            # Z-score standardization
            mean = np.mean(signal, axis=0)
            std = np.std(signal, axis=0)
            std[std == 0] = 1.0
            return (signal - mean) / std
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline: Filter -> Normalize.
        
        Args:
            signal: Raw ECG signal.
            
        Returns:
            Processed signal.
        """
        filtered = self.bandpass_filter(signal)
        normalized = self.normalize(filtered)
        return normalized

if __name__ == "__main__":
    # Test stub
    preprocessor = ECGPreprocessor()
    dummy_signal = np.random.randn(3600, 2) # 10 seconds, 2 leads
    processed = preprocessor.process(dummy_signal)
    print(f"Processed signal shape: {processed.shape}, Range: [{np.min(processed)}, {np.max(processed)}]")
