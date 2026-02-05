import numpy as np

class WindowGenerator:
    def __init__(self, fs: float = 360.0, window_seconds: float = 5.0, step_seconds: float = 2.5):
        """
        Initialize the window generator.
        
        Args:
            fs: Sampling frequency (Hz).
            window_seconds: Length of each window in seconds.
            step_seconds: Step size for sliding window in seconds (overlap = window - step).
        """
        self.fs = fs
        self.window_size = int(fs * window_seconds)
        self.step_size = int(fs * step_seconds)

    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Segment the continuous signal into windows.
        
        Args:
            signal: Continuous ECG signal (samples, channels) or (samples,).
            
        Returns:
            windows: Array of shape (num_windows, window_size, channels).
        """
        num_samples = signal.shape[0]
        if num_samples < self.window_size:
            return np.array([])

        # Use stride tricks for efficient sliding window views if 1D, 
        # but for 2D generic approach is safer or iterative. 
        # Here we use a simple iterative approach for clarity and compatibility with channels.
        
        windows = []
        for start in range(0, num_samples - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append(signal[start:end])
            
        return np.array(windows)

    def segment_metadata(self, num_samples: int) -> list:
        """
        Generate (start_index, end_index) tuples for each window.
        Useful for mapping annotations.
        """
        if num_samples < self.window_size:
            return []

        indices = []
        for start in range(0, num_samples - self.window_size + 1, self.step_size):
            end = start + self.window_size
            indices.append((start, end))
            
        return indices
