import torch
import numpy as np
import os
from src.model import ECGCNN, ECGAutoencoder
from src.preprocessing import ECGPreprocessor

class ECGInference:
    def __init__(self, model_path: str, model_type: str = 'cnn', device: str = None):
        """
        Inference engine for ECG models.
        
        Args:
            model_path: Path to the .pth model file.
            model_type: 'cnn' or 'autoencoder'.
            device: 'cpu' or 'cuda'.
        """
        self.model_type = model_type
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load Model
        if model_type == 'cnn':
            self.model = ECGCNN()
        else:
            self.model = ECGAutoencoder()
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded {model_type} model from {model_path}")
        else:
            print(f"Warning: Model path {model_path} does not exist. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = ECGPreprocessor()

    def predict(self, signal_window: np.ndarray):
        """
        Perform inference on a single window.
        
        Args:
            signal_window: Raw ECG signal window (samples, channels).
            
        Returns:
            dict: Prediction result.
        """
        # Preprocess
        # Shape: (samples, channels)
        processed = self.preprocessor.process(signal_window)
        
        # Prepare Tensor
        # Model expects (Batch, Channels, Length)
        # Transpose to (Channels, Length)
        tensor = torch.tensor(processed.transpose(1, 0), dtype=torch.float32).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            
            if self.model_type == 'cnn':
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                return {
                    "type": "classification",
                    "prediction": "Abnormal" if predicted_class == 1 else "Normal",
                    "confidence": confidence,
                    "probabilities": probabilities.cpu().numpy().tolist()
                }
            else:
                # Autoencoder: Calculate reconstruction error (MSE)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(output, tensor)
                mse = loss.item()
                
                # Threshold logic would go here (e.g. if mse > threshold then Abnormal)
                return {
                    "type": "reconstruction",
                    "mse": mse,
                    "is_anomaly": mse > 0.1 # Placeholder threshold
                }
