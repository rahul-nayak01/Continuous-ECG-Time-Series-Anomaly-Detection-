import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    def __init__(self, num_channels: int = 2, num_classes: int = 2):
        """
        1D Convolutional Neural Network for ECG Classification.
        
        Args:
            num_channels: Number of input ECG leads/channels.
            num_classes: Number of output classes (2 for Binary: Normal vs Abnormal).
        """
        super(ECGCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Block 4
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        # Fully Connected
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten: (Batch, 256, 1) -> (Batch, 256)
        x = x.squeeze(-1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ECGAutoencoder(nn.Module):
    def __init__(self, num_channels: int = 2, length: int = 1800):
        """
        Convolutional Autoencoder for ECG Reconstruction.
        
        Args:
            num_channels: Input channels.
            length: Sequence length (needs to be carefully handled for unpooling/upsampling).
        """
        super(ECGAutoencoder, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2, stride=2) # L/2
        self.enc_relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2) # L/4
        self.enc_relu2 = nn.ReLU()
        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2) # L/8
        self.enc_relu3 = nn.ReLU()
        
        # Decoder (Mirroring Encoder)
        self.dec_convtrans1 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.dec_relu1 = nn.ReLU()
        self.dec_convtrans2 = nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.dec_relu2 = nn.ReLU()
        self.dec_convtrans3 = nn.ConvTranspose1d(32, num_channels, kernel_size=5, padding=2, stride=2, output_padding=1)
        
        # Output padding logic is tricky for arbitrary lengths. 
        # For 1800:
        # 1800 -> 900 -> 450 -> 225
        # 225 * 2 = 450 (needs output_padding=1 likely if odd/even mismatches occur, depending on input)
        # 225 -> 450 -> 900 -> 1800
        # If standard stride 2 halves exactly, fine. 
        # But 225 is odd. 225 // 2 depends on floor. 
        # Actually my stride=2 convs:
        # L_out = floor((L_in + 2*p - k)/s + 1)
        # 1800: (1800+4-5)/2 + 1 = floor(899.5) + 1 = 900.
        # 900:  (900+4-5)/2 + 1 = 450.
        # 450:  (450+2-3)/2 + 1 = 225.
        
        # Decoder (Transpose):
        # L_out = (L_in - 1)*s - 2*p + k + output_padding
        # In=225. s=2, p=1, k=3. 
        # (224)*2 - 2 + 3 + op = 448 + 1 + op = 449 + op. Need 450 => op=1.
        # In=450. s=2, p=2, k=5.
        # (449)*2 - 4 + 5 + op = 898 + 1 + op = 899 + op. Need 900 => op=1.
        # In=900. s=2, p=2, k=5.
        # (899)*2 - 4 + 5 + op = 1798 + 1 + op = 1799 + op. Need 1800 => op=1.
        
        # So output_padding=1 seems correct for all layers if starting with 1800.
    
    def forward(self, x):
        encoded = self.enc_relu1(self.enc_conv1(x))
        encoded = self.enc_relu2(self.enc_conv2(encoded))
        encoded = self.enc_relu3(self.enc_conv3(encoded))
        
        decoded = self.dec_relu1(self.dec_convtrans1(encoded))
        decoded = self.dec_relu2(self.dec_convtrans2(decoded))
        decoded = self.dec_convtrans3(decoded)
        
        return decoded
