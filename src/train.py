import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.dataset import ECGDataset
from src.model import ECGCNN, ECGAutoencoder
import os
import copy
import argparse

def train_model(model_type='cnn', epochs=10, batch_size=32, lr=0.001, data_dir='data/raw', save_dir='models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Dataset
    # Ideally split by patient ID here. For simplicity in this script, we'll randomize split.
    # In production, use explicit patient IDs for train/val split.
    full_dataset = ECGDataset(data_dir=data_dir)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    if model_type == 'cnn':
        model = ECGCNN().to(device)
        criterion = nn.CrossEntropyLoss()
    elif model_type == 'autoencoder':
        model = ECGAutoencoder().to(device)
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid model type")
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'cnn':
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            else: # autoencoder
                # Autoencoder reconstructs input
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        epoch_acc = correct / total if model_type == 'cnn' else 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if model_type == 'cnn':
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / val_size
        val_acc = val_correct / val_total if model_type == 'cnn' else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        if model_type == 'cnn':
            print(f"Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, f"best_{model_type}.pth"))
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='cnn or autoencoder')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    train_model(model_type=args.model, epochs=args.epochs)
