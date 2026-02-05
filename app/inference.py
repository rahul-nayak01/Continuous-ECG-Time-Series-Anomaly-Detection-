import numpy as np
from pathlib import Path
import torch
from src.model import ECGCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Absolute path relative to THIS file
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_cnn.pth"

model = ECGCNN().to(DEVICE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_ecg(window: np.ndarray):
    """
    window shape: (T, C)
    """
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, C)
    x = x.permute(0, 2, 1).to(DEVICE)                            # (1, C, T)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0, 1].item()

    label = "Abnormal" if prob > 0.5 else "Normal"
    return label, prob
