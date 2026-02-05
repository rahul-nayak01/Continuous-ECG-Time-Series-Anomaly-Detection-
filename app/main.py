from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np

from app.inference import predict_ecg

app = FastAPI(title="ECG Anomaly Detection")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    data = np.load(file.file)

    # -------- Case 1: Single window (T, C) --------
    if data.ndim == 2:
        label, prob = predict_ecg(data)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": label,
                "probability": round(prob, 3)
            }
        )

    # -------- Case 2: Multiple windows (N, T, C) --------
    elif data.ndim == 3:
        results = [predict_ecg(w) for w in data]

        abnormal_probs = [p for _, p in results]
        abnormal_ratio = sum(p > 0.5 for p in abnormal_probs) / len(abnormal_probs)

        final_label = "Abnormal" if abnormal_ratio > 0.3 else "Normal"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": final_label,
                "probability": round(abnormal_ratio, 3)
            }
        )

    # -------- Invalid input --------
    else:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Invalid ECG file",
                "probability": None
            }
        )
