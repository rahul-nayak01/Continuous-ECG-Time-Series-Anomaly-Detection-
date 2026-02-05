# 🫀 Continuous ECG Time-Series Anomaly Detection

**End-to-End Deep Learning, MLOps & Cloud Deployment System**

---

## 📌 Project Overview
This project implements a production-ready ECG anomaly detection system that classifies continuous ECG time-series data into **Normal** or **Abnormal** cardiac activity using deep learning.

It demonstrates end-to-end ownership of a real-world machine learning pipeline, covering signal preprocessing, window-based modeling, deep learning, model persistence, FastAPI inference, web UI, Dockerization, and CI/CD-driven cloud deployment on AWS.

## 🎯 Problem Statement
Continuous ECG monitoring produces high-frequency physiological time-series data. Detecting abnormal cardiac patterns (e.g., arrhythmias) in real time is challenging due to:
* **Noise and artifacts** in raw ECG signals.
* **Long continuous recordings** requiring efficient segmentation.
* **Severe class imbalance** (abnormal events are rare).
* **Low-latency inference** requirements for clinical monitoring.

## 🧠 System Architecture



1.  **Raw ECG** → Signal Preprocessing (Filtering + Normalization)
2.  **Sliding Window Segmentation** → Window-Level Labeling
3.  **Deep Learning Models** (CNN for Classification & Autoencoder for Reconstruction)
4.  **Inference Service** (FastAPI) → Web UI / REST API
5.  **Deployment** (Docker + AWS CI/CD)

---

## 📂 Project Structure

```text
Continuous-ECG-Time-Series-Anomaly-Detection/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── inference.py           # Model loading & inference
│   ├── templates/
│   │   └── index.html         # Frontend UI
│   └── static/
│       └── style.css          # CSS styling
│
├── src/
│   └── model.py               # ECGCNN & ECGAutoencoder definitions
│
├── data/
│   ├── raw/                   # Original ECG (read-only)
│   ├── processed/             # Filtered & normalized ECG
│   └── windows/               # Windowed ECG + labels
│
├── models/
│   ├── best_cnn.pth           # Trained CNN weights
│   └── best_autoencoder.pth   # Trained Autoencoder weights
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_windowing.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── .github/
│   └── workflows/
│       └── cicd.yaml          # CI/CD pipeline (GitHub Actions)
│
├── Dockerfile                 # Docker image definition
├── .dockerignore              # Docker ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation