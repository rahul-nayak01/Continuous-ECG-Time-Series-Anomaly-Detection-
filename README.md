# Continuous-ECG-Time-Series-Anomaly-Detection-
End to End pipeline on ECD Data

🫀 ECG Continuous Time-Series Anomaly Detection
End-to-End Deep Learning, MLOps & Cloud Deployment System
📌 Project Overview

This project implements a production-ready ECG anomaly detection system that classifies continuous ECG time-series data into Normal or Abnormal cardiac activity using deep learning.

It demonstrates end-to-end ownership of a real-world machine learning pipeline, covering signal preprocessing, window-based modeling, deep learning, model persistence, FastAPI inference, web UI, Dockerization, and CI/CD-driven cloud deployment on AWS.

🎯 Problem Statement

Continuous ECG monitoring produces high-frequency physiological time-series data. Detecting abnormal cardiac patterns (e.g., arrhythmias) in real time is challenging due to:

Noise and artifacts in ECG signals

Long continuous recordings

Severe class imbalance (abnormal events are rare)

Low-latency inference requirements

Reproducible and scalable deployment needs

This project addresses these challenges using a window-based deep learning approach combined with industry-grade engineering and MLOps practices.

🧠 System Architecture

Raw ECG
→ Signal Preprocessing (Filtering + Normalization)
→ Sliding Window Segmentation
→ Window-Level Labeling
→ Deep Learning Models
 • CNN (Supervised Classification)
 • Autoencoder (Anomaly Detection)
→ FastAPI Inference Service
→ Web UI / REST API
→ Docker + AWS CI/CD Deployment

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
│   └── model.py               # ECGCNN & ECGAutoencoder
│
├── data/
│   ├── raw/                   # Original ECG (read-only)
│   ├── processed/             # Filtered & normalized ECG
│   └── windows/               # Windowed ECG + labels
│
├── models/
│   ├── best_cnn.pth
│   └── best_autoencoder.pth
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_windowing.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── .github/
│   └── workflows/
│       └── cicd.yaml          # CI/CD pipeline
│
├── Dockerfile                 # Docker image definition
├── .dockerignore              # Docker ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


🔬 Data Pipeline
Raw Data (data/raw)

Original ECG recordings

Never modified

Not committed to Git

Processed Data (data/processed)

Bandpass filtering

Z-score normalization

Noise-reduced ECG signals

Windowed Data (data/windows)

Fixed-length sliding windows

Shape: (num_windows, time_steps, channels)

Labels:

0 → Normal

1 → Abnormal

Strict separation between stages ensures no data leakage and full reproducibility.

🤖 Models
CNN (Primary Model)

1D Convolutional Neural Network

Learns temporal ECG patterns directly from windowed signals

Supervised binary classification: Normal vs Abnormal

Autoencoder (Secondary / Experimental)

Trained primarily on normal ECG data

Uses reconstruction error as an anomaly score

Useful when labeled abnormal data is limited

🧪 Training & Evaluation
Loss Functions

CNN: CrossEntropyLoss

Autoencoder: Mean Squared Error (MSE)

Evaluation Metrics

Precision

Recall (Sensitivity)

F1-Score

ROC-AUC

Confusion Matrix

🚀 Inference System
FastAPI Backend

Loads trained models from the models/ directory

Accepts ECG window files in .npy format

Supports:

Single-window inference (T, C)

Multi-window inference (N, T, C) with aggregation

Web Interface

Lightweight HTML + CSS frontend

Upload ECG window file

Displays prediction (Normal / Abnormal) with confidence score

🐳 Dockerization

The entire application is containerized using Docker to ensure:

Environment consistency

Platform independence

Reproducible deployment

🔁 CI/CD Pipeline

A GitHub Actions-based CI/CD pipeline automates:

Docker image build

Authentication with AWS ECR

Pushing image to ECR

Pulling image on EC2 (self-hosted runner)

Stopping old container

Running latest version

This enables fully automated production deployment on every push to the main branch.

☁️ Cloud Deployment

AWS ECR — Docker image registry

AWS EC2 — FastAPI hosting

Self-Hosted GitHub Runner — Secure deployment