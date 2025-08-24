# Aura 🔍
AI-Powered Surveillance System for Real-Time Behavioral Anomaly Detection

## 🌟 Overview
**Aura** is an end-to-end video intelligence system that detects behavioral anomalies (Loitering, Running, Abandoned Object) from CCTV footage in real time. Built for the Honeywell Hackathon, Aura combines YOLOv8 detection, multi-object tracking, rule-based anomaly logic, and a clean Streamlit dashboard with alert logs, screenshots, and one-click PDF reporting. Experimental modules include Autoencoders (unsupervised anomaly detection) and GAN-based rare-event synthesis.

**Live Demo:** _Add your Streamlit Cloud URL_  
**Demo Video:** _Add a short 20–30s clip link_  

## 🚀 Features
- **Real-Time Detection & Tracking**: YOLOv8 + persistent IDs across frames  
- **Anomalies Detected**:
  - **Loitering**: stationarity beyond time & radius thresholds
  - **Running**: sudden high displacement between frames
  - **Abandoned Object**: owner–object link broken for ≥ threshold
- **Interactive Dashboard (Streamlit)**: live overlays (green = normal, red = anomaly), timestamped alert panel, screenshots grid
- **PDF Reporting (ReportLab)**: one-click export with alerts + evidence
- **Advanced (Optional)**: Autoencoder (unsupervised) and GAN synthesis for rare scenarios

## 🏗️ Tech Stack
| Component | Technology |
|---|---|
| Detection | Ultralytics YOLOv8 (nano by default) |
| CV / Utils | OpenCV, NumPy, Pillow |
| UI | Streamlit |
| Reporting | ReportLab (PDF) |
| Optional AI | Autoencoders (PyTorch), GANs (for rare event synthesis) |

## 📐 System Architecture
`files/sys arch.png`
