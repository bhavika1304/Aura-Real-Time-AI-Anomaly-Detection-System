# Aura: AI-Powered Surveillance System

An automated surveillance system designed to detect behavioral anomalies in real-time from video feeds. This project was developed for the Honeywell Hackathon.

**Live Demo:** [PASTE YOUR STREAMLIT CLOUD URL HERE]

![Dashboard Screenshot](files/demo_2_compressed.mp4)

---

## ## Features

The system processes video to identify and track individuals and objects, flagging several types of anomalies with visual alerts.

### ### Detected Anomalies
* **Loitering:** An individual remaining stationary in a single area for an extended period.
* **Running:** An individual moving at a speed significantly higher than the surrounding people.
* **Abandoned Object:** An object (e.g., a bag) that is left behind by a person who then moves away from the area.

### ### Dashboard Functionality
* **Video Upload:** Users can upload their own `.mp4` or `.avi` video files.
* **Real-Time Visualization:** A live video feed displays color-coded bounding boxes (green for normal, red for anomaly) and tracking IDs.
* **Alerts Log:** A sidebar provides a timestamped log of all detected anomalies for review.
* **Alert Screenshots:** The dashboard automatically captures and displays screenshots of each detected anomaly.
* **PDF Reporting:** Users can generate and download a complete PDF report of all alerts and screenshots from the session.

---

## ## How to Run

1.  **Deploy:** The application is deployed on Streamlit Community Cloud. Access it via the "Live Demo" link above.
2.  **Upload:** Use the sidebar to upload a video file.
3.  **Process:** Click the "Start Processing" button to begin the analysis.
4.  **Monitor:** Observe the video feed for alerts and review the logs and screenshots.

---

## ## Technology Stack

* **Language:** Python
* **Core Libraries:** OpenCV, NumPy
* **AI/ML:** Ultralytics (for YOLOv8), Streamlit
* **Reporting:** ReportLab
