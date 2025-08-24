import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import datetime
import tempfile
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Aura: AI Surveillance Dashboard", layout="wide")
st.title("Aura: Real-Time AI Anomaly Detection System")
st.markdown("Upload a video and click 'Start Processing' to begin.")

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.pt'
TARGET_WIDTH = 800
LOITER_TIME_THRESH = 5
LOITER_DIST_THRESH = 20
RUNNING_SPEED_THRESH = 20
ABANDONMENT_TIME_THRESH = 5
ABANDONMENT_DIST_THRESH = 100

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)
model = load_yolo_model(MODEL_PATH)

# --- SESSION STATE INITIALIZATION ---
def initialize_state():
    if 'processing' not in st.session_state: st.session_state.processing = False
    if 'video_buffer' not in st.session_state: st.session_state.video_buffer = None
    if 'alerts_log' not in st.session_state: st.session_state.alerts_log = []
    if 'alert_screenshots' not in st.session_state: st.session_state.alert_screenshots = {}
    if 'track_data' not in st.session_state:
        st.session_state.track_data = defaultdict(lambda: {'positions': [], 'frames': [], 'class_id': None, 'anomaly': None, 'last_seen': 0})
    if 'object_person_links' not in st.session_state: st.session_state.object_person_links = {}

initialize_state()

# --- PDF REPORT GENERATION FUNCTION ---
def generate_pdf_report():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Aura: Anomaly Detection Report", styles['h1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Alerts Summary", styles['h2']))
    if not st.session_state.alerts_log:
        elements.append(Paragraph("No anomalies detected.", styles['Normal']))
    else:
        for alert in reversed(st.session_state.alerts_log):
            elements.append(Paragraph(alert, styles['Normal']))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Alert Screenshots", styles['h2']))
    if not st.session_state.alert_screenshots:
        elements.append(Paragraph("No screenshots captured.", styles['Normal']))
    else:
        for caption, img_array in reversed(list(st.session_state.alert_screenshots.items())):
            img_pil = Image.fromarray(img_array)
            img_buffer = BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            elements.append(Paragraph(caption, styles['h4']))
            # --- THIS IS THE FIX ---
            # The invalid 'preserveAspectRatio' argument is removed.
            elements.append(ReportLabImage(img_buffer, width=400))
            elements.append(Spacer(1, 12))
    doc.build(elements)
    return buffer.getvalue()

# --- UI SIDEBAR ---
with st.sidebar:
    st.title("Controls")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"], key="file_uploader")
    if uploaded_file is not None:
        if st.session_state.get('uploaded_filename') != uploaded_file.name:
            st.session_state.video_buffer = uploaded_file.getvalue()
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.processing = False
            st.session_state.alerts_log, st.session_state.alert_screenshots = [], {}
    if st.session_state.video_buffer is not None:
        if not st.session_state.processing:
            if st.button("Start Processing"):
                st.session_state.processing = True
                st.rerun()
        else:
            if st.button("Stop Processing"):
                st.session_state.processing = False
                st.rerun()
    st.title("Anomaly Alerts")
    alerts_placeholder = st.empty()
    st.title("Reporting")
    if st.button("Generate Alert Report"):
        if not st.session_state.alerts_log:
            st.warning("No alerts to report.")
        else:
            pdf_data = generate_pdf_report()
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_data,
                file_name=f"anomaly_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

# --- MAIN DISPLAY AREA ---
video_placeholder = st.empty()
screenshot_placeholder = st.empty()

# --- CORE PROCESSING LOGIC ---
if st.session_state.processing and st.session_state.video_buffer is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(st.session_state.video_buffer)
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    LOITER_FRAME_THRESH = int(LOITER_TIME_THRESH * fps)
    ABANDONMENT_FRAME_THRESH = int(ABANDONMENT_TIME_THRESH * fps)
    frame_num = 0
    while cap.isOpened() and st.session_state.processing:
        ret, frame = cap.read()
        if not ret:
            st.write("Video processing finished.")
            st.session_state.processing = False
            break
        frame_num += 1
        h, w, _ = frame.shape
        scale = TARGET_WIDTH / w
        TARGET_HEIGHT = int(h * scale)
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        h, w, _ = frame.shape
        results = model.track(frame, persist=True, classes=[0, 24, 25, 26, 28], verbose=False)
        annotated_frame = frame.copy()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_id_to_box = {tid: b for tid, b in zip(track_ids, boxes)}
            current_persons, current_objects = {}, {}
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x, y, _, _ = box
                data = st.session_state.track_data[track_id]
                data['positions'].append((x, y))
                data['frames'].append(frame_num)
                data['class_id'] = cls_id
                data['last_seen'] = frame_num
                if len(data['positions']) > LOITER_FRAME_THRESH * 2:
                    data['positions'].pop(0)
                    data['frames'].pop(0)
                if cls_id == 0: current_persons[track_id] = (x, y)
                else: current_objects[track_id] = (x, y)
            for track_id, data in st.session_state.track_data.items():
                if frame_num - data['last_seen'] > fps: continue
                new_anomaly = None
                if data['class_id'] == 0 and len(data['positions']) > 2:
                    pos_now, pos_prev = np.array(data['positions'][-1]), np.array(data['positions'][-2])
                    if np.linalg.norm(pos_now - pos_prev) > RUNNING_SPEED_THRESH: new_anomaly = "Running"
                    elif len(data['positions']) > LOITER_FRAME_THRESH:
                        start_pos = np.array(data['positions'][-LOITER_FRAME_THRESH])
                        if np.linalg.norm(pos_now - start_pos) < LOITER_DIST_THRESH: new_anomaly = "Loitering"
                elif data['class_id'] != 0:
                    if track_id not in st.session_state.object_person_links:
                        min_dist, best_person = float('inf'), None
                        for p_id, p_pos in current_persons.items():
                            dist = np.linalg.norm(np.array(data['positions'][-1]) - p_pos)
                            if dist < 150: min_dist, best_person = dist, p_id
                        if best_person: st.session_state.object_person_links[track_id] = {'p_id': best_person, 'abandon_timer': 0}
                    if track_id in st.session_state.object_person_links:
                        link = st.session_state.object_person_links[track_id]
                        p_id = link['p_id']
                        if p_id in current_persons:
                            dist = np.linalg.norm(np.array(data['positions'][-1]) - current_persons[p_id])
                            link['abandon_timer'] = link['abandon_timer'] + 1 if dist > ABANDONMENT_DIST_THRESH else 0
                        else: link['abandon_timer'] += 1
                        if link['abandon_timer'] > ABANDONMENT_FRAME_THRESH: new_anomaly = "Abandoned Object"
                if new_anomaly and data['anomaly'] != new_anomaly:
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    alert_message = f"{timestamp}: {new_anomaly} (ID: {track_id})"
                    st.session_state.alerts_log.insert(0, alert_message)
                    if track_id in track_id_to_box:
                        x_c, y_c, w_b, h_b = track_id_to_box[track_id]
                        size = int(max(w_b, h_b) + 20)
                        x1, y1, x2, y2 = max(0, int(x_c - size / 2)), max(0, int(y_c - size / 2)), min(w, int(x_c + size / 2)), min(h, int(y_c + size / 2))
                        screenshot = frame[y1:y2, x1:x2]
                        if screenshot.size > 0:
                            resized = cv2.resize(screenshot, (150, 150))
                            st.session_state.alert_screenshots[alert_message] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                data['anomaly'] = new_anomaly
            for box, track_id, _ in zip(boxes, track_ids, class_ids):
                is_anomaly = st.session_state.track_data.get(track_id, {}).get('anomaly')
                x, y, box_w, box_h = box
                x1, y1, x2, y2 = int(x - box_w/2), int(y - box_h/2), int(x + box_w/2), int(y + box_h/2)
                color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                label = f"ID:{track_id}" + (f" {is_anomaly}" if is_anomaly else "")
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        alerts_placeholder.markdown("\n".join(st.session_state.alerts_log))
        with screenshot_placeholder.container():
            if st.session_state.alert_screenshots:
                st.subheader("Alert Screenshots")
                cols = st.columns(5)
                screenshots = list(st.session_state.alert_screenshots.items())
                screenshots.reverse()
                for i, (caption, img) in enumerate(screenshots):
                    with cols[i % 5]: st.image(img, caption=caption, use_container_width=True)
    cap.release()
else:
    st.info("Please upload a video file and click 'Start Processing'.")
