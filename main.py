import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = 'data/Avenue Dataset/testing_videos/01.avi'  # IMPORTANT: Set this to your test video

TARGET_WIDTH = 800

# --- ANOMALY THRESHOLDS ---
LOITER_TIME_THRESH = 5  # seconds
LOITER_DIST_THRESH = 20  # pixels
RUNNING_SPEED_THRESH = 20  # pixels per frame (for instantaneous check)
ABANDONMENT_TIME_THRESH = 5  # seconds
ABANDONMENT_DIST_THRESH = 100  # pixels

# --- INITIALIZATION ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video at {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30

LOITER_FRAME_THRESH = int(LOITER_TIME_THRESH * fps)
ABANDONMENT_FRAME_THRESH = int(ABANDONMENT_TIME_THRESH * fps)

# --- STATE TRACKING ---
track_data = defaultdict(lambda: {'positions': [], 'frames': [], 'class_id': None, 'anomaly': None})
object_person_links = {}

frame_num = 0

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_num += 1

    # Resize frame for performance
    h, w, _ = frame.shape
    scale = TARGET_WIDTH / w
    TARGET_HEIGHT = int(h * scale)
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # Run tracker
    results = model.track(frame, persist=True, classes=[0, 24, 25, 26, 28], verbose=False)
    annotated_frame = frame.copy()  # Use a copy for drawing

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        current_persons = {}
        current_objects = {}

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x, y, _, _ = box
            track_data[track_id]['positions'].append((x, y))
            track_data[track_id]['frames'].append(frame_num)
            track_data[track_id]['class_id'] = cls_id

            if len(track_data[track_id]['positions']) > LOITER_FRAME_THRESH * 2:
                track_data[track_id]['positions'].pop(0)
                track_data[track_id]['frames'].pop(0)

            if cls_id == 0:
                current_persons[track_id] = (x, y)
            else:
                current_objects[track_id] = (x, y)

        for track_id, data in track_data.items():
            data['anomaly'] = None

            # --- Module 1: Loitering & Running (for Persons) ---
            if data['class_id'] == 0 and len(data['positions']) > 2:

                # --- THIS IS THE CORRECTED RUNNING CHECK ---
                pos_now = np.array(data['positions'][-1])
                pos_prev = np.array(data['positions'][-2])
                instant_speed = np.linalg.norm(pos_now - pos_prev)

                if instant_speed > RUNNING_SPEED_THRESH:
                    data['anomaly'] = "Running"

                # --- LOITERING CHECK ---
                elif len(data['positions']) > LOITER_FRAME_THRESH:
                    start_pos = np.array(data['positions'][-LOITER_FRAME_THRESH])
                    end_pos = np.array(data['positions'][-1])
                    if np.linalg.norm(end_pos - start_pos) < LOITER_DIST_THRESH:
                        data['anomaly'] = "Loitering"

            # --- Module 2: Object Abandonment ---
            elif data['class_id'] != 0:
                if track_id not in object_person_links:
                    min_dist, best_person = float('inf'), None
                    for p_id, p_pos in current_persons.items():
                        dist = np.linalg.norm(np.array(data['positions'][-1]) - np.array(p_pos))
                        if dist < 150:
                            min_dist, best_person = dist, p_id
                    if best_person:
                        object_person_links[track_id] = {'p_id': best_person, 'abandon_timer': 0}

                if track_id in object_person_links:
                    link = object_person_links[track_id]
                    p_id = link['p_id']
                    if p_id in current_persons:
                        dist = np.linalg.norm(np.array(data['positions'][-1]) - np.array(current_persons[p_id]))
                        if dist > ABANDONMENT_DIST_THRESH:
                            link['abandon_timer'] += 1
                        else:
                            link['abandon_timer'] = 0
                    else:
                        link['abandon_timer'] += 1
                    if link['abandon_timer'] > ABANDONMENT_FRAME_THRESH:
                        data['anomaly'] = "Abandoned Object"

        # --- DRAWING LOGIC (must be inside the 'if' block) ---
        for box, track_id, _ in zip(boxes, track_ids, class_ids):
            is_anomaly = track_data.get(track_id, {}).get('anomaly')
            x, y, box_w, box_h = box
            x1, y1, x2, y2 = int(x - box_w / 2), int(y - box_h / 2), int(x + box_w / 2), int(y + box_h / 2)
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            label = f"ID:{track_id}"
            if is_anomaly:
                label = f"{label} {is_anomaly}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("AI Surveillance", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()