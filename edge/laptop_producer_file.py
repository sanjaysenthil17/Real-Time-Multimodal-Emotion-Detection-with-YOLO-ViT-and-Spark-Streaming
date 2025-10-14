import cv2
import numpy as np
import json
import time
import os
import socket
from datetime import datetime, timezone
from collections import deque
import pyttsx3

from ultralytics import YOLO

# Text to speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Settings
SOURCE_ID = socket.gethostname()
WEIGHTS = os.path.abspath("best.pt")  # Path to your trained YOLO weights
IMGSZ = 224
BATCH_SIZE = 100
LOOP_DELAY = 0.03
CAM_INDEX = 0
OUTPUT_DIR = os.path.abspath("streaming_output/yolo_batches")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
yolo = YOLO(WEIGHTS)

batch = []
batch_idx = 0
next_person_id = 1
max_dist_for_tracking = 120
prev_centers = {}
person_emotion_histories = {}
last_spoken_emotion = {}

def save_batch(records):
    global batch_idx
    ts = int(time.time())
    fname = f"yolo_batch_{batch_idx}_{ts}.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records → {fpath}")
    batch_idx += 1

def euclidean_distance(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def assign_person_ids(prev_centers, curr_boxes, threshold=max_dist_for_tracking):
    global next_person_id
    assigned_ids = []
    used_prev = set()

    curr_centers = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2) in curr_boxes]

    for cc in curr_centers:
        dists = [(pid, euclidean_distance(cc, pc)) for pid, pc in prev_centers.items()]
        dists = [d for d in dists if d[1] < threshold and d[0] not in used_prev]
        if dists:
            pid, _ = min(dists, key=lambda x: x[1])
            assigned_ids.append(pid)
            used_prev.add(pid)
        else:
            assigned_ids.append(next_person_id)
            prev_centers[next_person_id] = cc
            used_prev.add(next_person_id)
            next_person_id += 1
    pids_to_remove = set(prev_centers.keys()) - used_prev
    for pid in pids_to_remove:
        del prev_centers[pid]

    return assigned_ids

def gradcam_overlay(frame, box):
    # Simulate Grad-CAM by overlaying a colored rectangle with alpha
    (x1,y1,x2,y2) = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

def text_to_speech(pid, emotion):
    history = person_emotion_histories.setdefault(pid, deque(maxlen=5))
    history.append(emotion)
    current_emotion = max(set(history), key=history.count)
    if last_spoken_emotion.get(pid) != current_emotion:
        engine.say(f"Person {pid}: {current_emotion}")
        engine.runAndWait()
        last_spoken_emotion[pid] = current_emotion

def main():
    global prev_centers
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            H,W = frame.shape[:2]
            results = yolo(frame, imgsz=IMGSZ, verbose=False)
            if len(results) == 0:
                time.sleep(LOOP_DELAY)
                frame_id += 1
                continue
            res = results[0]

            if res.boxes is None or len(res.boxes) == 0:
                time.sleep(LOOP_DELAY)
                frame_id += 1
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            person_ids = assign_person_ids(prev_centers, boxes)

            for i, (box, clsid, conf, pid) in enumerate(zip(boxes, classes, confs, person_ids)):
                x1, y1, x2, y2 = map(int, box)
                emotion_label = yolo.names[int(clsid)]

                # Text to speech announce
                text_to_speech(pid, emotion_label)

                # Grad-CAM simulation
                frame = gradcam_overlay(frame, (x1, y1, x2, y2))
                # Draw box and label
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{pid} {emotion_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Save JSON record
                rec = {
                    "ts_ms": int(time.time()*1000),
                    "iso_ts": datetime.now(timezone.utc).isoformat(),
                    "source_id": SOURCE_ID,
                    "frame_id": frame_id,
                    "bbox": [x1, y1, x2, y2],
                    "cls_name": emotion_label,
                    "cls_id": int(clsid),
                    "conf": float(conf),
                    "img_w": W,
                    "img_h": H,
                    "person_id": pid
                }
                batch.append(rec)

            cv2.imshow("YOLO Emotion Detection + Grad-CAM + Audio", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # you can kill with q
                break

            if len(batch) >= BATCH_SIZE:
                save_batch(batch)
                batch.clear()

            frame_id += 1
            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        if batch:
            save_batch(batch)
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()

if __name__ == "__main__":
    main()
