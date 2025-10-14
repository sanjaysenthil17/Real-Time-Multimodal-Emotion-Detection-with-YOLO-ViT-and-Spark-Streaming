# Full corrected ViT Emotion Detection Script with Grad-CAM, IDs, Audio and JSON batch writing
import cv2
import numpy as np
import torch
import time
import json
import os
from PIL import Image
from collections import deque, Counter
from transformers import ViTImageProcessor, ViTForImageClassification
import pyttsx3
import threading
import queue
from datetime import datetime, timezone
import socket

SOURCE_ID = socket.gethostname()


# =============== Speech queue class ===============
class SpeechQueue:
    def __init__(self, driver='sapi5', rate=175, volume=1.0):
        self.q = queue.Queue()
        self.engine = pyttsx3.init(driver)
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
    def _loop(self):
        while True:
            text = self.q.get()
            if text is None: break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass
    def say(self, text):
        self.q.put(text)
    def stop(self):
        self.q.put(None)

# =============== Emotion announcer with stability and persistence ===============
class EmotionAnnouncer:
    def __init__(self, speak, window_frames=5, change_cooldown=1.0, persist_secs=2.0):
        self.speak = speak
        self.window = window_frames
        self.change_cd = change_cooldown
        self.persist_secs = persist_secs
        self.buffers = {}
        self.stable_label = {}
        self.last_change_t = {}
        self.hold_start_t = {}
        self.last_persist_t = {}

    def update(self, pid, label):
        buf = self.buffers.setdefault(pid, deque(maxlen=self.window))
        buf.append(label)
        if len(buf) < self.window:
            return
        stable = max(set(buf), key=list(buf).count)
        now = time.time()

        prev = self.stable_label.get(pid)
        if prev != stable:
            if now - self.last_change_t.get(pid, 0) >= self.change_cd:
                self.speak(f"Person {pid} is {stable}")
                self.last_change_t[pid] = now
            self.stable_label[pid] = stable
            self.hold_start_t[pid] = now
            self.last_persist_t[pid] = 0
            return

        hold_start = self.hold_start_t.get(pid, now)
        held_secs = now - hold_start
        if held_secs >= self.persist_secs:
            last_persist = self.last_persist_t.get(pid, 0)
            if now - last_persist >= self.persist_secs:
                self.speak(f"Person {pid} still {stable}")
                self.last_persist_t[pid] = now

# =============== Settings ===================================
MODEL_ID = "trpakov/vit-face-expression"
USE_AUDIO = True
ID_MAX_DIST = 100
WINDOW_FRAMES = 5
PERSIST_SECS = 2.0
CHANGE_COOLDOWN = 1.0

# =============== Setup ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
try:
    model.set_attn_implementation("eager")
except Exception:
    pass

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emoji_map = {
    "happy":"😊","sad":"😢","angry":"😡","neutral":"😐",
    "disgust":"🤢","surprise":"😮","fear":"😱","contempt":"🙄"
}
feedback_map = {
    "sad":"Take a deep breath...",
    "angry":"Try to relax.",
    "fear":"You're safe.",
    "disgust":"Reset your mind."
}
class_list = list(emoji_map.keys())

# =============== Persistent ID assignment ===================
def assign_ids(prev, curr, next_id_start=1, max_dist=100):
    assigned = {}
    used_prev = set()
    for b in curr:
        x,y,w,h = b
        cx, cy = x+w//2, y+h//2
        best, bestd = None, 1e18
        for pid, pb in prev.items():
            if pid in used_prev:
                continue
            px,py,pw,ph = pb
            pcx,pcy = px+pw//2, py+ph//2
            d = (pcx-cx)**2 + (pcy-cy)**2
            if d < bestd:
                bestd, best = d, pid
        if best is not None and bestd**0.5 < max_dist:
            assigned[best] = b
            used_prev.add(best)
        else:
            nid = next_id_start
            while nid in prev or nid in assigned:
                nid += 1
            assigned[nid] = b
            next_id_start = nid + 1
    return assigned

# =============== ViT Grad-CAM via attention rollout with pixel-grad fallback ===================
def vit_gradcam_overlay(pil_img, target_label_idx=None, size=None, alpha=0.45):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    inputs["pixel_values"].requires_grad_(True)
    outputs = model(**inputs, output_attentions=True)
    logits = outputs.logits
    probs = logits.softmax(-1)
    cls_idx = int(probs.argmax(-1).item()) if target_label_idx is None else target_label_idx
    score = logits[0, cls_idx]

    cam = None
    attns = getattr(outputs, "attentions", None)
    if attns and len(attns) > 0:
        A = None
        for A_l in attns:
            a = A_l[0].mean(0).detach().cpu().numpy()
            a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)
            A = a if A is None else A @ a
        cls_attn = A[0]
        num_patches = cls_attn.shape[0] - 1
        side = int(num_patches**0.5) if num_patches > 0 else 1
        if side * side == num_patches and num_patches > 0:
            patch_map = cls_attn[1:].reshape(side, side)
            cam = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)

    if cam is None:
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        grads = inputs["pixel_values"].grad.detach().cpu().numpy()[0]
        cam = np.mean(np.abs(grads), axis=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    base = np.array(pil_img)
    if size:
        base = cv2.resize(base, size, interpolation=cv2.INTER_LINEAR)

    H,W = base.shape[:2]
    cam_rs = cv2.resize(cam.astype(np.float32), (W,H), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((cam_rs*255).astype(np.uint8), cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 1-alpha, heat, alpha, 0.0)
    return overlay, cls_idx, float(probs[0, cls_idx].item())

# =============== Audio speaker setup ==========================
speaker = SpeechQueue('sapi5', rate=175, volume=1.0) if USE_AUDIO else None
announcer = EmotionAnnouncer(
    speak=(speaker.say if speaker else (lambda s: None)),
    window_frames=WINDOW_FRAMES,
    change_cooldown=CHANGE_COOLDOWN,
    persist_secs=PERSIST_SECS
)

# =============== JSON batch saving setup =======================
batch = []
batch_idx = 0
BATCH_SIZE = 100
OUTPUT_DIR = r"streaming_output/vit_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_batch(records):
    global batch_idx
    ts = int(time.time())
    fname = f"vit_batch_{batch_idx}_{ts}.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records → {fpath}")
    batch_idx += 1

# =============== Main loop ========================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Webcam not detected")

print("Press Ctrl+C to quit.")
frame_id = 0
prev_ids = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
        faces = [tuple(map(int, b)) for b in faces]
        prev_ids = assign_ids(prev_ids, faces, next_id_start=1, max_dist=ID_MAX_DIST)

        emotions = []
        for pid, (x,y,w,h) in prev_ids.items():
            x2, y2 = x+w, y+h
            x, y = max(0,x), max(0,y)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x or y2 <= y: continue

            face_bgr = frame[y:y2, x:x2]
            if face_bgr.size == 0: continue
            pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))

            overlay, cls_idx, score = vit_gradcam_overlay(pil, size=(x2-x, y2-y), alpha=0.45)
            label = model.config.id2label[cls_idx].lower()
            emotions.append(label)

            frame[y:y2, x:x2] = overlay
            emj = emoji_map.get(label, "")
            cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"P{pid} {emj} {label} ({score:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            announcer.update(pid, label)

            rec = {
                "ts_ms": int(time.time()*1000),
                "iso_ts": datetime.now(timezone.utc).isoformat(),
                "source_id": SOURCE_ID,
                "frame_id": frame_id,
                "bbox": [x, y, x2, y2],
                "cls_name": label,
                "cls_id": cls_idx,
                "conf": score,
                "img_w": W,
                "img_h": H,
                "person_id": pid
            }
            batch.append(rec)

        # Visualize group mood and feedback
        grp = Counter(emotions)
        group_mood = grp.most_common(1)[0][0] if grp else "neutral"
        cv2.putText(frame, f"Group Mood: {group_mood}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        fb = feedback_map.get(group_mood, "")
        if fb:
            cv2.putText(frame, fb, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Time-series visualization
        history = deque(maxlen=100)
        history.append(group_mood)
        for i, m in enumerate(history):
            idx = class_list.index(m) if m in class_list else class_list.index("neutral")
            cv2.circle(frame, (10 + i*4, H - 10 - idx*20), 2, (255, 255, 0), -1)
        cv2.putText(frame, "Emotion Time-Series", (10, H - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("ViT Emotion Grad-CAM + IDs + Audio", frame)

        cv2.waitKey(1)
        if len(batch) >= BATCH_SIZE:
            save_batch(batch)
            batch.clear()

        frame_id += 1

except KeyboardInterrupt:
    print("Interrupted by user, closing...")

    if batch:
        save_batch(batch)
        batch.clear()

    cap.release()
    if speaker:
        try:
            speaker.stop()
        except:
            pass
    cv2.destroyAllWindows()
