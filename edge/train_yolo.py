# /dist_home/suryansh/vignesh/s5/bd/train_yolo.py
import os, json
from ultralytics import YOLO

YAML = "/dist_home/suryansh/vignesh/s5/bd/emotion_data.yaml"
PROJECT = "/dist_home/suryansh/vignesh/s5/bd/yolo_runs"
RUN1 = "emotion_yolo"
RUN2 = "emotion_yolo_finetune"
RESULTS_DIR = "/dist_home/suryansh/vignesh/s5/bd/yolo_results"  # final consolidated folder

os.makedirs(PROJECT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_summary(results, out_json):
    # Ultralytics results has .metrics dict-like fields; convert to JSON-serializable
    try:
        data = {
            "metrics": getattr(results, "metrics", None),
            "speed": getattr(results, "speed", None),
            "maps": getattr(results, "maps", None)
        }
    except Exception:
        data = {"string": str(results)}
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

# Phase 1: GPU1 training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = YOLO("yolov8m.pt")
model.train(
    data=YAML,
    epochs=30,
    imgsz=224,
    batch=32,
    device=0,
    project=PROJECT,
    name=RUN1,
    augment=True,
    deterministic=True,
    workers=8
)

# Phase 2: GPU0 fine-tune
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
best_pt = f"{PROJECT}/{RUN1}/weights/best.pt"
model2 = YOLO(best_pt)
model2.train(
    data=YAML,
    epochs=10,
    imgsz=224,
    batch=16,
    device=0,
    project=PROJECT,
    name=RUN2,
    deterministic=True,
    workers=8
)

# Validation with plots
val_res = model2.val(
    data=YAML,
    project=PROJECT,
    name=f"{RUN2}_val",
    save_json=True,   # COCO-style JSON where applicable
    plots=True
)

# Copy key artifacts to RESULTS_DIR
import shutil, glob
final_best = f"{PROJECT}/{RUN2}/weights/best.pt"
final_plots = glob.glob(f"{PROJECT}/{RUN2}_val/*.*") + \
              glob.glob(f"{PROJECT}/{RUN2}_val/plots/*.*") + \
              glob.glob(f"{PROJECT}/{RUN2}/results*.csv")

# Save summary JSON
save_summary(val_res, f"{RESULTS_DIR}/validation_summary.json")

# Copy weights
shutil.copy2(final_best, f"{RESULTS_DIR}/best.pt")

# Copy plots, confusion matrix, PR/ROC, results.csv, etc.
for p in final_plots:
    try:
        shutil.copy2(p, RESULTS_DIR)
    except Exception:
        pass

print(f"ALL DONE. Results packed in: {RESULTS_DIR}")
