

# Real‑Time Multimodal Emotion Detection
YOLO + ViT + Spark Streaming (Medallion: Bronze → Silver → Gold)

A production‑style, big‑data pipeline that:
- Detects 9 facial expressions in real time on the edge (YOLO and ViT with Grad‑CAM).
- Preprocesses the Kaggle dataset with Scala into Parquet on HDFS for faster training/analytics.
- Streams batched JSON events into Spark, producing Bronze, Silver, and Gold layers.

Made with ❤️ by VigneshVS2005,
                 sanjaysenthil,
                 karthick ruban,
                 hariprakash

## ✨ What’s inside
- Edge producers (YOLO + ViT): bounding boxes, emotions, confidence, persistent person IDs, optional audio prompts.
- Scala preprocessing to convert raw dataset images to columnar Parquet.
- Spark Structured Streaming with event‑time windows and stateful aggregations.
- Medallion design: Bronze (raw), Silver (cleaned), Gold (windowed KPIs).

Repo layout
```
edge/
  laptop_producer_file.py       # YOLO realtime producer (IDs/audio/JSON)
  laptop_produce_file.py        # ViT realtime producer (Grad-CAM/audio/JSON)
  train_yolo.py                 # YOLO training
  emotion_data.yaml             # YOLO dataset config

spark/
  xyz.scala                     # Scala preprocessing → Parquet on HDFS
  spark_vit_yolo_streaming.py   # Streaming → Bronze/Silver/Gold
```

## 📦 Data: 9 Facial Expressions (Kaggle)
Classes: angry, disgust, fear, happy, neutral, sad, surprise, (and your set’s two more). Organize as train/valid/test with images/labels. The Scala job reads image bytes + labels, sets 224×224×3 meta, and writes Parquet shards to HDFS for parallel IO.

Commands (example)
```bash
hdfs dfs -mkdir -p /vignesh/s5/bd/dataset
hdfs dfs -put dataset/* /vignesh/s5/bd/dataset/
```

## 🔧 Preprocess → Parquet on HDFS (Scala)
Key idea: turn many small files into columnar Parquet for fast scans and model IO.

Compile + submit
```bash
scalac -target:jvm-11 -classpath $(echo $SPARK_HOME/jars/*.jar | tr ' ' ':') spark/xyz.scala
jar cf EmotionPreprocessing.jar *.class

spark-submit \
  --class EmotionPreprocessing \
  --master spark://asaicomputenode03.amritanet.edu:7077 \
  --deploy-mode cluster \
  --driver-memory 2g \
  --num-executors 12 \
  --executor-cores 5 \
  --executor-memory 46g \
  --conf spark.hadoop.fs.defaultFS=hdfs://172.17.16.11:9000 \
  EmotionPreprocessing.jar
```

Output
```
hdfs://asaicomputemaster:9000/vignesh/s5/bd/dataset_parquet
```

Why Parquet?
- Columnar compression (Snappy), predicate pushdown, vectorized reads → faster training and downstream analytics.

## 🧠 Train YOLO on the Parquet‑derived dataset
Configure classes/paths in edge/emotion_data.yaml, then:
```bash
python edge/train_yolo.py
```
- Produces best.pt (store with Git LFS or keep locally).
- You used this .pt for realtime detection.

## 🎥 Edge: Real‑Time Emotion Detection (YOLO + ViT)
Run either producer on your laptop webcam:

YOLO producer
```bash
python edge/yolo_opencv_emotion_detection.py
```
ViT producer (with Grad‑CAM overlays)
```bash
python edge/vit_opencv_emotion_detection.py
```

What happens:
- Webcam → model inference (emotions, conf, boxes).
- Stable person IDs (Euclidean matching) + optional audio prompts.
- JSON batches of size 100 written locally, each record fields:
  tsms, isots, sourceid, frameid, bbox, clsname, clsid, conf, imgw, imgh, personid, model.

Sync to cluster + land in HDFS
```bash
rsync -avz /path/to/local/json/ user@cluster:/path/incoming/
hdfs dfs -mkdir -p /vignesh/s5/bd/streaming_input
hdfs dfs -put /path/incoming/*.json /vignesh/s5/bd/streaming_input/
```

## ⚡ Streaming Analytics: Bronze → Silver → Gold
Start the Spark job:
```bash
spark-submit --master spark://asaicomputenode03.amritanet.edu:7077 \
  spark/spark_vit_yolo_streaming.py
```

What the job does
- Reads JSON from HDFS, tags model column (YOLO/ViT), unions streams.
- Watermark on event time; sliding windows (e.g., 30s window, 10s slide).
- Writes Medallion layers:
  - Bronze: raw events (as-is).
  - Silver: cleaned and filtered (confidence threshold, schema fixes).
  - Gold: windowed KPIs per time window/model/emotion/source.

Example KPIs
- Per‑window counts per emotion/model/source.
- Mean/median confidence.
- Top‑k emotions per device.

## 📊 Validate
Open Spark UI → check Stages and DAGs:
- Exchange before stateful windows (shuffle),
- StateStoreRestore/Save around aggregates,
- Completed stages with task parallelism (e.g., 200/200).

HDFS layout
```
/vignesh/s5/bd/bronze_emotions/...
/vignesh/s5/bd/silver_emotions/...   # optional for inspection
/vignesh/s5/bd/gold_emotions/...
```

## 🚀 Quick Start (TL;DR)
1) Put Kaggle data into HDFS.  
2) Run spark/xyz.scala → Parquet in HDFS.  
3) Train YOLO → best.pt.  
4) Run edge producers → local JSON batches.  
5) rsync JSON → HDFS.  
6) Run streaming job → Bronze/Silver/Gold.  
7) Inspect Gold for dashboards.

## 🧭 Roadmap
- Add dashboards (Superset/Metabase).
- Add latency/throughput metrics in Gold.
- Package edge apps with a cross‑platform launcher.

## 🤝 Contributing
PRs welcome. Please open an issue for bugs or enhancements.

## 📄 License
MIT

Best option for you to explore: add small demo GIFs later (YOLO detection, Grad‑CAM, Spark stages) under a Screenshots section—this will make the README pop without pushing huge assets.
