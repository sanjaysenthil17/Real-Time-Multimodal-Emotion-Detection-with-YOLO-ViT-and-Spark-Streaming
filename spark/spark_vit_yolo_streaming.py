from pyspark.sql import SparkSession, functions as F, types as T
import os

# Base paths (adjust these if needed)
HDFS_BASE = "hdfs://172.17.16.11:9000/vignesh/s5/bd"  # or "file:///local/path" for local mode

INPUTS = {
    "ViT": os.path.join(HDFS_BASE, "streaming_input_vit"),
    "YOLO": os.path.join(HDFS_BASE, "streaming_input_yolo")
}

BRONZE = os.path.join(HDFS_BASE, "bronze_emotions")
SILVER = os.path.join(HDFS_BASE, "silver_emotions")
GOLD = os.path.join(HDFS_BASE, "gold_emotions")

CKPT_BASE = os.path.join(HDFS_BASE, "checkpoints")

schema = T.StructType([
    T.StructField("ts_ms", T.LongType()),
    T.StructField("iso_ts", T.StringType()),
    T.StructField("source_id", T.StringType()),
    T.StructField("frame_id", T.LongType()),
    T.StructField("bbox", T.ArrayType(T.IntegerType())),
    T.StructField("cls_name", T.StringType()),
    T.StructField("cls_id", T.IntegerType()),
    T.StructField("conf", T.DoubleType()),
    T.StructField("img_w", T.IntegerType()),
    T.StructField("img_h", T.IntegerType()),
    T.StructField("person_id", T.IntegerType())
])

spark = (SparkSession.builder.appName("MultiModelEmotionStreaming").getOrCreate())

def create_df(model_name, path):
    df = (spark.readStream.format("json")
          .schema(schema)
          .load(path)
          .withColumn("model", F.lit(model_name))
          .withColumn("ingestion_ts", F.current_timestamp()))
    return df

# Create streaming DataFrames for both ViT and YOLO paths
df_vit = create_df("ViT", INPUTS["ViT"])
df_yolo = create_df("YOLO", INPUTS["YOLO"])

# Union both streams
raw_stream = df_vit.union(df_yolo)

# Bronze layer: raw data landing
bronze_stream = raw_stream

bronze_query = (
    bronze_stream.writeStream
    .format("parquet")
    .option("path", BRONZE)
    .option("checkpointLocation", os.path.join(CKPT_BASE, "bronze"))
    .partitionBy("model")
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start()
)

# Silver layer: clean, filter conf≥0.35
silver_stream = (bronze_stream
                 .withColumn("event_ts", F.to_timestamp("iso_ts"))
                 .withColumn("date", F.to_date("event_ts"))
                 .filter((F.col("conf") >= 0.35) & F.col("cls_name").isNotNull()))

silver_query = (
    silver_stream.writeStream
    .format("parquet")
    .option("path", SILVER)
    .option("checkpointLocation", os.path.join(CKPT_BASE, "silver"))
    .partitionBy("date", "source_id", "cls_name", "model")
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start()
)

# Gold layer: windowed aggregation per model/source/emotion
gold_stream = (silver_stream
               .withWatermark("event_ts", "2 minutes")
               .groupBy(
                   F.window("event_ts", "30 seconds", "10 seconds"),
                   F.col("source_id"),
                   F.col("cls_name"),
                   F.col("model"))
               .agg(
                   F.count("*").alias("detections"),
                   F.avg("conf").alias("avg_conf")
               )
               .select(
                   F.col("source_id"),
                   F.col("cls_name"),
                   F.col("model"),
                   F.col("detections"),
                   F.col("avg_conf"),
                   F.col("window.start").alias("win_start"),
                   F.col("window.end").alias("win_end"),
                   F.current_timestamp().alias("processed_ts")
               ))

gold_query = (
    gold_stream.writeStream
    .format("parquet")
    .option("path", GOLD)
    .option("checkpointLocation", os.path.join(CKPT_BASE, "gold"))
    .partitionBy("source_id", "cls_name", "model")
    .outputMode("append")
    .trigger(processingTime="20 seconds")
    .start()
)

print("Streaming job started. Open Spark UI (e.g., http://<master-node>:4040)")

spark.streams.awaitAnyTermination()
