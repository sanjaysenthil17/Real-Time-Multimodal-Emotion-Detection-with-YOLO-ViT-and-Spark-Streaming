import org.apache.spark.sql.{SparkSession}
import org.apache.spark.sql.functions._

object EmotionPreprocessing {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Emotion Preprocessing")
      .getOrCreate()

    import spark.implicits._

    val basePath = "hdfs://asaicomputemaster:9000/vignesh/s5/bd"
    val datasetPath = s"$basePath/dataset"
    val outputPath = s"$basePath/dataset_parquet"

    def loadImages(split: String) = {
      val imgPath = s"$datasetPath/$split/images"
      spark.read.format("image")
        .load(s"$imgPath/*.{jpg,png}")
        .select($"image.origin".alias("path"), $"image.data".alias("data"))
        .withColumn("key", regexp_extract($"path", "([^/]+)\\.(jpg|png)$", 1))
    }

    def loadLabels(split: String) = {
      val lblPath = s"$datasetPath/$split/labels"
      spark.read.textFile(s"$lblPath/*.txt")
        .withColumn("label", org.apache.spark.sql.functions.split($"value", " ").getItem(0).cast("int"))
        .withColumn("key", regexp_extract(input_file_name(), "([^/]+)\\.txt$", 1))
        .select("key", "label")
        .distinct()
    }

    def prepareSplit(split: String) = {
      val images = loadImages(split)
      val labels = loadLabels(split)
      images.join(labels, "key")
        .withColumn("height", lit(224))
        .withColumn("width", lit(224))
        .withColumn("nChannels", lit(3))
        .withColumn("split", lit(split))
    }

    val trainDF = prepareSplit("train")
    val validDF = prepareSplit("valid")
    val testDF  = prepareSplit("test")
    val fullDF  = trainDF.union(validDF).union(testDF)

    // Save parquet with compression
    spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
    fullDF.write.mode("overwrite").parquet(outputPath)

    // Print dataset counts
    println(s"Train set count: ${trainDF.count()}")
    println(s"Validation set count: ${validDF.count()}")
    println(s"Test set count: ${testDF.count()}")
    println(s"Total dataset count: ${fullDF.count()}")

    spark.stop()
  }
}
