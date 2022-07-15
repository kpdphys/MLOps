package streaming

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.mllib.regression.{LabeledPoint, StreamingLinearRegressionWithSGD}
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.mllib.linalg.Vectors
import org.json4s._
import org.json4s.jackson.JsonMethods._
import scala.sys.exit


object LRStreamRDD {
  def jsonStrToMap(jsonStr: String): Map[String, String] = {
    implicit val formats: DefaultFormats.type = org.json4s.DefaultFormats
    parse(jsonStr).extract[Map[String, String]]
  }

  def one_hot_encode(category: String,
                     indexes: Map[String, Int]): Array[Double] = {
    val one_hot: Array[Double] = Array.fill(indexes.size + 1)(0.0)
    val ind = indexes.getOrElse(category, 0)
    one_hot(ind) = 1.0
    one_hot
  }

  def min_max_scaler(value: String,
                     val_min: Double = 0.0,
                     val_max: Double = 1.0): Array[Double] = {
    def toDouble(s: String): Option[Double] = {
      try {
        Some(s.toDouble)
      } catch {
        case _: Throwable => None
      }
    }

    def norm(value: Double,
             val_min: Double,
             val_max: Double): Double = {
      (value - val_min) / (val_max - val_min)
    }

    val value_double = toDouble(value) match {
      case Some(x) => norm(x, val_min, val_max)
      case None => norm(0.5 * (val_min + val_max), val_min, val_max)
    }
    Array(value_double)
  }

  def flat_arrays(arr: Array[Array[Double]]): Array[Double] = {
    arr.flatMap(_.toList)
  }

  def main(args: Array[String]): Unit = {
    val usage =
      """
      Usage: spark-submit ./SparkStreamingKafkaSNA-assembly-1.0.jar <bootsrap-servers> <kafka-topic>
      """

    if (args.length < 2) {
      println(usage)
      exit(1)
    }

    val conf = new SparkConf()
      .setAppName("SparkStreamingKafkaSNA")
      .setMaster("local[*]")
      .set("spark.streaming.kafka.maxRatePerPartition", "100")

    val sparkContext = new SparkContext(conf)
    val streamingContext = new StreamingContext(sparkContext, Seconds(1))

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> args(0),
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "sna_data_1",
      "auto.offset.reset" -> "latest",
      "enable.auto.commit" -> (true: java.lang.Boolean)
    )

    val topics = Array(args(1))
    val stream = KafkaUtils.createDirectStream[String, String](
      streamingContext,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )

    val indexed_features = Map(
      "audit_client_type" -> Map("WEB" -> 1, "MOB" -> 2, "API" -> 3),
      "instanceId_objectType" -> Map("Photo" -> 1, "Post" -> 2, "Video" -> 3),
      "audit_resourceType" -> Map("3" -> 1, "6" -> 2, "7" -> 3, "8" -> 4, "14" -> 5),
    )

    val dict_stream = stream.map(record=>jsonStrToMap(record.value()))
    val data = dict_stream.map(m => (Array(
      one_hot_encode(m("audit_clientType"), indexed_features("audit_client_type")),
      one_hot_encode(m("instanceId_objectType"), indexed_features("instanceId_objectType")),
      one_hot_encode(m("audit_resourceType"), indexed_features("audit_resourceType")),
      min_max_scaler(m("auditweights_numDislikes"), 0.0, 2E+6),
      min_max_scaler(m("auditweights_ctr_negative"), 0.0, 17.0),
      min_max_scaler(m("auditweights_numLikes"), -2000.0, 8E+6),
      min_max_scaler(m("metadata_numSymbols"), 0.0, 64000.0),
    ),
      min_max_scaler(m("metadata_numPhotos"), 0.0, 326.0))
    )

    val data_labeled_points = data.map(m => LabeledPoint(m._2(0), Vectors.dense(flat_arrays(m._1))))

    val numFeatures = 4 + indexed_features.values.map(x => x.size + 1).sum

    val model = new StreamingLinearRegressionWithSGD()
      .setInitialWeights(Vectors.zeros(numFeatures))
      .setRegParam(0.05)

    model.trainOn(data_labeled_points)
    data_labeled_points.reduce((x, y) => LabeledPoint(0.0, Vectors.dense(Array(0.0))))
      .map(_ => model.latestModel().weights.toArray.mkString("weights: [", ", ", "]")).print()

    streamingContext.start()
    streamingContext.awaitTermination()
    streamingContext.stop()
  }
}
