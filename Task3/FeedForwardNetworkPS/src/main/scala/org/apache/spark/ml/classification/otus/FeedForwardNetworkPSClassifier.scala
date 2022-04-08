package org.apache.spark.ml.classification.otus

import org.apache.spark.ml.classification.ProbabilisticClassifier
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.sql.{Dataset, Row}
import java.net.{InetAddress, ServerSocket}

class FeedForwardNetworkPSClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, FeedForwardNetworkPSClassifier,
    FeedForwardNetworkPSClassifierModel]
    with FeedForwardNetworkPSClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("ff_network_ps_classifier"))

  def setLayers(value: Array[Int]): this.type = set(layers, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): FeedForwardNetworkPSClassifierModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, predictionCol,
      rawPredictionCol, layers, maxIter, learningRate, thresholds)

    instr.logNumClasses($(layers).last)
    instr.logNumFeatures($(layers).head)

    val encodedLabelCol = "_encoded" + $(labelCol)
    val encodeModel = new OneHotEncoderModel(uid, Array($(layers).last))
      .setInputCols(Array($(labelCol)))
      .setOutputCols(Array(encodedLabelCol))
      .setDropLast(false)

    val encodedDataset = encodeModel.transform(dataset)
    val data = encodedDataset.select($(featuresCol), encodedLabelCol).rdd.map {
      case Row(features: Vector, encodedLabel: Vector) => (features, encodedLabel)
    }

    val rest_host: String = dataset.sqlContext.sparkSession.sparkContext.getConf.get("spark.driver.host")
    val socket = new ServerSocket(0, 0, InetAddress.getLoopbackAddress)
    val rest_port = try {
      socket.getLocalPort
    } finally {
      socket.close()
    }

    val neural_network = new NeuralNetwork($(layers))
    val weights_count: Integer = neural_network.get_weights_count()
    val rest_service = new RESTService(weights_count, getLearningRate, rest_host, rest_port)

    data.foreachPartition(iter => new ExecutorNetworkTrainer().worker_learning(getMaxIter, rest_host,
      rest_port, getLayers, iter))

    val weights: Array[Double] = rest_service.get_weights_and_close()
    new FeedForwardNetworkPSClassifierModel(Vectors.dense(weights))
  }
}

object FeedForwardNetworkPSClassifier extends DefaultParamsReadable[FeedForwardNetworkPSClassifier]