package org.apache.spark.ml.classification.otus

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, max, sum}
import breeze.numerics.exp
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.VectorImplicits.mlVectorToMLlibVector
import org.apache.spark.mllib.linalg.{Vectors => mllibVectors}

class FeedForwardNetworkPSClassifierModel private[ml] (override val uid: String,
                                                       val weights: Vector)
  extends ProbabilisticClassificationModel[Vector, FeedForwardNetworkPSClassifierModel]
  with FeedForwardNetworkPSClassifierParams with Serializable with MLWritable {

  private[otus] def this(uid : String) = this(uid, null)

  def this(weights: Vector) = this(
    Identifiable.randomUID("ff_network_ps_model"), weights)

  override lazy val numFeatures: Int = $(layers).head

  override def numClasses: Int = $(layers).last

  override def toString: String = {
    s"FeedForwardNetworkPSClassifierModel: uid=$uid, numLayers=${$(layers).length}, " +
      s"numClasses=$numClasses, numFeatures=$numFeatures"
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction.asBreeze match {
      case v: DenseVector[Double] =>
        Vectors.fromBreeze(exp(v - max(v)) / sum(exp(v - max(v))))
      case _: SparseVector[Double] =>
        throw new RuntimeException("Unexpected error in FeedForwardNetworkPSClassifierModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    protected override def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      sqlContext.sparkSession.sparkContext
        .parallelize(Seq(weights.toJson), 1).saveAsTextFile(path + "/weights")
    }
  }

  override def predictRaw(features: Vector): Vector = {
    val neural_network: NeuralNetwork = new NeuralNetwork($(layers))
    neural_network.set_weights(weights.toArray)
    val logits: DenseMatrix[Double] = neural_network.forward(features.asBreeze.toDenseVector.asDenseMatrix)
    Vectors.fromBreeze(logits.toDenseVector)
  }

  override def copy(extra: ParamMap): FeedForwardNetworkPSClassifierModel = {
    copyValues(new FeedForwardNetworkPSClassifierModel(weights), extra)
  }
}

object FeedForwardNetworkPSClassifierModel extends MLReadable[FeedForwardNetworkPSClassifierModel] {
  override def read: MLReader[FeedForwardNetworkPSClassifierModel] =
    new MLReader[FeedForwardNetworkPSClassifierModel] {
      override def load(path: String): FeedForwardNetworkPSClassifierModel = {
        val original = new DefaultParamsReader().load(path)
          .asInstanceOf[FeedForwardNetworkPSClassifierModel]

        val weights_json: String = sqlContext.sparkSession.sparkContext
          .textFile(path + "/weights", 1).first()

        original.copyValues(new FeedForwardNetworkPSClassifierModel(
          mllibVectors.fromJson(weights_json).asML)
        )
      }
    }
}
