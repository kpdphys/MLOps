package org.apache.spark.ml.classification.otus

import org.apache.spark.ml.classification.ProbabilisticClassifierParams
import org.apache.spark.ml.param.{DoubleParam, IntArrayParam, ParamValidators}
import org.apache.spark.ml.param.shared.HasMaxIter

private[otus] trait FeedForwardNetworkPSClassifierParams extends ProbabilisticClassifierParams
  with HasMaxIter {

  final val layers: IntArrayParam = new IntArrayParam(this, "layers",
    "Sizes of layers from input layer to output layer. " +
      "E.g., Array(780, 100, 10) means 780 inputs, " +
      "one hidden layer with 100 neurons and output layer of 10 neurons.",
    (t: Array[Int]) => t.forall(ParamValidators.gt(0)) && t.length > 1)

  final def getLayers: Array[Int] = $(layers)

  final val learningRate: DoubleParam = new DoubleParam(this, "learningRate",
    "The learning rate for the model", (x: Double) => x > 0)

  final def getLearningRate: Double = $(learningRate)

  setDefault(maxIter -> 50, learningRate -> 0.01)
}
