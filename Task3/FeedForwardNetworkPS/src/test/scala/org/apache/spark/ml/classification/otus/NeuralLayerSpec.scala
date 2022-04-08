package org.apache.spark.ml.classification.otus

import breeze.linalg.{DenseMatrix, sum}
import org.scalatest.flatspec.AnyFlatSpec
import scala.math.sqrt

trait NeuralLayerSpec extends AnyFlatSpec {
  protected def frobenuis_norm(x: DenseMatrix[Double]): Double = {
    sqrt(sum(x *:* x))
  }
}
