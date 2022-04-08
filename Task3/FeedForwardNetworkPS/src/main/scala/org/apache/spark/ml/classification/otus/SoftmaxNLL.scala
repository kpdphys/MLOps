package org.apache.spark.ml.classification.otus

import breeze.linalg._

private[otus] class SoftmaxNLL(val labels: DenseMatrix[Double]) {
  private var y: DenseMatrix[Double] = _

  def forward(x: DenseMatrix[Double]): Double = {
    val exp_x = breeze.numerics.exp(x)
    y = exp_x(::, *) /:/ breeze.linalg.sum(exp_x(*, ::))

    -1.0 / x.rows * breeze.linalg.sum(x *:* labels) +
      1.0 / x.rows * breeze.linalg.sum(
        breeze.numerics.log(
          breeze.linalg.sum(
            exp_x(*,::)
          )
        )
      )
  }

  def backward(): DenseMatrix[Double] = {
    1.0 / y.rows * (y - labels).t
  }
}

