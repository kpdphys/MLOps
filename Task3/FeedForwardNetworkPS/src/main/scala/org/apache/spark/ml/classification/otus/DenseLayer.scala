package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix

private[otus] class DenseLayer(val in_dim: Int,
                               val out_dim: Int) extends NeuralLayer {

  private var w: DenseMatrix[Double] = DenseMatrix.zeros[Double](in_dim + 1, out_dim)
  private var x: DenseMatrix[Double] = _
  private var dL_dw: DenseMatrix[Double] = _

  override def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.x = DenseMatrix.horzcat(x, DenseMatrix.ones[Double](x.rows, 1))
    this.x * w
  }

  override def backward(dL_dy: DenseMatrix[Double]): DenseMatrix[Double] = {
    dL_dw = dL_dy * this.x
    val dL_dx = w * dL_dy
    dL_dx(0 to -2, ::)
  }

  override def set_weights_from_array(arr: Array[Double]): Unit = {
    w = new DenseMatrix(in_dim + 1, out_dim, arr)
  }

  override def get_gradients_as_array(): Array[Double] = {
    dL_dw.t.toArray
  }

  override def get_weights_count(): Int = { (in_dim + 1) * out_dim }

  override def step(learning_rate: Double): Unit = {
    w = w - learning_rate * dL_dw.t
  }
}
