package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix

private[otus] class ReLULayer extends NeuralLayer {
  private var is_x_gt_0: DenseMatrix[Double] = _

  override def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    is_x_gt_0 = (x >:> 0.0).map(i => if (i) 1.0 else 0.0)
    x *:* is_x_gt_0
  }

  override def backward(dL_dy: DenseMatrix[Double]): DenseMatrix[Double] = {
    dL_dy *:* is_x_gt_0.t
  }

  override def set_weights_from_array(arr: Array[Double]): Unit = {}
  override def get_gradients_as_array(): Array[Double] = {Array[Double]()}
  override def get_weights_count(): Int = { 0 }
  override def step(learning_rate: Double): Unit = {}
}
