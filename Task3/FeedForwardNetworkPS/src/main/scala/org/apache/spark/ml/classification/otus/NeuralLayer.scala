package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix

private[otus] trait NeuralLayer {
  def forward(x: DenseMatrix[Double]): DenseMatrix[Double]
  def backward(dL_dy: DenseMatrix[Double]): DenseMatrix[Double]
  def set_weights_from_array(arr: Array[Double]): Unit
  def get_gradients_as_array(): Array[Double]
  def get_weights_count(): Int
  def step(learning_rate: Double): Unit
}
