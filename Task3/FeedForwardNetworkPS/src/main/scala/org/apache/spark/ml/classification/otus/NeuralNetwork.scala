package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix

private[otus] class NeuralNetwork(val layers_dims: Array[Int]) {
  private val layers: Array[NeuralLayer] = create_layers(layers_dims)
  private var offsets: Array[Int] = Array[Int]()
  for (layer <- layers) {
    offsets = offsets :+ layer.get_weights_count()
  }
  offsets = offsets.scanLeft(0)(_ + _)

  private def create_layers(l_dims: Array[Int]): Array[NeuralLayer] = {
    var layers = Array[NeuralLayer]()
    l_dims.zip(l_dims.tail).zipWithIndex.foreach{ case ((in, out), i) =>
      layers = layers :+ new DenseLayer(in, out)
      if (i < l_dims.length - 2) {
        layers = layers :+ new ReLULayer()
      }
    }
    layers
  }

  def get_weights_count(): Int = offsets.last

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    var input = x.copy
    for (layer <- layers) {
      input = layer.forward(input)
    }
    input
  }

  def backward(dL_dy: DenseMatrix[Double]): Unit = {
    var input = dL_dy.copy
    for (layer <- layers.reverse) {
      input = layer.backward(input)
    }
  }

  def collect_gradients(): Array[Double] = {
    var all_gradients = Array[Array[Double]]()
    for (layer <- layers) {
      all_gradients = all_gradients :+ layer.get_gradients_as_array()
    }
    all_gradients.flatten
  }

  def set_weights(arr: Array[Double]): Unit = {
    for ((layer, i) <- layers.zipWithIndex) {
      layer.set_weights_from_array(arr.slice(offsets(i), offsets(i + 1)))
    }
  }

  def step(learning_rate: Double): Unit = {
    for (layer <- layers) {
      layer.step(learning_rate)
    }
  }
}
