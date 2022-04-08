package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix

class DenseLayerSpec extends NeuralLayerSpec  {
  val dense_layer = new DenseLayer(2, 3)
  dense_layer.set_weights_from_array(Array[Double](
    0.1, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.6, 0.9))
  val x: DenseMatrix[Double] = DenseMatrix((1.0, 2.0), (4.0, 5.0), (7.0, 8.0))
  val forward_result: DenseMatrix[Double] = dense_layer.forward(x)
  val dL_dy: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
  val backward_result: DenseMatrix[Double] = dense_layer.backward(dL_dy)

  "A DenseLayer" should "get weights count" in {
    assert(dense_layer.get_weights_count() == 9)
  }

  "A DanseLayer" should "execute forward pass" in {
    val x_dot_w = DenseMatrix((1.6, 2.0, 2.4), (3.1, 4.1, 5.1), (4.6, 6.2, 7.8))
    assert(frobenuis_norm(forward_result -:- x_dot_w) < 1E-10)
  }

  "A DanseLayer" should "execute backward pass" in {
    val w_dot_dL_dy = DenseMatrix((3.0, 3.6, 4.2), (6.6, 8.1, 9.6))
    assert(frobenuis_norm(backward_result -:- w_dot_dL_dy) < 1E-10)
  }

  "A DanseLayer" should "get gradients as array" in {
    val grads = Array[Double](30.0, 36.0, 6.0, 66.0, 81.0, 15.0, 102.0, 126.0, 24.0)
    assert( dense_layer.get_gradients_as_array() sameElements grads)
  }
}
