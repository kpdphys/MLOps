package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix
import scala.language.postfixOps

class ReLULayerSpec extends NeuralLayerSpec {
  val relu_layer = new ReLULayer()
  val x: DenseMatrix[Double] = DenseMatrix((1.1, -2.5, 4.6), (-3.6, -2.1, -7.8), (2.4, 4.9, -6.8))
  val forward_result: DenseMatrix[Double] = relu_layer.forward(x)
  val dL_dy: DenseMatrix[Double] = DenseMatrix((-1.5, 3.8, -2.1), (4.6, 3.1, 7.2), (8.0, -3.4, -1.1))
  val backward_result: DenseMatrix[Double] = relu_layer.backward(dL_dy)

  "A ReLULayer" should "get weights count" in {
    assert(relu_layer.get_weights_count() == 0)
  }

  "A ReLULayer" should "execute forward pass" in {
    val x_masked = DenseMatrix((1.1, 0.0, 4.6), (0.0, 0.0, 0.0), (2.4, 4.9, 0.0))
    assert(frobenuis_norm(forward_result -:- x_masked) < 1E-10)
  }

  "A ReLULayer" should "execute backward pass" in {
    val dL_dy_masked = DenseMatrix((-1.5, 0.0, -2.1), (0.0, 0.0, 7.2), (8.0, 0.0, 0.0))
    assert(frobenuis_norm(backward_result -:- dL_dy_masked) < 1E-10)
  }

  "A RELULayer" should "get gradients as array" in {
    assert( relu_layer.get_gradients_as_array() isEmpty)
  }
}
