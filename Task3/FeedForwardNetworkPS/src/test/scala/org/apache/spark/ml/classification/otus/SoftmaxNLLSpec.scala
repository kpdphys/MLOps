package org.apache.spark.ml.classification.otus

import breeze.linalg.DenseMatrix
import scala.math.abs

class SoftmaxNLLSpec extends NeuralLayerSpec {
  val labels: DenseMatrix[Double] = DenseMatrix((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
  val softmax_nll = new SoftmaxNLL(labels)
  val logits: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (-1.0, -0.5, 0.0), (2.0, -1.5, 0.0))
  val forward_result: Double = softmax_nll.forward(logits)
  val backward_result: DenseMatrix[Double] = softmax_nll.backward()

  "A SoftmaxNLL" should "execute forward pass" in {
    val loss = 1.08035
    assert(abs(forward_result - loss) < 1E-4)
  }

  "A SoftmaxNLL" should "execute backward pass" in {
    val dL_dx = DenseMatrix(
      (-0.303323143, 0.062107908, -0.047341063),
      (0.081576157, 0.102398629, 0.008636218),
      (0.221746985, -0.164506537, 0.038704845)
    )
    assert(frobenuis_norm(backward_result -:- dL_dx) < 1E-8)
  }
}
