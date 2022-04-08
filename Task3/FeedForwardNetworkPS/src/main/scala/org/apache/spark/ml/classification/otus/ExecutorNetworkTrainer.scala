package org.apache.spark.ml.classification.otus

import breeze.linalg.{Axis, DenseMatrix, argmax, sum}
import org.apache.spark.SparkEnv
import org.slf4j.LoggerFactory

private[otus] class ExecutorNetworkTrainer extends Serializable {
  private def calc_accuracy_metric(logits: DenseMatrix[Double],
                                   labels: DenseMatrix[Double]): Double = {
    val logits_index = argmax(logits, Axis._1)
    val labels_index = argmax(labels, Axis._1)
    sum((logits_index :== labels_index).map(x => if (x) 1 else 0)).toDouble / logits_index.size
  }

  private def make_worker_iteration(network: NeuralNetwork,
                                    loss_object: SoftmaxNLL,
                                    features: DenseMatrix[Double],
                                    targets: DenseMatrix[Double]): (Double, Double, Array[Double]) = {
    val logits = network.forward(features)
    val loss = loss_object.forward(logits)
    val acc = calc_accuracy_metric(logits, targets)
    val gradients = loss_object.backward()
    network.backward(gradients)
    val grads = network.collect_gradients()

    (loss, acc, grads)
  }

  def worker_learning(iterations_count: Int,
                      rest_host: String,
                      rest_port: Int,
                      network_layers: Array[Int],
                      iter: Iterator[(org.apache.spark.ml.linalg.Vector,
                          org.apache.spark.ml.linalg.Vector)]): Unit = {

    val features_and_targets = iter.map(row => (row._1.toArray, row._2.toArray)).toArray.unzip
    val features = breeze.linalg.DenseMatrix(features_and_targets._1 : _*)
    val targets = breeze.linalg.DenseMatrix(features_and_targets._2 : _*)
    val rest_client = new RESTClient(rest_host, rest_port)

    val network = new NeuralNetwork(network_layers)
    val loss_object = new SoftmaxNLL(targets)

    val logger = LoggerFactory.getLogger(classOf[ExecutorNetworkTrainer])

    (1 to iterations_count).foreach(i => {
      var weights = rest_client.get_weights()
      network.set_weights(weights)
      var (loss, accuracy, gradients) = make_worker_iteration(network, loss_object, features, targets)
      val pars = Map("gradients" -> gradients,
        "id" -> SparkEnv.get.executorId,
        "iteration" -> i,
        "loss" -> loss,
        "train_accuracy" -> accuracy)
      rest_client.set_parameters(pars)

      logger.debug(f"ID = ${SparkEnv.get.executorId}%s, iteration = $i%3d, " +
        f"loss = $loss%.4f, train_accuracy = $accuracy%.4f")
    })
    rest_client.close()
  }
}
