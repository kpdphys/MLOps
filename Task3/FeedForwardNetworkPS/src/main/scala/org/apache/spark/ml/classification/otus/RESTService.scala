package org.apache.spark.ml.classification.otus

import akka.actor.typed.ActorSystem
import akka.actor.typed.scaladsl.Behaviors
import akka.http.scaladsl.server.Route
import akka.http.scaladsl.server.Directives._
import spray.json.enrichAny
import spray.json.DefaultJsonProtocol._
import spray.json._
import akka.Done
import akka.http.scaladsl.Http
import org.slf4j.{Logger, LoggerFactory}
import scala.concurrent.{ExecutionContextExecutor, Future}

private[otus] class RESTService(val weights_count: Int,
                                val learning_rate: Double,
                                val host: String,
                                val port: Int) {

  private implicit val system: ActorSystem[Nothing] = ActorSystem(Behaviors.empty, "REST_Service")
  private implicit val executionContext: ExecutionContextExecutor = system.executionContext
  private var losses: Array[Double] = Array[Double]()

  private var weights = new Array[Double](weights_count)
  weights = weights.map(_ => scala.util.Random.nextDouble - 0.5)

  private val route: Route =
    concat (
      get {
        pathPrefix("api" / "weights") {
          onSuccess(
            Future {
              Map("weights" -> weights).toJson.prettyPrint
            }) { x => complete(x) }
        }
      },

      post {
        path("api" / "gradients") {
          entity(as[String]) { order =>
            val result = Future {
              val pars = order.parseJson.convertTo[Map[String, Array[Double]]]

              val grads = pars.get("gradients") match {
                case Some(x: Array[Double]) => x
                case _ => Array[Double]()
              }

              val id = pars.get("id") match {
                case Some(x: Array[Double]) => x(0).toInt
                case _ => 0
              }

              val iteration = pars.get("iteration") match {
                case Some(x: Array[Double]) => x(0).toInt
                case _ => 0
              }

              val loss = pars.get("loss") match {
                case Some(x: Array[Double]) => x(0)
                case _ => 0.0
              }

              val train_accuracy = pars.get("train_accuracy") match {
                case Some(x: Array[Double]) => x(0)
                case _ => 0.0
              }

              weights = weights.zip(grads).map { case (x, y) => x - learning_rate * y }
              losses = losses :+ loss
              logger.info(f"Executor id = $id%s, iteration = $iteration%3d, " +
                f"loss = $loss%.4f, train_accuracy = $train_accuracy%.4f")
              Done
            }
            onSuccess(result) { _ => complete("Ok") }
          }
        }
      }
    )

  private val bindingFuture: Future[Http.ServerBinding] = Http().newServerAt(host, port).bind(route)
  private val logger: Logger = LoggerFactory.getLogger(classOf[RESTService])
  logger.info(s"REST Service is available at http://$host:$port/api/...")

  def get_losses(): Array[Double] = losses

  def get_weights_and_close(): Array[Double] = {
    bindingFuture
      .flatMap(_.unbind())
      .onComplete(_ => system.terminate())
    logger.info(s"REST Service is shutdown")
    weights
  }
}
