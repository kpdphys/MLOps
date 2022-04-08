package org.apache.spark.ml.classification.otus

import org.apache.http.HttpHeaders
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import org.apache.http.client.methods.{CloseableHttpResponse, HttpGet, HttpPost}
import org.apache.http.util.EntityUtils
import org.apache.http.entity.StringEntity
import spray.json._
import DefaultJsonProtocol._

class RESTClient(val host: String,
                 val port: Int) {
  private val client: CloseableHttpClient = HttpClientBuilder.create().build()
  private val post = new HttpPost(s"http://$host:$port/api/gradients")
  post.addHeader(HttpHeaders.CONTENT_TYPE,"application/json")
  private val get  = new HttpGet(s"http://$host:$port/api/weights")

  def get_weights(): Array[Double] = {
    val response: CloseableHttpResponse = client.execute(get)
    val entity = response.getEntity
    val json_weights = EntityUtils.toString(entity,"UTF-8")
    json_weights.parseJson.convertTo[Map[String, Array[Double]]]
      .get("weights") match {
      case Some(x: Array[Double]) => x
      case _ => Array[Double]()
    }
  }

  def set_parameters(pars: Map[String, Any]): Unit = {
    def convert_to_array(x: Any): Array[Double] = {
      x match {
        case x: Array[Double] => x
        case x: Double => Array[Double](x)
        case x => Array[Double](x.toString.toDouble)
      }
    }
    val pars_array = pars map { case (key, value) => (key, convert_to_array(value)) }
    val json = pars_array.toJson.prettyPrint
    post.setEntity(new StringEntity(json))
    val response = client.execute(post)
    val entity = response.getEntity
    val answer = EntityUtils.toString(entity,"UTF-8")
    if (answer != "Ok") throw new RuntimeException("POST request has got an invalid response!")
  }

  def close(): Unit = {
    client.close()
  }
}
