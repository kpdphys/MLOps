package example

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.otus.FeedForwardNetworkPSClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}

import java.nio.file.Paths
import java.nio.file.Path
import scala.annotation.tailrec
import scala.sys.exit

object MainApp extends App {
  def check_args_and_get_params(args: Array[String]): (Boolean, String, String) = {
    val usage =
      """
      Usage: spark-submit ./FeedForwardNetworkPS-assembly-1.0.jar [--mode train] [--dir model_save] filename
      """

    if (args.length == 0) {
      println(usage)
      exit(1)
    }

    val arglist = args.toList
    type OptionMap = Map[Symbol, String]

    @tailrec
    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      list match {
        case Nil => map
        case "--mode" :: value :: tail =>
          nextOption(map ++ Map('mode -> value), tail)
        case "--dir" :: value :: tail =>
          nextOption(map ++ Map('dir -> value), tail)
        case string :: Nil =>  nextOption(map ++ Map('input -> string), list.tail)
        case option :: _ => println("Unknown option: " + option)
          exit(1)
      }
    }

    val options: OptionMap = nextOption(Map(), arglist)
    var is_train: Boolean = false
    var dir_path: Path = null
    var input_path: Path = null

    options.get('mode) match {
      case Some(value) if value == "train" =>
        is_train = true
      case Some(value) if value == "predict" =>
        is_train = false
      case _ => println("Mode is incorrect!")
        exit(1)
    }

    options.get('dir) match {
      case Some(value) if value.nonEmpty =>
        dir_path = Paths.get(value)
      case _ => println("Dir is incorrect!")
        exit(1)
    }

    options.get('input) match {
      case Some(value) if value.nonEmpty =>
        input_path = Paths.get(value)
      case _ => println("Input file is incorrect!")
        exit(1)
    }

    println("Command line options: " + options)
    (is_train, dir_path.toString, input_path.toString)
  }

  val (is_train, dir_str, df_str) = check_args_and_get_params(args)
  val spark = SparkSession.builder
    .appName("Feed Forward NN PS Example")
    .getOrCreate()

  val df = spark.read.parquet(df_str)
  df.printSchema()

  import spark.implicits._
  val maxUsers = 100
  val data = df.select("instanceId_userId", "instanceId_objectType",
    "user_gender", "auditweights_userAge")

  val pipeline = new Pipeline()
    .setStages(Array(
      new StringIndexer()
        .setHandleInvalid("skip")
        .setInputCol("instanceId_userId")
        .setOutputCol("instanceId_userId_label"),
      new StringIndexer()
        .setHandleInvalid("keep")
        .setInputCols(Array("instanceId_objectType", "user_gender"))
        .setOutputCols(Array("instanceId_objectType_labels", "user_gender_label")),
      new OneHotEncoder()
        .setInputCols(Array("instanceId_objectType_labels", "user_gender_label"))
        .setOutputCols(Array("instanceId_objectType_ohe", "user_gender_ohe")),
      new Imputer()
        .setInputCol("auditweights_userAge")
        .setOutputCol("auditweights_userAge_imp"),
      new VectorAssembler()
        .setInputCols(Array("auditweights_userAge_imp"))
        .setOutputCol("auditweights_userAge_vec"),
      new StandardScaler()
        .setInputCol("auditweights_userAge_vec")
        .setOutputCol("auditweights_userAge_norm"),
      new VectorAssembler()
        .setInputCols(Array("instanceId_objectType_ohe", "user_gender_ohe", "auditweights_userAge_norm"))
        .setOutputCol("features"),
      new FeedForwardNetworkPSClassifier()
        .setLayers(Array[Int](6, 50, 200, 50, maxUsers))
        .setMaxIter(10000)
        .setLearningRate(0.005)
        .setFeaturesCol("features")
        .setLabelCol("instanceId_userId_label")
    ))

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("instanceId_userId_label")
    .setMetricName("accuracy")

  val accuracy = if (is_train) {
    // Количество пользователей в исходном датасете очень большое. Поэтому, для решения задачи
    // классификации были отобраны 100 самых активных пользователей.
    val active_users = data.select("instanceId_userId").
      groupBy("instanceId_userId").count().sort($"count".desc).take(maxUsers).
      map(x => x(0).asInstanceOf[Int])
    val clipped_users = data.filter($"instanceId_userId".isin(active_users: _*))
    val splits = clipped_users.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train_dataset = splits(0)
    val test_dataset = splits(1)
    val model = pipeline.fit(train_dataset)
    model.save(dir_str)
    evaluator.evaluate(model.transform(test_dataset))
  } else {
    val model = PipelineModel.load(dir_str)
    evaluator.evaluate(model.transform(data))
  }

  println(s"Test set accuracy = $accuracy")
  spark.close()
}