import sbtassembly.MergeStrategy

ThisBuild / version := "1.0"

ThisBuild / scalaVersion := "2.12.15"

lazy val root = (project in file("."))
  .settings(
    name := "SparkStreamingKafkaSNA",
    assembly / mainClass := Some("streaming.LRStreamRDD")
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.2.1" % "provided",
  "org.scalanlp" %% "breeze" % "1.2" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.2.1" % "provided",
  "org.apache.spark" %% "spark-streaming-kafka-0-10" % "3.2.1",
  "org.slf4j" % "slf4j-api" % "1.7.36",
  "org.slf4j" % "slf4j-simple" % "1.7.36",
)

ThisBuild / assemblyMergeStrategy := {
  case PathList("reference.conf") => MergeStrategy.concat
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}
