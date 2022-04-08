import sbtassembly.MergeStrategy

ThisBuild / version := "1.0"

ThisBuild / scalaVersion := "2.12.15"

lazy val root = (project in file("."))
  .settings(
    name := "FeedForwardNetworkPS",
    assembly / mainClass := Some("example.MainApp")
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.2.1" % "provided",
  "org.scalanlp" %% "breeze" % "1.2" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.2.1" % "provided",
  "com.typesafe.akka" %% "akka-actor-typed" % "2.6.18",
  "com.typesafe.akka" %% "akka-distributed-data" % "2.6.18",
  "com.typesafe.akka" %% "akka-stream" % "2.6.18",
  "com.typesafe.akka" %% "akka-http" % "10.2.9",
  "com.typesafe.akka" %% "akka-http-spray-json" % "10.2.9",
  "io.spray" %%  "spray-json" % "1.3.6",
  "org.apache.httpcomponents" % "httpclient" % "4.5.13",
  "org.slf4j" % "slf4j-api" % "1.7.36",
  "org.slf4j" % "slf4j-simple" % "1.7.36",
  "org.scalatest" %% "scalatest" % "3.2.11" % "test",
  "org.scalatest" %% "scalatest-flatspec" % "3.2.11" % "test",
)

ThisBuild / assemblyMergeStrategy := {
  case PathList("reference.conf") => MergeStrategy.concat
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}