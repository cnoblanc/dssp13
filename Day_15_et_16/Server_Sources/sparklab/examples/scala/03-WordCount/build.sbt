assemblySettings
name := "WordCount"
version := "1.0"
scalaVersion := "2.10.4-local"
scalaHome := Some(file("/opt/scala/scala-2.10.4"))
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
