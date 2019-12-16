/* Planets.scala 					*/
/*							*/
/*							*/
/*							*/

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Planets {
  def main(args: Array[String]) {
  
    println("***********************************************************************************************")
    println("Hi, this is the Planets application for Spark.")
    
    // Create spark configuration and spark context
    val conf = new SparkConf().setAppName("Planets App")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //import sqlContext.createSchemaRDD
    
    val inputFile = "hdfs://master2-bigdata/dssp/data/planets/planets.json"

    println(inputFile)

    val planets = sqlContext.jsonFile(inputFile)

    planets.printSchema()
    planets.registerTempTable("planets")

    val smallPlanets = sqlContext.sql("SELECT name,sundist,radius FROM planets WHERE radius < 10000")

    smallPlanets.collect().foreach(println) 
    
    sc.stop()
    println("***********************************************************************************************")
  }
}
