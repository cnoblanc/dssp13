/* WordCount.scala */

import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}

object WordCount {

  def main(args: Array[String]): Unit = {
    
    println("***********************************************************************************************")
    println("***********************************************************************************************")
    
    println("Hi, this is the WordCount application for Spark.")
   
    // Create spark configuration
    val sparkConf = new SparkConf()
      .setAppName("WordCount")

    // Create spark context  
    val sc = new SparkContext(sparkConf)  // create spark context
    
    val currentDir = System.getProperty("user.dir")  // get the current directory
    val inputFile = "hdfs://master2-bigdata/dssp/data/gutenberg/*"
    //val outputDir = "file://" + currentDir + "/output"

    println("reading from input file: " + inputFile)

    val txtFile = sc.textFile(inputFile)
    
    val result = txtFile.flatMap(line => line.split(" ")) // split each line based on spaces
      .map(word => (word,1)) // map each word into a word,1 pair
      .reduceByKey(_+_) // reduce

    val localResult = result.take(100)
    localResult.foreach(println) 

    sc.stop()

    println("***********************************************************************************************")
    println("***********************************************************************************************") 
    
  }
}


