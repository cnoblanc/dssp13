/* ConnectedComponents.scala */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._

object ConnectedComponents {

  def main(args: Array[String]): Unit = { 
  
    // Create spark configuration and spark context
    val conf = new SparkConf().setAppName("ConnectedComponents App").setMaster("local[2]")
    val sc = new SparkContext(conf)
    
    val currentDir = System.getProperty("user.dir")  // get the current directory
    val edgeFile = "file://" + currentDir + "/graph.txt"
    
    // Load the edges as a graph
    val graph = GraphLoader.edgeListFile(sc, edgeFile)
    
    // Find the connected components
    val cc = graph.connectedComponents().vertices
    

    // Print the result
    println(cc.collect().mkString("\n"))
    
  }
}


