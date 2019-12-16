/* TriangleCounting.scala */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._

object TriangleCounting {

  def main(args: Array[String]): Unit = { 
  
    // Create spark configuration and spark context
    val conf = new SparkConf().setAppName("TriangleCounting App") //.setMaster("local[2]")
    val sc = new SparkContext(conf)
    
    val currentDir = System.getProperty("user.dir")  // get the current directory
    //val edgeFile = "file://" + currentDir + "/enron.txt"
    val edgeFile = "hdfs://master2-bigdata/dssp/data/enron"
             
    // Load the edges in canonical order and partition the graph for triangle count
    val graph = GraphLoader.edgeListFile(sc, edgeFile, true).partitionBy(PartitionStrategy.RandomVertexCut)

    // Find the triangle count for each vertex
    val triCounts = graph.triangleCount().vertices
    
    println(triCounts.take(100).mkString("\n"))
    
  }
}


