/* PageRank.scala */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._

object PageRank {

  def main(args: Array[String]): Unit = { 
  
    // Create spark configuration and spark context
    val conf = new SparkConf().setAppName("PageRank App").setMaster("local[2]")
    val sc = new SparkContext(conf)
    
    val currentDir = System.getProperty("user.dir")  // get the current directory
    val edgeFile = "file://" + currentDir + "/followers.txt"  // define the edge file
    
    // Load the edges as a graph
    val graph = GraphLoader.edgeListFile(sc, edgeFile)
    
    // Run PageRank
    val ranks = graph.pageRank(0.0001).vertices
    
    // Join the ranks with the usernames
    val userFile = "file://" + currentDir + "/users.txt"
    val users = sc.textFile(userFile).map { line =>
      val fields = line.split(",")
      (fields(0).toLong, fields(1))
    }
    val ranksByUsername = users.join(ranks).map {
      case (id, (username, rank)) => (username, rank)
    }
    
    // Print the result
    println(ranksByUsername.collect().mkString("\n"))
  }
}


