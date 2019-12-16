import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
  
object PageRank {
  
  def main(args: Array[String]) {
  
    println("***********************************************************************************************")
    println("***********************************************************************************************")
  
    val iters = 10  // number of iterations for pagerank computation
    val currentDir = System.getProperty("user.dir")  // get the current directory
    val inputFile = "file://" + currentDir + "/webgraph.txt"
    val outputDir = "file://" + currentDir + "/output"
  
    println("reading from input file: " + inputFile)
  
    val sparkConf = new SparkConf().setAppName("PageRank").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val lines = sc.textFile(inputFile, 1)
  
    val links = lines.map{ s =>
    val parts = s.split("\\s+")
    (parts(0), parts(1))
    }.distinct().groupByKey().cache()
 
  
    var ranks = links.mapValues(v => 1.0)
    for (i <- 1 to iters) {
      println("Iteration: " + i)
      val contribs = links.join(ranks).values.flatMap{ case (urls, rank) =>
      val size = urls.size
      urls.map(url => (url, rank / size)) }

      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }

    val output = ranks.collect()
    output.foreach(tup => println(tup._1 + " has rank: " + tup._2 + "."))
  
    sc.stop()
  
    println("***********************************************************************************************")
    println("***********************************************************************************************")
  
  }
}