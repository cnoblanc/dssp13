/* LineCount.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object LineCount {
  def main(args: Array[String]) {
  
    println("***********************************************************************************************")
    println("***********************************************************************************************")
    println("Hi, this is the LineCount application for Spark.")
    
    // Create spark configuration and spark context
    val conf = new SparkConf().setAppName("LineCount App")
    val sc = new SparkContext(conf)
    
    val currentDir = System.getProperty("user.dir")  // get the current directory
    val inputFile = "file://" + currentDir + "/leonardo.txt"
    val outputDir = "file://" + currentDir + "/output"
    
    println("reading from input file: " + inputFile)
    
    val myData = sc.textFile(inputFile, 2).cache()
    val num1 = myData.filter(line => line.contains("the")).count()
    val num2 = myData.filter(line => line.contains("and")).count()
    val totalLines = myData.map(line => 1).count
    println("Total lines: %s, lines with \"the\": %s, lines with \"and\": %s".format(totalLines, num1, num2))
    
    sc.stop()
    println("***********************************************************************************************")
    println("***********************************************************************************************")
  }
}
