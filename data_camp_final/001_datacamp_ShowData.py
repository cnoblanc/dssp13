from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from functools import partial
import datetime
import re

############################
# General Parameters #######
############################
HDFS_base_path="/user/christophe.noblanc/datacamp"
appName='christophe'
RowCountToShow=5

startTime=datetime.datetime.now()
#start "spark session" 
spark = SparkSession.builder.appName(appName).getOrCreate()
sc = spark.sparkContext

#########################################################
# Load Parquet files
#########################################################
from pyspark.sql import SQLContext
fileName_train=HDFS_base_path+"/train_features_001.parquet"
fileName_valid=HDFS_base_path+"/valid_features_001.parquet"
sqlContext = SQLContext(sc)
print "################ Start loading Train DataFrame"
dataDF = sqlContext.read.parquet("hdfs://" + fileName_train)
print "################ Start loading Valid DataFrame"
validDF = sqlContext.read.parquet("hdfs://" + fileName_valid)
print "################ Loading DataFrame End"

# print lines of the dataframe
print "################ Train Data read"
print dataDF.show(RowCountToShow)
#print "Source Data (Train&Test) Row Count=",dataDF.count()
print "############### Valid Data read"
print validDF.show(RowCountToShow)
#print "Validation Row Count=",validDF.count()

