from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from functools import partial
import datetime
import re

############################
# General Parameters #######
############################
HDFS_base_path="/dssp/shared/noblanc"
appName='dssp13_noblanc'
RowCountToShow=5

startTime=datetime.datetime.now()
#start "spark session" 
spark = SparkSession.builder.appName(appName).getOrCreate()
sc = spark.sparkContext

# this is added to support utf-8 international character sets
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#########################################################
# Load Parquet files
#########################################################
from pyspark.sql import SQLContext
fileName_dvf=HDFS_base_path+"/DVF_X_df_All.parquet"
sqlContext = SQLContext(sc)
loaded_df = sqlContext.read.parquet("hdfs://" + fileName_dvf)
# RowCount=1790754

#DVF_df=loaded_df.where(loaded_df.valeurfonc == null).collect()
#print "################ Data null values : "
#loaded_df.na.fill({'valeurfonc': 10}).show()

DVF_df=loaded_df.filter(loaded_df.valeurfonc.isNotNull())

# print one line of the dataframe
print "################ Data read : ", fileName_dvf
print DVF_df.show(RowCountToShow)
print "Source Data Row Count=", DVF_df.count()
print "################"

print "################ Saving Parquet Files."
fileName=HDFS_base_path+"/DVF_X_df_All_spark.parquet"
DVF_df.write.format("parquet").mode("overwrite").save("hdfs://"+fileName)
print "################ dataDF saved."

print "END."
