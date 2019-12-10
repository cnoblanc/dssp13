from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from functools import partial

#start "spark session" 
spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext
#A.load tsv into a data frame:
#1.read the raw text and split it to fields (the text file does not contain a header)
dataRDD = sc.textFile('/dssp/datacamp/train.tsv').map(lambda x:x.strip().split('\t'))
#2. convert the rdd to a DataFrame and names to columns 
dataDF=dataRDD.toDF(['id','title','body','tags'])
# print one line of the dataframe
RowCountToShow=1
print "DATA to DF"
print dataDF.head(RowCountToShow)
print "################"


#B.UDF
#applying functions to a DATAFRAME requires the "SQL" logic of 
#User Defined Functions (UDF)
#as an example: split string of tags into an array
#1. define what data type your UDF returns (array of strings) : UDF SCHEMA
custom_udf_schema = ArrayType(StringType())
#2. define function and create a udf from that function and the schema
split_string_udf = udf(lambda x:x.split(','),custom_udf_schema)
#3. apply UDF to DF and create new column
dataDF = dataDF.withColumn('array_tags',split_string_udf(dataDF.tags))
print "UDF EXAMPLE"
print dataDF.head(RowCountToShow)
print "################"


#C. Drop columns
dataDF = dataDF.drop(dataDF.tags)
print "DROP COLUMN EXAMPLE"
print dataDF.head(RowCountToShow)
print "################"


#D. The next example will create 4 columns by transforming the DF to an RDD and back to a DF

#1. We need all the possible tags to create a column for each one
possible_tags = sc.broadcast([u'javascript', u'css', u'jquery', u'html'])

#2. The next function takes as a primary input a Row 
def array_string_transform(row,tags):#tags is broadcasted
	data = row.asDict() #Rows are immutable; We convert them to dictionaries
	# add new fields to the dictionary (all the tags)
	for tag in tags.value: 
		data[tag] = 0
	#set a value of 1 if the corresponding tags exists
	for existing_tag in data['array_tags']:
		data[existing_tag] = 1
	#convert the dictionary back to a Row
	newRow = Row(*data.keys()) #a. the Row object specification (column names)
	newRow = newRow(*data.values()) #b. the corresponding column values
	return newRow

#3. Create the function that we pass to "map" on an RDD 
# takes one input (the first one) 
# which is  ONE element from the RDD
# We need to "plug-in" the second parameter with the broadcasted variable 
mapFunction = partial(array_string_transform,tags=possible_tags)
#from a DF to an RDD and back	
dataDF = dataDF.rdd.map(mapFunction).toDF()
print "FROM DF TO RDD AND BACK : 4 new columns appear"
print dataDF.head(RowCountToShow)
print "###############"
print dataDF.show()
