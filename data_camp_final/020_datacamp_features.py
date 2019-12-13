from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from functools import partial
import datetime
import sys,re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from nltk.stem.porter import SnowballStemmer

############################
# General Parameters #######
############################
HDFS_base_path="/user/christophe.noblanc/datacamp"
appName='christophe'
RowCountToShow=5

# Christophe: This function was post in the Slack during the DeepLearning weekend.
def clean_str(str_dirty):
	# str_dirty will be analyzed characher by character or using regular expressions
	str_dirty=str_dirty.strip() # Remove spaces before and after
	stop_words = stopwords.words('english')
	exclude = set(string.punctuation)
	exclude_number = '0123456789'
	exclude_other = ''
	
	str_dirty = ''.join(ch for ch in str_dirty if ch not in exclude)
	str_dirty = ''.join(ch for ch in str_dirty if ch not in exclude_number)
	str_dirty = ''.join(ch for ch in str_dirty if ch not in exclude_other)
	str_dirty = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str_dirty)
	str_dirty = re.sub(r"\'s", " \'s", str_dirty)
	str_dirty = re.sub(r"\'ve", " \'ve", str_dirty)
	str_dirty = re.sub(r"n\'t", " n\'t", str_dirty)
	str_dirty = re.sub(r"\'re", " \'re", str_dirty)
	str_dirty = re.sub(r"\'d", " \'d", str_dirty)
	str_dirty = re.sub(r"\'ll", " \'ll", str_dirty)
	str_dirty = re.sub(r",", " , ", str_dirty)
	str_dirty = re.sub(r"!", " ! ", str_dirty)
	str_dirty = re.sub(r"\(", " \( ", str_dirty)
	str_dirty = re.sub(r"\)", " \) ", str_dirty)
	str_dirty = re.sub(r"\?", " \? ", str_dirty)
	str_dirty = re.sub(r"\s{2,}", " ", str_dirty)
	str_clean=str_dirty
	return str_clean.strip()

def remove_stop_words(word_dirty):
	# Check if a specific word (parameter) should be excluded or not
	stop_words = stopwords.words('english')
	other_specific_exclude=['']

	str_clean=word_dirty.lower().strip() # Remove spaces before and after
	if str_clean in stop_words:
		str_clean=''
	if str_clean in other_specific_exclude:
		str_clean=''
	return str_clean

def stemmer_(list_of_words):
	# version 1 : 
	stemmer = PorterStemmer()
	# Version 2 : SnowballStemmer (The 'english' stemmer is better than the original 'porter' stemmer)
	# stemmer = SnowballStemmer("english", ignore_stopwords=True)
	doc = [stemmer.stem(w) for w in list_of_words]
	return(doc)

startTime=datetime.datetime.now()
#start "spark session" 
spark = SparkSession.builder.appName(appName).getOrCreate()
sc = spark.sparkContext
#A.load tsv into a data frame:
#1.read the raw text and split it to fields (the text file does not contain a header)
dataRDD = sc.textFile('/dssp/datacamp/train.tsv').map(lambda x:x.strip().split('\t'))
#2. convert the rdd to a DataFrame and names to columns 
print "################ Start loading Train data"
dataDF=dataRDD.toDF(['id','title','body','tags'])
# print one line of the dataframe
print "################ DATA to DF"
print dataDF.show(RowCountToShow)
print "Source Data (Train&Test) Row Count=",dataDF.count()
#print dataDF.head(RowCountToShow)
print "################ columns :"
print dataDF.printSchema()


#########################################################
# PART I : Manage the Tags (only applyied in Train dataset)
# Extract the Tags into an array and then split the Tags in columns Flags 0/1
#########################################################
# B.UDF
# applying functions to a DATAFRAME requires the "SQL" logic of 
# User Defined Functions (UDF)
# as an example: split string of tags into an array
# 1. define what data type your UDF returns (array of strings) : UDF SCHEMA
custom_udf_schema = ArrayType(StringType())
# 2. define function and create a udf from that function and the schema
split_string_udf = udf(lambda x:x.split(','),custom_udf_schema)
# 3. apply UDF to DF and create new column
dataDF = dataDF.withColumn('array_tags',split_string_udf(dataDF.tags))
print "################ UDF EXAMPLE : split string of tags into an array"
#print dataDF.show(RowCountToShow)
#print dataDF.head(RowCountToShow)
#print "################"

#C. Drop columns
dataDF = dataDF.drop(dataDF.tags)
#print "DROP COLUMN EXAMPLE"
#print dataDF.head(RowCountToShow)
#print "################"

# D. The next example will create 4 columns by transforming the DF
#    to an RDD and back to a DF
# 1. We need all the possible tags to create a column for each one
possible_tags = sc.broadcast([u'javascript', u'css', u'jquery', u'html'])

# 2. The next function takes as a primary input a Row 
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
# takes one input (the first one), which is  ONE element from the RDD
# We need to "plug-in" the second parameter with the broadcasted variable 
mapFunction = partial(array_string_transform,tags=possible_tags)
#from a DF to an RDD and back	
dataDF = dataDF.rdd.map(mapFunction).toDF()
print "############### : FROM DF TO RDD AND BACK : 4 new columns appear"
#print dataDF.show(RowCountToShow)

#########################################################
# PART II
# Features EXAMPLES
#########################################################
# Classic TF-IDF (with hashing)
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# 1. split text field into words (Tokenize)
tokenizer = Tokenizer(inputCol="title", outputCol="words_title")
dataDF = tokenizer.transform(dataDF)
print "################ : New column with tokenized Title"
#print dataDF.show(RowCountToShow)

# 2. compute term frequencies
hashingTF = HashingTF(inputCol="words_title", outputCol="tf_title")
dataDF = hashingTF.transform(dataDF)
print "################ TERM frequencies:"
#print dataDF.show(RowCountToShow)
#print dataDF.head(RowCountToShow)
#print "################"

#3. IDF computation
idf = IDF(inputCol="tf_title", outputCol="tf_idf_title")
print "################ TF_IDF vector: Start fit()"
idfModel = idf.fit(dataDF) #model that contains "dictionary" and IDF values
print "################ TF_IDF vector: Start transform()"
dataDF = idfModel.transform(dataDF)
print "################ TF_IDF vector:"
print dataDF.show(RowCountToShow)
#print dataDF.head(RowCountToShow)
print "################"

#########################################################
# Apply same data Prep process (pipeline) on Valid Data
#########################################################
#1. load test data
validDF = sc.textFile('/dssp/datacamp/test.tsv').map(lambda x:x.strip().split('\t')).toDF(['id','title','body'])
print "##### (Valid) ########## dataset loaded  "
#2.transform test data
validDF = tokenizer.transform(validDF)
print "##### (Valid) ########## tokenized Title : done."
validDF = hashingTF.transform(validDF)
print "##### (Valid) ########## Term Frequencies : done."
validDF = idfModel.transform(validDF)
print "##### (Valid) ########## TF_IDF vector : done."

#########################################################
# Save prepared Data for ML next step
#########################################################
fileName_train=HDFS_base_path+"/train_features_001.parquet"
fileName_valid=HDFS_base_path+"/valid_features_001.parquet"
dataDF.write.format("parquet").mode("overwrite").save("hdfs://"+fileName_train)
validDF.write.format("parquet").mode("overwrite").save("hdfs://"+fileName_valid)

print "---------------------- SUMMARY :"
print "Source Data (Train&Test) Row Count=",dataDF.count()
print dataDF.show(RowCountToShow)
print "Validation Row Count=",validDF.count()
print validDF.show(RowCountToShow)
print "################ : END."
endTime=datetime.datetime.now()
duration = endTime - startTime 
duration_tuple=divmod(duration.total_seconds(), 60)
print "Total Duration Time (m)= ",duration_tuple[0] 
print "Total Duration Time (s)= ",duration_tuple[1] 
