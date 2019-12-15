from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf,concat,col,lit,lower,regexp_replace,ltrim,rtrim
from functools import partial
import datetime
import sys,re, string
import nltk
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from bs4 import BeautifulSoup

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover,NGram

############################
# General Parameters #######
############################
HDFS_base_path="/dssp/shared/noblanc"
appName='christophe'
RowCountToShow=5
#train_filename='/dssp/datacamp/little_train.tsv'
train_filename='/dssp/datacamp/train.tsv'

# Remove HTML Tags from a column
class BsTextExtractor(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(BsTextExtractor, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    def _transform(self, dataset):
        def f(s):
            cleaned_post = BeautifulSoup(s).text
            return cleaned_post
        t = StringType()
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))

####################################
################ It start Here     #
####################################
startTime=datetime.datetime.now()
#start "spark session" 
spark = SparkSession.builder.appName(appName).getOrCreate()
sc = spark.sparkContext
#A.load tsv into a data frame:
#1.read the raw text and split it to fields (the text file does not contain a header)
dataRDD = sc.textFile(train_filename).map(lambda x:x.strip().split('\t'))
# .sample(False,0.5,42)

#2. convert the rdd to a DataFrame and names to columns 
print "################ Start loading Train data"
dataDF=dataRDD.toDF(['id','title','body','tags'])
# print one line of the dataframe
print "################ DATA to DF"
print dataDF.show(RowCountToShow)
#print "Source Data (Train&Test) Row Count=",dataDF.count()
#print dataDF.show(RowCountToShow)
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
#print "################"

#C. Drop columns
dataDF = dataDF.drop(dataDF.tags)
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
	# EnCoding of Tags in the same order of possible_tags : ex '0101'=css+html
	data['tags_target']=''
	for existing_tag in possible_tags.value:
		data['tags_target']=data['tags_target']+str(data[existing_tag])
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

#columns_to_drop = ['javascript', 'css', 'jquery', 'html']
#dataDF = dataDF.drop(*columns_to_drop)

print "############### : FROM DF TO RDD AND BACK : 4 new columns appear"
#print dataDF.show(RowCountToShow)
#print "###############"

#########################################################
# PART II
# Features EXAMPLES
# TODO : DONE. create a new col "text" that contains only the title,
#        and use it to extract words and calculate the TF & TF_IDF
#		 -> impact on the next step ML (changing the column name of the splitted words)
# TODO : DONE. using Stemming
# TODO : DONE. remove HTML tags from the "body"
# TODO : DONE. not only use "title" , but also use "body" and "title".
# TODO : DONE. exclude ponctuation and unwanted caracters, exclude stopwords
# TODO : DONE. Using n-grams (not only 1-word, but sequence of words) : 2-grams, 3-grams, 4-grams.
#	 	 https://spark.apache.org/docs/latest/ml-features.html#n-gram
# TODO : DONE. feature selection (enlever les mots les moins utiles)
# TODO : In order to increase the accuracy of the results you may select a limited number of 
#		  features using the chi- squared test. This way, you may decrease the number of dimensions 
#		  by keeping the most important once, eliminating features that make your data noisy.
# TODO : replace lemma (synomymes, nettoyage des conjugaisons, pluriels)
# TODO : in "text" field, use 2 or 3 times the "title" (to increase importance of the title)
#
#########################################################

# DQ Function on the Text fields (entire document columns)
def data_quality(DF):
	print "################ : New column with Cleaned data in Title & body"
	bs_title_extractor = BsTextExtractor(inputCol="title", outputCol="cleaned_title")
	DF = bs_title_extractor.transform(DF)
	bs_body_extractor = BsTextExtractor(inputCol="body", outputCol="cleaned_body")
	DF = bs_body_extractor.transform(DF)	
	print "################ : trim & lower on body & title"
	#DF = DF.withColumn('cleaned_title_2',ltrim(rtrim(lower(regexp_replace(DF.cleaned_title, '[^\sa-zA-Z0-9]','')))))
	#DF = DF.withColumn('cleaned_body_2',ltrim(rtrim(lower(regexp_replace(DF.cleaned_title, '[^\sa-zA-Z0-9]','')))))
	#print "################ : New column the concat of title & body"
	DF=DF.withColumn('text_all', concat(col("cleaned_title")))
	#DF=DF.withColumn('text_all', concat(col("cleaned_title"), lit(" "), col("cleaned_body")))

	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '[^\sa-zA-Z(),!?\']',' ')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\'s',' \'s')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\'ve',' \'ve')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, 'n\'t',' n\'t')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\'re',' \'re')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\'d',' \'d')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\'ll',' \'ll')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, ',',' , ')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '!',' ! ')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\(',' \( ')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\)',' \) ')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\?',' \?')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '\s{2,}',' ')))
	#DF=DF.withColumn('text_all', lower(regexp_replace(DF.text_all, '','')))
	DF=DF.withColumn('text_all', ltrim(rtrim(lower(DF.text_all))))
	# Drop temporary columns
	columns_to_drop = ['title','body','cleaned_body','cleaned_title']
	FinalcleanDF = DF.drop(*columns_to_drop)
	return FinalcleanDF

# DQ Function on the list of words
def data_quality_words(df):
#	remover = StopWordsRemover(inputCol="words", outputCol="words_all")
#	df = remover.transform(df)
	df = df.withColumn("words_all", df.words)

#	stemmer = SnowballStemmer("english")
#	stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
#	df = df.withColumn("words_stem", stemmer_udf("words_stop"))

#	ngram2 = NGram(n=2, inputCol="words_stem", outputCol="words_all")
#	df = ngram2.transform(df)
#	ngram3 = NGram(n=3, inputCol="words_ngram2", outputCol="words_all")
#	df = ngram3.transform(df)

	# Drop temporary columns
	#columns_to_drop = ['text_all','words','words_stop','words_stem','words_ngram2']
	columns_to_drop = ['text_all','words']
	FinalcleanDF = df.drop(*columns_to_drop)
	return FinalcleanDF

#########################################################
# Classic TF-IDF (with hashing)
#########################################################
# Work on columns texts to create 'text_all'.
dataDF=data_quality(dataDF)

# 1. split text field into words (Tokenize)
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
tokenizer = Tokenizer(inputCol="text_all", outputCol="words")
dataDF = tokenizer.transform(dataDF)
print "################ : New column with tokenized words"

# DataQuality on the list of words (StopWordsRemover & Stemmer) to create 'words_all'
dataDF=data_quality_words(dataDF)
print "################ : New column with clean words"

#cv = CountVectorizer(inputCol="words_all", outputCol="vectors")
#model = cv.fit(df)
print "################ columns :"
print dataDF.printSchema()


# 2. compute term frequencies
dataDF.cache()
hashingTF = HashingTF(inputCol="words_all", outputCol="tf_all")
#,numFeatures=1000
dataDF = hashingTF.transform(dataDF)
print "################ TERM frequencies (TF): Done"
#print dataDF.show(RowCountToShow)
#print "################"

#Convert to RDD and back to DF
#dataDF = dataDF.rdd.toDF()

#3. IDF computation
dataDF.cache()
idf = IDF(inputCol="tf_all", outputCol="tf_idf_all")
print "################ TF_IDF vector: Start fit()"
idfModel = idf.fit(dataDF) #model that contains "dictionary" and IDF values
print "################ TF_IDF vector: Start transform()"
dataDF = idfModel.transform(dataDF)
print "################ TF_IDF vector: Done"
print dataDF.show(RowCountToShow)
#print dataDF.head(RowCountToShow)
print "################"

#########################################################
# Apply same data Prep process (pipeline) on Valid Data
#########################################################
#1. load test data
validDF = sc.textFile('/dssp/datacamp/test.tsv').map(lambda x:x.strip().split('\t')).toDF(['id','title','body'])
print "##### (Valid) ########## dataset loaded  "
# Use same remove HTML 
print "################ Data Quality transformations on Valid data"
validDF=data_quality(validDF)
dataDF.cache()
#2.transform test data
validDF = tokenizer.transform(validDF)
print "##### (Valid) ########## tokenized Title : done."
# DataQuality on the list of words (StopWordsRemover & Stemmer) to create 'words_all'
validDF=data_quality_words(validDF)
print "################ : New column with clean words"
dataDF.cache()
validDF = hashingTF.transform(validDF)
print "##### (Valid) ########## Term Frequencies : done."
validDF = idfModel.transform(validDF)
print "##### (Valid) ########## TF_IDF vector : done."

#########################################################
# Save prepared Data for ML next step
#########################################################
print "################ Saving Parquet Files."
fileName_train=HDFS_base_path+"/train_features_450.parquet"
fileName_valid=HDFS_base_path+"/valid_features_450.parquet"
dataDF.write.format("parquet").mode("overwrite").save("hdfs://"+fileName_train)
print "################ dataDF saved."
validDF.write.format("parquet").mode("overwrite").save("hdfs://"+fileName_valid)
print "################ validDF saved."
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
