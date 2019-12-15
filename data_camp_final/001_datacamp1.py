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
appName='christophe'

startTime=datetime.datetime.now()
RowCountToShow=5
#start "spark session" 
spark = SparkSession.builder.appName(appName).getOrCreate()
sc = spark.sparkContext
#A.load tsv into a data frame:
#1.read the raw text and split it to fields (the text file does not contain a header)
dataRDD = sc.textFile('/dssp/datacamp/train.tsv').map(lambda x:x.strip().split('\t'))
#2. convert the rdd to a DataFrame and names to columns 
dataDF=dataRDD.toDF(['id','title','body','tags'])
# print one line of the dataframe
print "################ DATA to DF"
print dataDF.show(RowCountToShow)
print "Source Data (Train&Test) Row Count=",dataDF.count()
#print dataDF.head(RowCountToShow)
print "################"

#########################################################
# PART I
# Extract the Tags into an array and then in OneHotEncoding
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
#print dataDF.head(RowCountToShow)
#print "###############"
# print dataDF.show()

#########################################################
# PART II
# Features EXAMPLES
#########################################################
#Classic TF-IDF (with hashing)
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# 1. split text field into words (Tokenize)
tokenizer = Tokenizer(inputCol="title", outputCol="words_title")
dataDF = tokenizer.transform(dataDF)
print "################ : New column with tokenized Title"
#print dataDF.show(RowCountToShow)
#print dataDF.head(RowCountToShow)
#print "################"	

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
print "Validation Row Count=",validDF.count()

#########################################################
# PART III
# Train/Test Split and Model Prediction score on Test
#########################################################
#B. Train and Evaluate Features with simple logistic regression ON ONE LABEL
#1. Simple evaluation methodology : train and test split
(train,test)=dataDF.rdd.randomSplit([0.8,0.2],seed=42)
#2.initialize model parameters ...we use a simple model here
from pyspark.ml.classification import LogisticRegression

logistic=LogisticRegression(featuresCol="tf_idf_title",labelCol="html",predictionCol='html_pred',rawPredictionCol="html_pred_raw",maxIter=10)
#3. Fit the model
print "################ Start fitting the model"
lrModel = logistic.fit(train.toDF())

#4.Apply model to test data
print "################ Apply model to train : starting transform()"
result_train=lrModel.transform(train.toDF())
print "################ Apply Model for Train : Done"
result_test=lrModel.transform(test.toDF())
print "################ Apply Model for Test : Done"

#5. Evaluation of results
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="html_pred_raw",labelCol='html',metricName="areaUnderPR",)

print "################ RESULT of (evaluator on Train) classifier for HTML label"
eval_train=evaluator.evaluate(result_train)
print eval_train
print "################ RESULT for : Test"
eval_test=evaluator.evaluate(result_test)
print eval_test
print "################"

# TODO : DO THE SAME FOR ALL THE LABELS 

# ##########################
# C. Multi-label evaluation
# ASSUMING all labels predicted evaluate the multi-label task

# 1. example of how to transform the prediction columns into one column of labels
# return new rows with original list of labels in one column and the predicted ones in the other
def predictions(row):
	data = row.asDict()
	labels=data['array_tags']
	predicted=[]
	for tag in [u'javascript_pred', u'css_pred', u'jquery_pred', u'html_pred']:
		if tag in data  and data[tag]==1:
			predicted.append(tag.split('_')[0])
	ret={'id':data['id'],'labels':labels,'predicted':predicted}
	newRow = Row(*ret.keys()) 
	newRow = newRow(*ret.values())
	return newRow
print "################ Concatenation of Predicted Labels"
result_test_predlabels=result_test.rdd.map(predictions)
print "################ Concatenation of Predicted Labels : Done."

# 2.We define here a metric to evaluate all predicted tags simultaneously
#we are going to use the F1-score (harmonic mean) of precision and recall

#This Function is mapped on an RDD of a dataframe to compute per element
# the ratio of the predicted correct labels over the sum of total number of predicted labels AND total number of actual labels 
#The final result would be the average over all element of the RDD: IT should be computed after the mapping of this function
#The dataframe should have a column with all the labels called labels and
#a columns with the correctly predicted labels called predicted
def F1_multilabel(x):
	 predicted=set(x['predicted'])
	 correct=set(x['labels'])
	 predicted_correct=len(predicted.intersection(correct))
	 return 2*predicted_correct/float(len(correct)+len(predicted))
	 
print "################ F1 Score "
F1_Test_score=result_test_predlabels.map(F1_multilabel).mean()
print("F1_Test_score = ",F1_Test_score)
print "################"


#########################################################
# PART IV
# Submit on Validation data
#########################################################

#3.Apply model to test data
def predictions_valid(row):
	data = row.asDict()
	predicted=[]
	for tag in [u'javascript_pred', u'css_pred', u'jquery_pred', u'html_pred']:
		if tag in data  and data[tag]==1:
			predicted.append(tag.split('_')[0])
	ret={'id':data['id'],'predicted':predicted}
	newRow = Row(*ret.keys()) 
	newRow = newRow(*ret.values())
	return newRow

result_valid=lrModel.transform(validDF)
print "##### (Valid) ########## Apply Model : done."
result_valid=result_valid.rdd.map(predictions_valid).toDF()
print "##### (Valid) ########## Concatenation of Predicted Labels : done."

#4.FINAL EVALUATION on "unknown data"
from dssp_evaluation import tools
#we need a data frame with the predictions and the ids
print "################ Scoring on Valid dataset"
DSSP_score=tools.evaluateDF(sc,result_valid,prediction_col='predicted',id_col='id')
print "---------------------- SUMMARY :"
print "eval_train (evaluator) =", eval_train
print "eval_test  (evaluator) =", eval_test
print "F1 Test Score          =", F1_Test_score
print "DSSP_score (valid)     =", DSSP_score
print "################ : END."
endTime=datetime.datetime.now()
duration = endTime - startTime 
duration_tuple=divmod(duration.total_seconds(), 60)
print "Total Duration Time (m)= ",duration_tuple[0] 
print "Total Duration Time (s)= ",duration_tuple[1] 
