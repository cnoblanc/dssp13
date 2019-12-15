from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import DoubleType,StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.ml.feature import OneHotEncoder, StringIndexer,IndexToString, VectorAssembler
from functools import partial
import datetime
import re

############################
# General Parameters #######
############################
HDFS_base_path="/dssp/shared/noblanc"
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
fileName_train=HDFS_base_path+"/train_features_600.parquet"
fileName_valid=HDFS_base_path+"/valid_features_600.parquet"
sqlContext = SQLContext(sc)
print "################ Start loading Train DataFrame"
dataDF = sqlContext.read.parquet("hdfs://" + fileName_train)
print "################ Start loading Valid DataFrame"
validDF = sqlContext.read.parquet("hdfs://" + fileName_valid)
print "################ Loading DataFrame End"

# print one line of the dataframe
print "################ Train Data read"
print dataDF.show(RowCountToShow)
#print "Source Data (Train&Test) Row Count=",dataDF.count()
print "############### Valid Data read"
print validDF.show(RowCountToShow)
#print "Validation Row Count=",validDF.count()

#########################################################
# PART III
# Train/Test Split and Model Prediction score on Test
#########################################################
possible_tags = [u'javascript', u'css', u'jquery', u'html']
tfidf_col_name="tf_idf_all"
col_name="label"

# Create the consolidated Target : 
label_stringIndexer = StringIndexer(inputCol = "tags_target", outputCol = col_name).fit(dataDF)
dataDF = label_stringIndexer.transform(dataDF)
print "############### Created Target of multi-Classes Tags"
#print dataDF.show(RowCountToShow)
#print label_stringIndexer.labels

print "############### Select only needed columns in dataDF for multi-Classes Tags"
dataDF=dataDF.select('id',tfidf_col_name,'array_tags','tags_target',col_name)

#B. Train and Evaluate Features with simple logistic regression
#1. Simple evaluation methodology : train and test split
(train,test)=dataDF.rdd.randomSplit([0.8,0.2],seed=42)

#2.initialize model parameters ...we use a simple model here
from pyspark.ml.classification import MultilayerPerceptronClassifier

#3. Fit the model
print "################ Start fitting the model : tags_target"
max_iterations=10

# create the trainer and set its parameters
layers = [262144,1000,64,16] # 1st layer : number of neurons should be the number of features in the tf-idf
# last layer : 16 = the number of classes to predict (2^4 in our case, of the StringIndexer).
trainer = MultilayerPerceptronClassifier(featuresCol=tfidf_col_name,labelCol=col_name,predictionCol=col_name+"_pred" \
	,maxIter=500, layers=layers, blockSize=128, seed=42)

model = trainer.fit(train.toDF())
print "################  fitting the model END"

#4.Apply model to test data
labelReversePred = IndexToString(inputCol = col_name+"_pred", outputCol="encoded_pred",labels=label_stringIndexer.labels)
def apply_model(DF):
	print "######## Model.Transform() to get predictions"
	result=model.transform(DF)
	print "################ columns of the dataframe:"
	print result.printSchema()
	print "######## Prediction plit into several columns"
	result=labelReversePred.transform(result)
	return result

print "################ Apply model to train : starting transform()"
result_train=apply_model(train.toDF())
print "################ Apply Model for Train : Done"
result_test=apply_model(test.toDF())
print "################ Apply Model for Test : Done"
print result_test.show(RowCountToShow)

#5. Evaluation of results
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
metricName="f1"
col_name="label"
#evaluator = MulticlassClassificationEvaluator(labelCol=col_name,predictionCol=col_name+"_pred",metricName=metricName,)

print "################ RESULT of (evaluator on Train) classifier for label"
#eval_train=evaluator.evaluate(result_train)
#print eval_train
print "################ RESULT for : Test"
#eval_test=evaluator.evaluate(result_test)
#print eval_test
print "################"


# ##########################
# C. Multi-label evaluation
# ASSUMING all labels predicted evaluate the multi-label task

# 1. example of how to transform the prediction columns into one column of labels
# return new rows with original list of labels in one column and the predicted ones in the other
def predictions(row):
	data = row.asDict()
	labels=data['array_tags']
	encoded_pred=data['encoded_pred']
	predicted=[]
	# Decode the encoded prediction
	for i, char in enumerate(encoded_pred):
		if char=='1':
			# Add the associated TAG in the predicted array
			predicted.append(possible_tags[i])
	ret={'id':data['id'],'labels':labels,'predicted':predicted}
	newRow = Row(*ret.keys()) 
	newRow = newRow(*ret.values())
	return newRow
print "################ Concatenation of Predicted Labels"
result_train_predlabels=result_train.rdd.map(predictions)
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

print "################ Train with Predictions "
#print result_train_predlabels.toDF().show(RowCountToShow)

print "################ F1 Score "
F1_Train_score=result_train_predlabels.map(F1_multilabel).mean()
print "F1_Train_score = ",F1_Train_score
F1_Test_score=result_test_predlabels.map(F1_multilabel).mean()
print "F1_Test_score = ",F1_Test_score
print "################"

#########################################################
# PART IV
# Submit on Validation data
#########################################################
#3.Apply model to test data
def predictions_valid(row):
	data = row.asDict()
	encoded_pred=data['encoded_pred']
	predicted=[]

	# Decode the encoded prediction
	for i, char in enumerate(encoded_pred):
		if char=='1':
			# Add the associated TAG in the predicted array
			predicted.append(possible_tags[i])

	ret={'id':data['id'],'predicted':predicted}
	newRow = Row(*ret.keys()) 
	newRow = newRow(*ret.values())
	return newRow 

print "############### Select only needed columns in dataDF for multi-Classes Tags"
validDF=validDF.select('id',tfidf_col_name)

result_valid=apply_model(validDF)
#result_valid=Model.transform(validDF)
print "##### (Valid) ########## Apply Model : done."
result_valid=result_valid.rdd.map(predictions_valid).toDF()
print "##### (Valid) ########## Concatenation of Predicted Labels : done."

#4.FINAL EVALUATION on "unknown data"
from dssp_evaluation import tools
#we need a data frame with the predictions and the ids
print "################ Scoring on Valid dataset"
DSSP_score=tools.evaluateDF(sc,result_valid,prediction_col='predicted',id_col='id')
print "---------------------- SUMMARY :"
#print "eval_train (MulticlassClassificationEvaluator F1) =", eval_train
#print "eval_test  (MulticlassClassificationEvaluator F1) =", eval_test
print "F1_Train_score         =", F1_Train_score
print "F1 Test Score          =", F1_Test_score
print "DSSP_score (valid)     =", DSSP_score
print "################ : END."
endTime=datetime.datetime.now()
duration = endTime - startTime 
duration_tuple=divmod(duration.total_seconds(), 60)
print "Total Duration Time (m)= ",duration_tuple[0] 
print "Total Duration Time (s)= ",duration_tuple[1] 
