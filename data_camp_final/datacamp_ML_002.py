from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from functools import partial
import datetime
import re

startTime=datetime.datetime.now()
RowCountToShow=5
#start "spark session" 
spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext

#########################################################
# Load Parquet files
#########################################################
from pyspark.sql import SQLContext
base_path="/user/christophe.noblanc/datacamp"
fileName_train=base_path+"/train_features_001.parquet"
fileName_valid=base_path+"/valid_features_001.parquet"
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
tfidf_col_name="tf_idf_title"

#B. Train and Evaluate Features with simple logistic regression
#1. Simple evaluation methodology : train and test split
(train,test)=dataDF.rdd.randomSplit([0.8,0.2],seed=42)
#2.initialize model parameters ...we use a simple model here
from pyspark.ml.classification import LogisticRegression

#3. Fit the model
print "################ Start fitting the model : html"
max_iterations=10
col_name="html"
reg_html=LogisticRegression(featuresCol=tfidf_col_name,labelCol=col_name,predictionCol=col_name+"_pred",rawPredictionCol=col_name+"_pred_raw",maxIter=max_iterations)
Model_html = reg_html.fit(train.toDF())
print "################ Start fitting the model : jquery"
col_name="jquery"
reg_jquery=LogisticRegression(featuresCol=tfidf_col_name,labelCol=col_name,predictionCol=col_name+"_pred",rawPredictionCol=col_name+"_pred_raw",maxIter=max_iterations)
Model_jquery = reg_jquery.fit(train.toDF())
print "################ Start fitting the model : css"
col_name="css"
reg_css=LogisticRegression(featuresCol=tfidf_col_name,labelCol=col_name,predictionCol=col_name+"_pred",rawPredictionCol=col_name+"_pred_raw",maxIter=max_iterations)
Model_css = reg_css.fit(train.toDF())
print "################ Start fitting the model : javascript"
col_name="javascript"
reg_javascript=LogisticRegression(featuresCol=tfidf_col_name,labelCol=col_name,predictionCol=col_name+"_pred",rawPredictionCol=col_name+"_pred_raw",maxIter=max_iterations)
Model_javascript = reg_javascript.fit(train.toDF())

#4.Apply model to test data
def apply_model(DF):
    print "######## Model.Transform() for each TAG"
    result_html=Model_html.transform(DF).select(col("id").alias("html_id"),'html_pred','html_pred_raw')
    result_jquery=Model_jquery.transform(DF).select(col("id").alias("jquery_id"),'jquery_pred','jquery_pred_raw')
    result_css=Model_css.transform(DF).select(col("id").alias("css_id"),'css_pred','css_pred_raw')
    result_javascript=Model_javascript.transform(DF).select(col("id").alias("javascript_id"),'javascript_pred','javascript_pred_raw')
    print "######## Merge each TAG predictions"
    result=DF
    result = result.join(result_html, result.id == result_html.html_id,how='left') 
    result = result.join(result_jquery, result.id == result_jquery.jquery_id,how='left') 
    result = result.join(result_css, result.id == result_css.css_id,how='left')
    result = result.join(result_javascript, result.id == result_javascript.javascript_id,how='left')
    return result

print "################ Apply model to train : starting transform()"
result_train=apply_model(train.toDF())
print "################ Apply Model for Train : Done"
result_test=apply_model(test.toDF())
print "################ Merge All TAGS predictions : Done"
print result_test.show(RowCountToShow)

#5. Evaluation of results
from pyspark.ml.evaluation import BinaryClassificationEvaluator
metricName="areaUnderPR"
col_name="html"
evaluator_html = BinaryClassificationEvaluator(rawPredictionCol=col_name+"_pred_raw",labelCol=col_name,metricName=metricName,)
col_name="jquery"
evaluator_jquery = BinaryClassificationEvaluator(rawPredictionCol=col_name+"_pred_raw",labelCol=col_name,metricName=metricName,)
col_name="css"
evaluator_css = BinaryClassificationEvaluator(rawPredictionCol=col_name+"_pred_raw",labelCol=col_name,metricName=metricName,)
col_name="javascript"
evaluator_javascript = BinaryClassificationEvaluator(rawPredictionCol=col_name+"_pred_raw",labelCol=col_name,metricName=metricName,)

#print "################ RESULT of (evaluator on Train) classifier for HTML label"
#eval_train=evaluator_html.evaluate(result_train)
#print eval_train
#print "################ RESULT for : Test"
#eval_test_html=evaluator_html.evaluate(result_test)
#print eval_test_html
#print "################"


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
	predicted=[]
	for tag in [u'javascript_pred', u'css_pred', u'jquery_pred', u'html_pred']:
		if tag in data  and data[tag]==1:
			predicted.append(tag.split('_')[0])
	ret={'id':data['id'],'predicted':predicted}
	newRow = Row(*ret.keys()) 
	newRow = newRow(*ret.values())
	return newRow 

result_valid=apply_model(validDF)
#result_valid=Model_html.transform(validDF)
print "##### (Valid) ########## Apply Model : done."
result_valid=result_valid.rdd.map(predictions_valid).toDF()
print "##### (Valid) ########## Concatenation of Predicted Labels : done."

#4.FINAL EVALUATION on "unknown data"
from dssp_evaluation import tools
#we need a data frame with the predictions and the ids
print "################ Scoring on Valid dataset"
DSSP_score=tools.evaluateDF(sc,result_valid,prediction_col='predicted',id_col='id')

print "---------------------- SUMMARY :"
#print "eval_train (evaluator) =", eval_train
#print "eval_test  (evaluator) =", eval_test
print "F1_Train_score         =", F1_Train_score
print "F1 Test Score          =", F1_Test_score
print "DSSP_score (valid)     =", DSSP_score
print "################ : END."
endTime=datetime.datetime.now()
duration = endTime - startTime 
duration_tuple=divmod(duration.total_seconds(), 60)
print "Total Duration Time (m)= ",duration_tuple[0] 
print "Total Duration Time (s)= ",duration_tuple[1] 
