from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from functools import partial


spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext
dataRDD = sc.textFile('/dssp/datacamp/train.tsv').map(lambda x:x.strip().split('\t'))
dataDF=dataRDD.toDF(['id','title','body','tags'])
custom_udf_schema = ArrayType(StringType())
split_string_udf = udf(lambda x:x.split(','),custom_udf_schema)
dataDF = dataDF.withColumn('array_tags',split_string_udf(dataDF.tags))

dataDF = dataDF.drop(dataDF.tags)
possible_tags = sc.broadcast([u'javascript', u'css', u'jquery', u'html'])


def array_string_transform(row,tags):
	data = row.asDict() 
	for tag in tags.value: 
		data[tag] = 0
	for existing_tag in data['array_tags']:
		data[existing_tag] = 1
	newRow = Row(*data.keys()) 
	newRow = newRow(*data.values()) 
	return newRow


mapFunction = partial(array_string_transform,tags=possible_tags)	
dataDF = dataDF.rdd.map(mapFunction).toDF()
#########################################################
#Features EXAMPLES
#########################################################

#Classic TF-IDF (with hashing)
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
#1. split text field into words (Tokenize)
tokenizer = Tokenizer(inputCol="title", outputCol="words_title")
dataDF = tokenizer.transform(dataDF)
print "New column with tokenized text"
print dataDF.head(1)
print "################"	


#2. compute term frequencies
hashingTF = HashingTF(inputCol="words_title", outputCol="tf_title")
dataDF = hashingTF.transform(dataDF)
print "TERM frequencies:"
print dataDF.head(1)
print "################"


#3. IDF computation
idf = IDF(inputCol="tf_title", outputCol="tf_idf_title")
idfModel = idf.fit(dataDF) #model that contains "dictionary" and IDF values
dataDF = idfModel.transform(dataDF)
print "TF_IDF vector:"
print dataDF.head(1)
print "################"



#B. Train and Evaluate Features with simple logistic regression ON ONE LABEL
#1. Simple evaluation methodology : train and test split
(train,test)=dataDF.rdd.randomSplit([0.8,0.2])
#2.initialize model parameters ...we use a simple model here
from pyspark.ml.classification import LogisticRegression

logistic=LogisticRegression(featuresCol="tf_idf_title",labelCol="html",predictionCol='html_pred',rawPredictionCol="html_pred_raw",maxIter=10)

#3. Fit the model
lrModel = logistic.fit(train.toDF())

#4.Apply model to test data
result=lrModel.transform(test.toDF())

#5. Evaluation of results
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="html_pred_raw",labelCol='html',metricName="areaUnderPR",)

print "RESULT of classifier for HTML label"
print evaluator.evaluate(result)
print "################"



#DO THE SAME FOR ALL THE LABELS 

#C. Multi-label evaluation
#ASSUMING all labels predicted evaluate the multi-label task

#1. example of how to transform the prediction columns into one column of labels
#return new rows with original list of labels in one column and the predicted ones in the other
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

to_evaluate=result.rdd.map(predictions)


#2.We define here a metric to evaluate all predicted tags simultaneously
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
	 
	 
print to_evaluate.map(F1_multilabel).mean()
print "################"