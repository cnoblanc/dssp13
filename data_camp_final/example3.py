from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf
from functools import partial


spark = SparkSession.builder.appName('datacamp_christophe').getOrCreate()
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


from pyspark.ml.feature import HashingTF, IDF, Tokenizer

tokenizer = Tokenizer(inputCol="title", outputCol="words_title")
dataDF = tokenizer.transform(dataDF)
hashingTF = HashingTF(inputCol="words_title", outputCol="tf_title")
dataDF = hashingTF.transform(dataDF)

idf = IDF(inputCol="tf_title", outputCol="tf_idf_title")
idfModel = idf.fit(dataDF) 
dataDF = idfModel.transform(dataDF)


from pyspark.ml.classification import LogisticRegression

logistic=LogisticRegression(featuresCol="tf_idf_title",labelCol="html",predictionCol='html_pred',rawPredictionCol="html_pred_raw",maxIter=10)
lrModel = logistic.fit(dataDF)

#1. load test data
testDF = sc.textFile('/dssp/datacamp/test.tsv').map(lambda x:x.strip().split('\t')).toDF(['id','title','body'])

#2.transform test data
testDF = tokenizer.transform(testDF)
testDF = hashingTF.transform(testDF)
testDF = idfModel.transform(testDF)

#3.Apply model to test data
result=lrModel.transform(testDF)
def predictions(row):
	data = row.asDict()
	predicted=[]
	for tag in [u'javascript_pred', u'css_pred', u'jquery_pred', u'html_pred']:
		if tag in data  and data[tag]==1:
			predicted.append(tag.split('_')[0])
	ret={'id':data['id'],'predicted':predicted}
	newRow = Row(*ret.keys()) 
	newRow = newRow(*ret.values())
	return newRow
result=result.rdd.map(predictions).toDF()


#4.FINAL EVALUATION on "unknown data"
from dssp_evaluation import tools
#we need a data frame with the predictions and the ids
print tools.evaluateDF(sc,result,prediction_col='predicted',id_col='id')