from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.functions import udf, col, count, sum, when, avg, mean, min
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
fileName_dvf=HDFS_base_path+"/DVF_X_df_All_spark.parquet"
sqlContext = SQLContext(sc)
DVF_df = sqlContext.read.parquet("hdfs://" + fileName_dvf)

#DVF_df=loaded_df.where("departement='77'")

# print one line of the dataframe
print "################ Data read : ", fileName_dvf
#print DVF_df.show(RowCountToShow)
print "Source Data Row Count="
print DVF_df.count()
print "################"

#########################################################
# PART Zero
# Column Transformer to create a vectorized feature column
#########################################################
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=['sterr', 'geolong', 'geolat', 'nbpprinc', 'sbati', 'departement' \
        ,'Taux_Evasion_Client', 'Nb_Creation_Commerces' \
        ,'Moyenne_Salaires_Prof_Intermediaire_Horaires', 'Urbanite_Ruralite' \
        ,'Nb_Menages', 'Moyenne_Salaires_Cadre_Horaires', 'Taux_de_Hotel' \
        ,'Taux_de_Mineurs', 'Taux_de_dentistes_Liberaux' \
        ,'Nb_Creation_Enteprises', 'Moyenne_Salaires_Horaires' \
        ,'Taux_de_Occupants_Residence_Principale', 'Taux_de_Homme', 'n_days' \
        ,'quarter', 'department_city_dist'], \
    outputCol="features")
# Columns are :
#'sterr', 'geolong', 'geolat', 'nbpprinc', 'sbati', 'departement',
#'Taux_Evasion_Client', 'Nb_Creation_Commerces',
#'Moyenne_Salaires_Prof_Intermediaire_Horaires', 'Urbanite_Ruralite',
#'Nb_Menages', 'Moyenne_Salaires_Cadre_Horaires', 'Taux_de_Hotel',
#'Taux_de_Mineurs', 'Taux_de_dentistes_Liberaux',
#'Nb_Creation_Enteprises', 'Moyenne_Salaires_Horaires',
#'Taux_de_Occupants_Residence_Principale', 'Taux_de_Homme', 'n_days',
#'quarter', 'department_city_dist', 'valeurfonc'

print "################ Assembler to create 'features' "
DVF_df_prep = assembler.transform(DVF_df)
print("Assembled columns to vector column 'features' DONE.")
#DVF_df_prep.select("features", "valeurfonc").show(truncate=False)

label_Col="valeurfonc"
features_Col="features"

#print "################ print schema of DVF_df_prep "
#DVF_df_prep.printSchema()
#print "################"

#########################################################
# PART I
# Train/Test Split
#########################################################
#1. Simple evaluation methodology : train and test split
(train,test)=DVF_df_prep.rdd.randomSplit([0.8,0.2],seed=42)

#print "################ print schema of DVF_df_prep "
#train.toDF().printSchema()
#print "################"
#########################################################
# PART II
# ML Model
#########################################################
#2.initialize model parameters ...
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

print "################ Define the model : "
max_iterations=10
reg=RandomForestRegressor(featuresCol=features_Col, labelCol=label_Col, predictionCol=label_Col+"_pred"
        , maxDepth=5, minInstancesPerNode=1, numTrees=20
        , maxBins=1024, maxMemoryInMB=256, minInfoGain=0.0
        , seed=42, cacheNodeIds=True, featureSubsetStrategy="auto", checkpointInterval=10
        , impurity="variance", subsamplingRate=1.0)

#print "################ Training the model : Start"
#model = reg.fit(train.toDF())
#print "################ Apply model to train : starting transform()"
#result_train=model.transform(train.toDF())
#print "################ Apply Model for Train : Done"
#result_test=model.transform(test.toDF())
#print "################ Apply Model for Test : Done"

# Create ParamGrid for Cross Validation
print "################ Define the ParamGrid"
paramGrid = (ParamGridBuilder()
             .addGrid(reg.maxDepth, [5,10,20,30]) # Maximum depth of the tree
             .addGrid(reg.minInstancesPerNode, [5,10,20,50,100]) # Minimum number of instances each child must have after split.
             .addGrid(reg.numTrees, [200,500,800,1000]) # Number of trees to train 
             .build())

paramGrid = (ParamGridBuilder()
             .addGrid(reg.maxDepth, [30]) # Maximum depth of the tree
             .addGrid(reg.minInstancesPerNode, [20]) # Minimum number of instances each child must have after split.
             .addGrid(reg.numTrees, [50]) # Number of trees to train 
             .build())

# Define the evaluator
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol=label_Col,predictionCol=label_Col+"_pred",metricName="mae")

# Create 5-fold CrossValidator
print "################ Define the CrossValidator"
cv = CrossValidator(estimator=reg, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

# Run cross-validation, and choose the best set of parameters.
print "################ TRAIN : Start Cross Validation for the model"
model = cv.fit(train.toDF())
print "################  Cross Validation END"
print "################  Best Model Hyper parameters :"
bestModel = model.bestModel
print 'Best Param (maxDepth): ', bestModel._java_obj.getMaxDepth()
print 'Best Param (minInstancesPerNode): ', bestModel._java_obj.getMinInstancesPerNode()
print 'Best Param (numTrees): ', bestModel._java_obj.getNumTrees()

#########################################################
# PART III
# Model Accuracy for Train & Test
#########################################################
print "################ TRAIN : Apply model to train : starting transform()"
result_train=model.transform(train.toDF())
print "################ Apply Model for Train : Done"
print "################ Result of (evaluator on Train)"
eval_train=evaluator.evaluate(result_train)
print "Result : ",eval_train
print "################"

#4.Apply model to test data
print "################ TEST : Apply model to test : starting transform()"
result_test=model.transform(test.toDF())
print "################ Apply Model for Test : Done"
#5. Evaluation of test results
print "################ RESULT of (evaluator on Test)"
eval_test=evaluator.evaluate(result_test)
print "Result : ",eval_test
print "################"

#########################################################
# PART IV
# Model MAPE for Train & Test
#########################################################
def score_mape(x):
	 predicted=set(x[label_Col+"_pred"])
	 correct=set(x[label_Col])
	 return 100*abs(predicted-correct)/abs(correct)

#print "################ print schema of result_train "
#result_train.printSchema()
#print "################"


#print "################ MAPE Score "
#result_train = result_train.withColumn("mape", "valeurfonc" - "_pred")
#MAPE_Train_score=result_train.rdd.map(score_mape).mean()
#print "MAPE_Train_score = ",MAPE_Train_score
#MAPE_Test_score=result_test.rdd.map(score_mape).mean()
#print "MAPE_Test_score = ",MAPE_Test_score
#print "################"

#########################################################
# PART V
# Summary
#########################################################
print "---------------------- SUMMARY :"
#print "eval_train (MulticlassClassificationEvaluator F1) =", eval_train
#print "eval_test  (MulticlassClassificationEvaluator F1) =", eval_test
#print "MAPE_Train_score         =", MAPE_Train_score
print "MAE on Train score       =",eval_train
print "---------------------------"
#print "MAPE_Test_score          =", MAPE_Test_score
print "MAE on Test score        =", eval_test
print "################ : END."
endTime=datetime.datetime.now()
duration = endTime - startTime 
duration_tuple=divmod(duration.total_seconds(), 60)
print "Total Duration Time (m)= ",duration_tuple[0] 
print "Total Duration Time (s)= ",duration_tuple[1] 
print "END."
