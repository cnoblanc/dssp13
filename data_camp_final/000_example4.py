from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext
dataRDD = sc.textFile('/dssp/datacamp/train.tsv').map(lambda x:x.strip().split('\t'))
dataDF=dataRDD.toDF(['id','title','body','tags'])

#Small examples on various subtopics:
import re
#1. text cleaning
print "TEXT example"
tmp_text=dataRDD.map(lambda x:x[2]).first().encode('UTF-8')
print tmp_text
print "################"

#some posts have html tags in them and a QUICK way to remove them is with regex:
#1.a Remove all the tags
print "replace all tags with space"
print re.sub("<[^>]*>"," ",tmp_text)
print "################"


#1.b Remove all the tags and the text they enclose e.g. <a>LINK</a> : LINK is enclosed in an "a" tag
print "replace all tags and their content with space (greedy - last match)"
print re.sub("<[^>]*>(.*)</[^>]*>"," ",tmp_text)
print "################"

print "replace all tags and their content with space (lazy - first match)"
print re.sub("<[^>]*>(.*?)</[^>]*>"," ",tmp_text)
print "################"


#1.c Remove a specific tags and the text
print "replace specific tag and the content with space"
print re.sub("<code[^>]*>(.*?)</code>"," ",tmp_text)
print "################"


#1.d Remove a specific tags and keep text
print "replace specific tag and keep the content"
print re.sub("<code[^>]*>(.*?)</code>",r"\g<1>",tmp_text)
print "################"