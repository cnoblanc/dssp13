#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import sys
from operator import add
from pyspark import SparkContext

import string
from nltk.corpus import stopwords


if __name__ == "__main__":
    exclude = set(string.punctuation)
    stop_w=stopwords.words('english')
    stop_w.append(["abcd"])

    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonWordCount")
    lines = sc.textFile(sys.argv[1])
    counts = lines.flatMap(lambda x: ''.join(ch for ch in x if ch not in exclude).split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add) 
    sorted_counts =counts.map(lambda (x,y):(y,x)) \
            .sortByKey(ascending=False)\
            .map(lambda(x,y): (y,x))
    output = sorted_counts.take(100)
    for (word, count) in output:
        print("%s: %i" % (word, count))

    sc.stop()
