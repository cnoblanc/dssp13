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
        print("Usage: groceries.py <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonGroceries")
    lines = sc.textFile(sys.argv[1])

    counts=lines.map(lambda line: line.split(",")) \
            .map(lambda x,y: (x+","+y,1) if x<y else (y+","+x,1)) \
            .collect()

    #output = counts.take(10)
    for (word, count) in counts:
        print("%s: %i" % (word, count))

    sc.stop()
