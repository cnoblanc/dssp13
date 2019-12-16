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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file1> <file2>", file=sys.stderr)
        exit(-1)

    exclude = set(string.punctuation)
    sc = SparkContext(appName="PythonCommonWords")
    lines_f1 = sc.textFile(sys.argv[1],10)
    lines_f2 = sc.textFile(sys.argv[2],100)#.sample(False, fraction=0.1, seed=42)
    
    words_f1=lines_f1.map(lambda x: ''.join(ch for ch in x if ch not in exclude)) \
                  .flatMap(lambda x: x.split(' ')) \
                  .distinct()
    words_f2=lines_f2.map(lambda x: ''.join(ch for ch in x if ch not in exclude)) \
                  .flatMap(lambda x: x.split(' ')) \
                  .distinct()
    common_w=words_f1.intersection(words_f2)

    print("Total numbers of commun words:",common_w.count())

    sc.stop()
