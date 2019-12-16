'''
Creates an array of random numbers, parallelizes the array and then
sorts the data.
'''

from __future__ import print_function

import numpy.random as random
import sys

from pyspark import SparkContext


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort size", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonSort")
    size = int(sys.argv[1])
    
    data = [int(1000*random.random()) for i in xrange(size)]

    '''
    Parallelize the data into a number of partitions
    '''
    dataRDD = sc.parallelize(data,10)

    '''
    Sort the data and collect
    '''
    sortedData = dataRDD.map(lambda x: (int(x), 1)).sortByKey(lambda x: x).collect()

    '''
    Display the sorted data
    '''
    for num in sortedData:
        print(num)
    

    sc.stop()
