#!/usr/bin/env python

import sys,string
from nltk.corpus import stopwords

exclude = set(string.punctuation)

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line_wo_ponctuation=''.join(ch for ch in line if ch not in exclude)
    # split the line into words
    words = line_wo_ponctuation.split()
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        if (word[0] in ['a','b','c','d']):
            if (word not in stopwords.words('english')): 
                print '%s\t%s' % (word, 1)
