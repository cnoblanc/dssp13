#!/usr/bin/env python

import sys,string
from nltk.corpus import stopwords

exclude = set(string.punctuation)
stop_w=stopwords.words('english')
stop_w.append(["abcd"])

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line_wo_ponctuation=''.join(ch for ch in line if ch not in exclude)
    # split the line into words
    words = line_wo_ponctuation.split()
    # increase counters
    for i in range(len(words)-1):
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        word=words[i]
        if (word[0] in ['a','b','c','d']):
            if (word not in stop_w): 
                print '%s %s\t%s' % (word,words[i+1], 1)
