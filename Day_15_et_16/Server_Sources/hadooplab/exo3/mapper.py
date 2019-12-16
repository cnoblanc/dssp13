#!/usr/bin/env python

import sys

threshold=0.6

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    # increase counters
    if float(words[2]) >= threshold:
        print '%s\t%s' % (words[0], words[2])
        print '%s\t%s' % (words[1], words[2])

