#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    nodes = line.split()

    # get the first node
    v1 = nodes[0]
    # get the second node
    v2 = nodes[1]

    # print key,value pair
    print '%s\t%s' % (v1, 1)
