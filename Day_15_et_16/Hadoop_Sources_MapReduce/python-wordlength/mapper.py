#!/usr/bin/env python
 
 
import sys
 
 
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        if (word[0]>='A') and (word[0] <= 'Z'):
            print '%s\t%s' % (word[0].upper(), len(word))

