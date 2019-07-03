#! python3
# coding: utf-8

import sys
import string

for line in sys.stdin:
    res = line.strip()
    res = ''.join([char for char in res if char.isalpha() or char in string.punctuation
                   or char == ' ' or char == "'"])
    res = res.strip()
    if res:
        print(res)
