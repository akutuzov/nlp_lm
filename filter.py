#! python3
# coding: utf-8

import argparse
import string
from smart_open import open

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_corpus', '-c', help="Path to the input corpus", required=True)
    parser.add_argument('--out_corpus', '-o', help="Path to the cleaned corpus", required=True)
    args = parser.parse_args()

    with open(args.out_corpus, 'a') as out:
        for line in open(args.in_corpus, 'r'):
            res = line.strip()
            res = ''.join([char for char in res if char.isalpha() or char in string.punctuation
                           or char == ' ' or char == "'"])
            res = res.strip()
            if res:
                out.write(res + '\n')
