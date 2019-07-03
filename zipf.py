#! python3
# coding: utf-8

import argparse
import matplotlib.pyplot as plt
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c', help="Path to the corpus", required=True)
    args = parser.parse_args()

    EOL = 'endofline'

    lines = []  # Corpus to analyze
    for line in open(args.corpus, 'r'):
        res = line.strip() + ' ' + EOL
        lines.append(tokenize(res))

    model = FrequencyLanguageModel()
    print('Counting frequencies...', file=sys.stderr)
    model.train(lines)

    vocabulary = model.vocab
    frequencies = sorted([vocabulary[word] for word in vocabulary], reverse=True)
    words = sorted(vocabulary, key=vocabulary.get, reverse=True)
    ranks = [words.index(w) for w in words]
    plt.plot(ranks, frequencies, 'r')
    plt.title('Zipfian distribution (linear scale)')
    plt.xlabel('Word ranks')
    plt.ylabel('Word frequencies')
    plt.show()

    plt.close()
    plt.clf()
    plt.plot(ranks, frequencies, 'r')
    plt.yscale('log')
    plt.title('Zipfian distribution (log scale)')
    plt.xlabel('Word ranks')
    plt.ylabel('Word frequencies')
    plt.show()



