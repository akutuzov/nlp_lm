#! python3
# coding: utf-8

import argparse
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', help="Path to training file", required=True)
    parser.add_argument('--model', '-m', default='random', required=True,
                        choices=['random', 'freq', 'trigram', 'rnn'])
    parser.add_argument('--save', '-s', help='Save model to...')
    args = parser.parse_args()

    EOL = 'endofline'
    k = 2

    if args.model == 'random':
        model = RandomLanguageModel()
    elif args.model == 'freq':
        model = FrequencyLanguageModel()
    elif args.model == 'trigram':
        model = MarkovLanguageModel(k=k)
    elif args.model == 'rnn':
        model = RNNLanguageModel(k=k)
    else:
        raise ValueError

    lines = []  # Training corpus
    for line in smart_open(args.train, 'r'):
        res = line.strip() + ' ' + EOL
        lines.append(tokenize(res))

    print('Training...', file=sys.stderr)
    model.train(lines)
    if args.save:
        model.save(args.save)
