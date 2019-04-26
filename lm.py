#! python3
# coding: utf-8
import argparse
import gzip
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', help="Path to training file")
    parser.add_argument('--model', '-m', default='random',
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
    for line in gzip.open(args.train, 'rt'):
        res = line.strip() + ' ' + EOL
        lines.append(tokenize(res))

    # Training
    model.train(lines)
    model.save(args.save)

    # Testing
    entropies = []
    perplexities = []

    for l in lines:
        for nr, token in enumerate(l):
            if nr < k:
                continue
            probability = model.score(token, context=(l[nr - 2], l[nr - 1]))
            entropy = - 1 * np.log2(probability)
            # print(token, probability, entropy)
            entropies.append(entropy)

    perplexities = [2 ** ent for ent in entropies]

    print('Perplexity: {} over {} trigrams'.format(np.mean(perplexities), len(perplexities)))

    # Generating...
    while True:
        text = input('Type any {} words...\n'.format(k))
        print('==============')
        text = text.lower().split()
        print(' '.join(text), end=' ')
        for i in range(5):
            prediction = model.generate(context=(text[-2], text[-1]))
            if prediction == EOL:
                print('\n')
            else:
                print(prediction, end=' ')
            text = (text[-1], prediction)
        print('\n==============')
