#! python3
# coding: utf-8

import argparse
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="Path to training file", required=True)
    parser.add_argument("--embeddings", "-e", help="Path to external embeddings file")
    parser.add_argument(
        "--ngrams",
        "-k",
        help="Number of context words to consider",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="random",
        required=True,
        choices=["random", "freq", "trigram", "rnn"],
    )
    parser.add_argument("--save", "-s", help="Save model to (filename)...")
    args = parser.parse_args()

    EOL = "endofline"  # Special token for line breaks
    k = args.ngrams

    lines = []  # Training corpus
    for line in open(args.train, "r"):
        res = line.strip() + " " + EOL
        lines.append(tokenize(res))

    if args.model == "random":
        model = RandomLanguageModel()
    elif args.model == "freq":
        model = FrequencyLanguageModel()
    elif args.model == "trigram":
        model = MarkovLanguageModel(k=k)
    elif args.model == "rnn":
        if args.embeddings:
            model = RNNLanguageModel(k=k, mincount=2, lstm=128, ext_emb=args.embeddings)
        else:
            model = RNNLanguageModel(k=k, mincount=5)
    else:
        raise ValueError

    print("Training...", len(lines), file=sys.stderr)
    model.train(lines)
    if args.save:
        model.save(args.save)
