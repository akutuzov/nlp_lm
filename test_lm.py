#! python3
# coding: utf-8

import argparse
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", help="Path to testing file", required=True)
    parser.add_argument(
        "--model",
        "-m",
        default="random",
        required=True,
        choices=["random", "freq", "trigram", "rnn"],
    )
    parser.add_argument("--modelfile", "-mf", required=True, help="File name")
    args = parser.parse_args()

    EOL = "endofline"  # Special token for line breaks

    print("Loading test corpus...", file=sys.stderr)
    lines = []
    for line in open(args.test, "r"):
        res = line.strip() + " " + EOL
        lines.append(tokenize(res))

    k = 2

    if args.model == "random":
        model = RandomLanguageModel()
    elif args.model == "freq":
        model = FrequencyLanguageModel()
    elif args.model == "trigram":
        model = MarkovLanguageModel(k=k)
    elif args.model == "rnn":
        model = RNNLanguageModel(k=k, mincount=2)
    else:
        raise ValueError

    model.load(args.modelfile)  # Loading the model from file

    print("Testing...", file=sys.stderr)
    entropies = []
    perplexities = []

    for l in lines:
        for nr, token in enumerate(l):
            if nr < k:
                continue
            # Model prediction:
            context = (l[nr - 2], l[nr - 1])
            probability = model.score(token, context=context)
            entropy = -1 * np.log2(probability)
            entropies.append(entropy)
    perplexities = [2 ** ent for ent in entropies]

    print(
        "Perplexity: {0:.5f} over {1} running trigrams".format(
            np.mean(perplexities), len(perplexities)
        )
    )

    # Generating text...
    while True:
        text = input("Type any {} words...\n".format(k))
        print("==============")
        text = text.lower().split()
        print(" ".join(text), end=" ")
        for i in range(10):
            try:
                prediction = model.generate(context=(text[-2], text[-1]))
            except IndexError:
                print("Error!")
                continue
            if prediction == EOL:
                print("\n")
            else:
                print(prediction, end=" ")
            text = (text[-1], prediction)
        print("\n==============")
