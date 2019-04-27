#! python3
# coding: utf-8
import json
import random
import re
import sys
import pickle
import numpy as np
import time
from collections import Counter
from keras.utils import to_categorical
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


def tokenize(string):
    token_pattern = re.compile('(?u)\w+')
    tokens = [t.lower() for t in token_pattern.findall(string)]
    return tokens


class RandomLanguageModel:
    def __init__(self):
        self.vocab = set()

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary build:', len(self.vocab), file=sys.stderr)
        return self.vocab

    def score(self, entity, context):
        probability = 1 / len(self.vocab)  # branching factor
        return probability

    def generate(self, context=None):
        prediction = random.sample(self.vocab, k=1)
        return prediction[0]

    def save(self, filename):
        out_voc = sorted(list(self.vocab))
        out_voc_serial = json.dumps(out_voc, ensure_ascii=False, indent=4, sort_keys=True)
        with open(filename, 'w') as out:
            out.write(out_voc_serial)


class FrequencyLanguageModel:
    def __init__(self):
        self.vocab = Counter()

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary build:', len(self.vocab), file=sys.stderr)
        return self.vocab

    def score(self, entity, context):
        probability = self.vocab[entity] / sum(self.vocab.values())  # Proportional to frequency
        return probability

    def generate(self, context=None):
        words = list(self.vocab)
        prediction = random.choices(words, weights=[self.vocab[i] for i in words])
        return prediction[0]

    def save(self, filename):
        out_voc = self.vocab
        with open(filename, 'wb') as out:
            pickle.dump(out_voc, out)


class MarkovLanguageModel:
    def __init__(self, k=2):
        self.vocab = Counter()
        self.trigrams = {}
        self.k = k
        self.corpus_size = 0

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
            for nr, token in enumerate(string):
                self.corpus_size += 1
                if nr < self.k:
                    continue
                prev_context = (string[nr - 2], string[nr - 1])
                if prev_context not in self.trigrams:
                    self.trigrams[prev_context] = Counter()
                self.trigrams[prev_context].update([token])
        print('Vocabulary built:', len(self.vocab), file=sys.stderr)
        print('Trigram model built:', len(self.trigrams), file=sys.stderr)
        return self.vocab, self.trigrams

    def score(self, entity, context=None):
        if context in self.trigrams:
            variants = self.trigrams[context]
            if entity in variants:
                probability = variants[entity] / sum(variants.values())  # Relative to context
                return probability
        probability = self.vocab[entity] / self.corpus_size  # Proportional to frequency
        return probability

    def generate(self, context=None):
        if context in self.trigrams:
            variants = self.trigrams[context]
            words = list(variants)
            prediction = random.choices(words, weights=[variants[i] for i in words])
        else:
            words = list(self.vocab)
            prediction = random.choices(words, weights=[self.vocab[i] for i in words])
        return prediction[0]

    def save(self, filename):
        out_voc = self.vocab
        with open(filename, 'wb') as out:
            pickle.dump(out_voc, out)


class RNNLanguageModel:
    def __init__(self, k=2, lstm=32, emb_dim=10):
        backend.clear_session()
        self.k = k
        self.vocab = Counter()
        self.embed = emb_dim
        self.rnn_size = lstm
        self.word_index = None
        self.model = None
        self.corpus_size = 0

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary built:', len(self.vocab), file=sys.stderr)
        vocab_size = len(self.vocab)
        self.word_index = list(self.vocab)

        sequences = list()
        for string in strings:
            for nr, token in enumerate(string):
                self.corpus_size += 1
                if nr < self.k:
                    continue
                data = [string[nr - 2], string[nr - 1], token]
                encoded = [self.word_index.index(w) for w in data]
                sequences.append(encoded)
        print('Total sequences to train on:', len(sequences), file=sys.stderr)
        sequences = np.array(sequences)

        # Separating input from output:
        contexts, words = sequences[:, :-1], sequences[:, -1]
        words = to_categorical(words, num_classes=vocab_size)

        # Describe the model architecture
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, self.embed, input_length=self.k, name='embeddings'))
        self.model.add(LSTM(self.rnn_size, name='LSTM'))
        self.model.add(Dense(vocab_size, activation='softmax', name='output'))
        print(self.model.summary())

        # Model compilation:
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Training:
        val_split = 0.1
        start = time.time()
        history = self.model.fit(contexts, words, epochs=5, verbose=1, validation_split=val_split)
        end = time.time()
        training_time = int(end - start)
        print('Training took {} seconds'.format(training_time), file=sys.stderr)

        return self.vocab

    def score(self, entity, context=None):
        # model.predict_classes
        raise NotImplemented

    def generate(self, context=None):
        # model.predict_classes
        raise NotImplemented

    def save(self, filename):
        self.model.save(filename)
        print('Model saved to', filename, file=sys.stderr)
