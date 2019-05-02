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
from keras.models import load_model
from smart_open import smart_open


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

    def load(self, filename):
        with smart_open(filename, 'rb') as f:
            self.vocab = json.loads(f)
        print('Model loaded from', filename, file=sys.stderr)


class FrequencyLanguageModel:
    def __init__(self):
        self.vocab = Counter()
        self.corpus_size = 0

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary build:', len(self.vocab), file=sys.stderr)
        self.corpus_size = sum(self.vocab.values())
        return self.vocab

    def score(self, entity, context):
        probability = self.vocab[entity] / self.corpus_size  # Proportional to frequency
        return probability

    def generate(self, context=None):
        words = list(self.vocab)
        prediction = random.choices(words, weights=[self.vocab[i] for i in words])
        return prediction[0]

    def save(self, filename):
        out_voc = self.vocab
        with open(filename, 'wb') as out:
            pickle.dump(out_voc, out)

    def load(self, filename):
        with smart_open(filename, 'rb') as f:
            self.vocab = pickle.load(f)
        self.corpus_size = sum(self.vocab.values())
        print('Model loaded from', filename, file=sys.stderr)


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
                print(entity, probability)
                return probability
        probability = self.vocab[entity] / self.corpus_size  # Proportional to frequency
        print(entity, probability, 'UNKN')
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
        out_dump = [self.vocab, self.trigrams, self.corpus_size]
        with smart_open(filename, 'wb') as out:
            pickle.dump(out_dump, out)

    def load(self, filename):
        with smart_open(filename, 'rb') as f:
            self.vocab, self.trigrams, self.corpus_size = pickle.load(f)
        print('Model loaded from', filename, file=sys.stderr)


class RNNLanguageModel:
    def __init__(self, k=2, lstm=32, emb_dim=10):
        backend.clear_session()
        self.k = k
        self.vocab = Counter()
        self.embed = emb_dim
        self.rnn_size = lstm
        self.word_index = None
        self.inv_index = {}
        self.model = None
        self.corpus_size = 0

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary built:', len(self.vocab), file=sys.stderr)
        vocab_size = len(self.vocab)
        self.word_index = list(self.vocab)
        for nr, word in enumerate(self.word_index):
            self.inv_index[word] = nr

        sequences = list()
        for string in strings:
            for nr, token in enumerate(string):
                self.corpus_size += 1
                if nr < self.k:
                    continue
                data = [string[nr - 2], string[nr - 1], token]
                encoded = [self.inv_index[w] for w in data]
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
        history = self.model.fit(contexts, words, epochs=10, verbose=1, validation_split=val_split)
        end = time.time()
        training_time = int(end - start)
        print('LSTM training took {} seconds'.format(training_time), file=sys.stderr)

        return self.vocab

    def score(self, entity, context=None):
        entity_id = self.inv_index[entity]
        context_ids = np.array([[self.inv_index[w] for w in context]])
        prediction = self.model.predict(context_ids).ravel()
        probability = prediction[entity_id]
        return probability

    def generate(self, context=None):
        context_ids = np.array([[self.inv_index[w] for w in context]])
        prediction = self.model.predict(context_ids).ravel()
        word_id = prediction.argmax()
        word = self.word_index[word_id]
        return word

    def save(self, filename):
        self.model.save(filename)
        out_dump = self.inv_index
        with smart_open(filename.split('.')[0] + '.json.gz', 'wb') as out:
            pickle.dump(out_dump, out)
        print('Model saved to {} and {} (vocabulary)'.format(filename, filename.split('.')[0] +
                                                             '.json'), file=sys.stderr)

    def load(self, filename):
        self.model = load_model(filename)
        voc_file = filename.split('.')[0] + '.json.gz'
        with smart_open(voc_file, 'rb') as f:
            self.inv_index = pickle.load(f)
        self.word_index = sorted(self.inv_index, key=self.inv_index.get)
        print('Model loaded from {} and {}'.format(filename, voc_file), file=sys.stderr)
