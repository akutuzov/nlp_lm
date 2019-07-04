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
from keras.callbacks import TensorBoard, EarlyStopping
from smart_open import open


def tokenize(string):
    token_pattern = re.compile('(?u)\w+')
    tokens = [t.lower() for t in token_pattern.findall(string)]
    return tokens


class RandomLanguageModel:
    """
    This model guesses the next word randomly (from the corpus-based dictionary)
    """

    def __init__(self):
        self.vocab = set()
        self.uniform_probability = None

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary build:', len(self.vocab), file=sys.stderr)
        return self.vocab

    def score(self, entity, context):
        # We ignore context here
        if entity in self.vocab:
            probability = self.uniform_probability  # branching factor

        # Decreased probability for out-of-vocabulary words:
        else:
            probability = 0.99 / len(self.vocab)
        return probability

    def generate(self, context=None):
        # We ignore context here
        prediction = random.sample(self.vocab, k=1)
        return prediction[0]

    def save(self, filename):
        out_voc = sorted(list(self.vocab))
        out_voc_serial = json.dumps(out_voc, ensure_ascii=False, indent=4, sort_keys=True)
        with open(filename, 'w') as out:
            out.write(out_voc_serial)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.vocab = set(json.loads(f.read()))
            self.uniform_probability = 1 / len(self.vocab)
        print('Model loaded from', filename, file=sys.stderr)


class FrequencyLanguageModel:
    """
    This model predicts the next word randomly from the corpus-based dictionary,
    with words' probabilities proportional to their corpus frequencies.
    """

    def __init__(self):
        self.vocab = Counter()
        self.corpus_size = 0
        self.probs = None

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        print('Vocabulary build:', len(self.vocab), file=sys.stderr)
        self.corpus_size = sum(self.vocab.values())
        # Word probabilities:
        self.probs = {word: self.vocab[word] / self.corpus_size for word in self.vocab}
        return self.vocab

    def score(self, entity, context):
        # We ignore context here
        if entity in self.probs:
            probability = self.probs[entity]  # Proportional to frequency
        # Decreased probability for out-of-vocabulary words:
        else:
            probability = 0.99 / self.corpus_size
        return probability

    def generate(self, context=None):
        # We ignore context here
        words = list(self.probs)
        probabilities = [self.probs[w] for w in words]
        prediction = np.random.choice(words, p=probabilities)
        return prediction

    def save(self, filename):
        out_voc = [self.probs, self.corpus_size]
        with open(filename, 'wb') as out:
            pickle.dump(out_voc, out)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.probs, self.corpus_size = pickle.load(f)
        print('Model loaded from', filename, file=sys.stderr)


class MarkovLanguageModel:
    """
    This model predicts the next word based on tri-gram statistics from the corpus
    (so it finally uses the context).
    """

    def __init__(self, k=2):
        self.vocab = Counter()
        self.trigrams = {}
        self.k = k
        self.corpus_size = 0
        self.probs = None

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
        # Word probabilities:
        self.probs = {word: self.vocab[word] / self.corpus_size for word in self.vocab}
        print('Trigram model built:', len(self.trigrams), 'trigrams', file=sys.stderr)
        return self.vocab, self.trigrams, self.probs

    def score(self, entity, context=None):
        if context in self.trigrams:
            variants = self.trigrams[context]
            if entity in variants:
                probability = variants[entity] / sum(variants.values())  # Relative to context
                # print(entity, probability)
                return probability
        if entity in self.probs:
            probability = self.probs[entity]  # Proportional to frequency
        # Decreased probability for out-of-vocabulary words:
        else:
            probability = 0.99 / self.corpus_size
            # print(entity, probability, 'UNKN')
        return probability

    def generate(self, context=None):
        if context in self.trigrams:
            variants = self.trigrams[context]
            bigram_freq = sum(variants.values())
            words = list(variants)
            probabilities = [variants[word] / bigram_freq for word in words]
            prediction = np.random.choice(words, p=probabilities)
        else:
            words = list(self.probs)
            probabilities = [self.probs[w] for w in words]
            prediction = np.random.choice(words, p=probabilities)
        return prediction

    def save(self, filename):
        print('Saving the model to', filename, file=sys.stderr)
        out_dump = [self.probs, self.trigrams, self.corpus_size]
        with open(filename, 'wb') as out:
            pickle.dump(out_dump, out)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.probs, self.trigrams, self.corpus_size = pickle.load(f)
        print('Model loaded from', filename, file=sys.stderr)


class RNNLanguageModel:
    """
    This model trains a simple LSTM on the training corpus and casts the next word prediction
    as a classification task (choose from all the words in the vocabulary).
    """

    def __init__(self, k=2, lstm=16, emb_dim=5, batch_size=16):
        backend.clear_session()
        self.k = k
        self.vocab = Counter()
        self.embed = emb_dim
        self.rnn_size = lstm
        self.word_index = None
        self.inv_index = {}
        self.model = None
        self.corpus_size = 0
        self.batch_size = batch_size

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
        # contexts, words = sequences[:, :-1], sequences[:, -1]
        # words = to_categorical(words, num_classes=vocab_size)

        # Describe the model architecture
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, self.embed, input_length=self.k, name='embeddings'))
        self.model.add(LSTM(self.rnn_size, name='LSTM'))
        self.model.add(Dense(vocab_size, activation='softmax', name='output'))
        print(self.model.summary(), file=sys.stderr)

        # Model compilation:
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        loss_plot = TensorBoard(log_dir='logs/LSTM')
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=2, verbose=1)

        # We are using the last lines of the corpus as a validation set:
        val_split = 0.05
        train_data_len = int(len(sequences) * (1 - val_split))
        val_data_len = int(len(sequences) * val_split)

        train_data = sequences[:train_data_len, :]
        val_data = sequences[-val_data_len:, :]

        print('Training on:', train_data_len, file=sys.stderr)
        print('Validating on:', val_data_len, file=sys.stderr)

        val_contexts, val_words = val_data[:, :-1], val_data[:, -1]
        val_words = to_categorical(val_words, num_classes=vocab_size)
        val_data = val_contexts, val_words

        # How many times per epoch we will ask the batch generator to yield a batch?
        steps = train_data_len / self.batch_size
        print('Steps:', int(steps), file=sys.stderr)

        # Training:
        start = time.time()
        history = self.model.fit_generator(self.batch_generator(
            train_data, vocab_size, self.batch_size), steps_per_epoch=steps, epochs=10,
            verbose=1, callbacks=[loss_plot, earlystopping], validation_data=val_data)
        end = time.time()
        training_time = int(end - start)
        print('LSTM training took {} seconds'.format(training_time), file=sys.stderr)

        self.model.corpus_size = self.corpus_size

        return self.vocab

    def batch_generator(self, data, vocab_size, batch_size):
        """
        Generates training batches
        """

        while True:
            # Separating input from output:
            contexts = np.empty((batch_size, self.k), dtype=int)
            words = np.empty((batch_size, vocab_size), dtype=int)
            inst_counter = 0
            for row in data:
                context, word = row[:-1], row[-1]
                word = to_categorical(word, num_classes=vocab_size)
                contexts[inst_counter] = context
                words[inst_counter] = word
                inst_counter += 1
                if inst_counter == batch_size:
                    yield (contexts, words)
                    contexts = np.empty((batch_size, self.k))
                    words = np.empty((batch_size, vocab_size))
                    inst_counter = 0

    def score(self, entity, context=None):
        if entity in self.inv_index and all([word in self.inv_index for word in context]):
            entity_id = self.inv_index[entity]
            context_ids = np.array([[self.inv_index[w] for w in context]])
        # Decreased probability for out-of-vocabulary words:
        else:
            return 0.99 / self.model.corpus_size

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
        with open(filename.split('.')[0] + '.pickle.gz', 'wb') as out:
            pickle.dump(out_dump, out)
        print('Model saved to {} and {} (vocabulary)'.format(filename, filename.split('.')[0] +
                                                             '.pickle.gz'), file=sys.stderr)

    def load(self, filename):
        self.model = load_model(filename)
        voc_file = filename.split('.')[0] + '.pickle.gz'
        with open(voc_file, 'rb') as f:
            self.inv_index = pickle.load(f)
        self.word_index = sorted(self.inv_index, key=self.inv_index.get)
        print('Model loaded from {} and {}'.format(filename, voc_file), file=sys.stderr)
