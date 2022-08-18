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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from smart_open import open
import logging
import gensim


def tokenize(string):
    token_pattern = re.compile("(?u)\w+")
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
        print("Vocabulary build:", len(self.vocab), file=sys.stderr)
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
        out_voc_serial = json.dumps(
            out_voc, ensure_ascii=False, indent=4, sort_keys=True
        )
        with open(filename, "w") as out:
            out.write(out_voc_serial)

    def load(self, filename):
        with open(filename, "r") as f:
            self.vocab = set(json.loads(f.read()))
            self.uniform_probability = 1 / len(self.vocab)
        print("Model loaded from", filename, file=sys.stderr)


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
        print("Vocabulary build:", len(self.vocab), file=sys.stderr)
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
        with open(filename, "wb") as out:
            pickle.dump(out_voc, out)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.probs, self.corpus_size = pickle.load(f)
        print("Model loaded from", filename, file=sys.stderr)


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
        print("Vocabulary built:", len(self.vocab), file=sys.stderr)
        # Word probabilities:
        self.probs = {word: self.vocab[word] / self.corpus_size for word in self.vocab}
        print("Trigram model built:", len(self.trigrams), "trigrams", file=sys.stderr)
        return self.vocab, self.trigrams, self.probs

    def score(self, entity, context=None):
        if context in self.trigrams:
            variants = self.trigrams[context]
            if entity in variants:
                probability = variants[entity] / sum(
                    variants.values()
                )  # Relative to context
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
        print("Saving the model to", filename, file=sys.stderr)
        out_dump = [self.probs, self.trigrams, self.corpus_size]
        with open(filename, "wb") as out:
            pickle.dump(out_dump, out)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.probs, self.trigrams, self.corpus_size = pickle.load(f)
        print("Model loaded from", filename, file=sys.stderr)


class RNNLanguageModel:
    """
    This model trains a simple LSTM on the training corpus and casts the next word prediction
    as a classification task (choose from all the words in the vocabulary).
    """

    def __init__(
        self, k=2, lstm=32, emb_dim=32, batch_size=8, ext_emb=None, mincount=None
    ):
        backend.clear_session()
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        self.k = k
        self.vocab = Counter()
        self.embed = emb_dim
        self.rnn_size = lstm
        self.word_index = None
        self.inv_index = {}
        self.model = None
        self.corpus_size = 0
        self.batch_size = batch_size
        self.mincount = mincount
        self.ext_emb = ext_emb
        self.ext_vectors = None
        self.ext_vocab = None
        if self.ext_emb:
            external_embeddings = None
            if ext_emb.endswith(".model"):
                external_embeddings = gensim.models.KeyedVectors.load(ext_emb)
            elif ext_emb.endswith("bin") or ext_emb.endswith("bin.gz"):
                external_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                    ext_emb, binary=True
                )
            else:
                print(
                    "Wrong file format for the external embedding file!",
                    file=sys.stderr,
                )
                print(
                    "Please use either Gensim models or binary word2vec models",
                    file=sys.stderr,
                )
                exit()
            self.ext_vectors = external_embeddings.vectors
            self.ext_vocab = external_embeddings.index2entity
            self.ext_word_index = {}
            for nr, word in enumerate(self.ext_vocab):
                self.ext_word_index[word] = nr

    def train(self, strings):
        for string in strings:
            self.vocab.update(string)
        if self.mincount:
            self.vocab = {
                word: self.vocab[word]
                for word in self.vocab
                if self.vocab[word] >= self.mincount
            }
        print("Vocabulary built:", len(self.vocab), file=sys.stderr)
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
                if self.ext_emb:
                    data = [string[nr - 2], string[nr - 1], token]
                    if (
                        all([word in self.ext_word_index for word in data[:2]])
                        and token in self.inv_index
                    ):
                        encoded_context = [self.ext_word_index[w] for w in data[:2]]
                        encoded_token = [self.inv_index[token]]
                        encoded = encoded_context + encoded_token
                        sequences.append(encoded)
                else:
                    data = [string[nr - 2], string[nr - 1], token]
                    if all([word in self.inv_index for word in data]):
                        encoded = [self.inv_index[w] for w in data]
                        sequences.append(encoded)
        print("Total sequences to train on:", len(sequences), file=sys.stderr)
        sequences = np.array(sequences)

        # Describe the model architecture
        self.model = Sequential()
        if self.ext_emb:
            weights = self.ext_vectors
            # Take the weights from the Gensim model, freeze the layer
            self.model.add(
                Embedding(
                    weights.shape[0],
                    weights.shape[1],
                    weights=[weights],
                    input_length=self.k,
                    trainable=False,
                    name="embeddings",
                )
            )
        else:
            self.model.add(
                Embedding(
                    vocab_size, self.embed, input_length=self.k, name="embeddings"
                )
            )
        self.model.add(LSTM(self.rnn_size, name="LSTM"))
        self.model.add(Dense(vocab_size, activation="softmax", name="output"))
        print(self.model.summary(), file=sys.stderr)

        # Model compilation:
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        loss_plot = TensorBoard(log_dir="logs/LSTM")
        earlystopping = EarlyStopping(
            monitor="val_accuracy", min_delta=0.001, patience=3, verbose=1
        )

        # We are using the last lines of the corpus as a validation set:
        val_split = 0.005
        train_data_len = int(len(sequences) * (1 - val_split))
        val_data_len = int(len(sequences) * val_split)

        train_data = sequences[:train_data_len, :]
        val_data = sequences[-val_data_len:, :]

        print("Training on:", train_data_len, file=sys.stderr)
        print("Validating on:", val_data_len, file=sys.stderr)

        val_contexts, val_words = val_data[:, :-1], val_data[:, -1]
        val_words = to_categorical(val_words, num_classes=vocab_size)
        val_data = val_contexts, val_words

        # How many times per epoch we will ask the batch generator to yield a batch?
        steps = train_data_len / self.batch_size
        print("Steps:", int(steps), file=sys.stderr)

        # Training:
        start = time.time()
        history = self.model.fit(
            self.batch_generator(train_data, vocab_size, self.batch_size),
            steps_per_epoch=steps,
            epochs=10,
            verbose=1,
            callbacks=[earlystopping],
            validation_data=val_data,
        )
        end = time.time()
        training_time = int(end - start)
        print("LSTM training took {} seconds".format(training_time), file=sys.stderr)

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
        if self.ext_vocab:
            context_vocab = self.ext_word_index  # if we use external word embeddings
        else:
            context_vocab = self.inv_index  # if we use only training corpus information

        if entity in self.inv_index and all(
            [word in context_vocab for word in context]
        ):
            entity_id = self.inv_index[entity]
            context_ids = np.array([[context_vocab[w] for w in context]])
        # Decreased probability for out-of-vocabulary words:
        else:
            return 1 / self.corpus_size

        prediction = self.model.predict(context_ids).ravel()  # Probability distribution
        probability = prediction[entity_id]  # Probability of the correct word
        return probability

    def generate(self, context=None):
        if self.ext_vocab:
            context_vocab = self.ext_word_index  # if we use external word embeddings
        else:
            context_vocab = self.inv_index  # if we use only training corpus information

        if all([word in context_vocab for word in context]):
            context_ids = np.array([[context_vocab[w] for w in context]])
            prediction = self.model.predict(
                context_ids
            ).ravel()  # Probability distribution
            word_id = prediction.argmax()  # Entry with the highest probability
            word = self.word_index[word_id]  # Word corresponding to this entry
        else:
            word = np.random.choice(self.word_index)
        return word

    def save(self, filename):
        self.model.save(filename)
        out_dump = [self.inv_index, self.corpus_size, self.ext_vocab]
        with open(filename.split(".")[0] + ".pickle.gz", "wb") as out:
            pickle.dump(out_dump, out)
        print(
            "Model saved to {} and {} (vocabulary)".format(
                filename, filename.split(".")[0] + ".pickle.gz"
            ),
            file=sys.stderr,
        )

    def load(self, filename):
        self.model = load_model(filename)
        voc_file = filename.split(".")[0] + ".pickle.gz"
        with open(voc_file, "rb") as f:
            self.inv_index, self.corpus_size, self.ext_vocab = pickle.load(f)
        self.word_index = sorted(self.inv_index, key=self.inv_index.get)
        if self.ext_vocab:
            self.ext_word_index = {}
            for nr, word in enumerate(self.ext_vocab):
                self.ext_word_index[word] = nr
        print("Model loaded from {} and {}".format(filename, voc_file), file=sys.stderr)
        print(self.model.summary(), file=sys.stderr)
