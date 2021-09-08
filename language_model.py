import nltk
import io
import os
import csv
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg

import collections
from collections import deque
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
import itertools
import tqdm
import random
import numpy as np
from scipy.spatial import distance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, BatchNormalization, Flatten, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam


class SliceableDeque(collections.deque):
    """Makes in easier to generate training data
    """
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return collections.deque.__getitem__(self, index)


class LanguageModel:

    def __init__(self):
        self._dataset = {}
        self._preprocessed_data = []
        self._x_train = []
        self._y_train = []
        self._vocabulary = {}
        self._decode_vocabulary = {}
        self._sequnce_len = 0
        self._model = None

    def read_dataset(self, dataset: dict) -> None:
        """Reads dataset

        Args:
            dataset: nltk.corpus.dataset
        """
        self._dataset = dataset

    def preprocess_dataset(self, save=False) -> None:
        """Cleans up dataset and turns it into single list
        (lemmatizes, drops out stopwords and punctuation)
        (for future dev: add processing options)

        :Args:
            save: saves dataset to file "corpus.csv" if True
        """
        data = []
        lemmatizer = WordNetLemmatizer()
        for key in self._dataset.keys():
            self._dataset[key] = [word for word in self._dataset[key]]
                # lemmatizer.lemmatize(word.lower()).translate(str.maketrans('', '', string.punctuation))
                # for word in self._dataset[key]
                # if word not in stopwords.words('english')]

            self._dataset[key] = list(filter(None, self._dataset[key]))
            data.extend(self._dataset[key])
            print(key)

        if save:
            with io.open('corpus.csv', 'w', newline='', encoding="utf-8") as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(data)

        self._preprocessed_data = data

    def load_preprocessed_data(self) -> None:
        """Loads preprocessed data from corpus.csv
        """
        with open('corpus.csv', newline='') as file:
            reader = csv.reader(file)
            self._preprocessed_data = list(reader)[0]

    def _encode_data(self) -> None:
        """"Creates unique id for every word
        """
        vocabulary = set(self._preprocessed_data)
        vectorizer = CountVectorizer(min_df=0, lowercase=False, tokenizer=lambda txt: txt.split())
        vectorizer.fit(vocabulary)
        self._preprocessed_data = [vectorizer.vocabulary_[word] for word in self._preprocessed_data]
        self._vocabulary = vectorizer.vocabulary_
        self._decode_vocabulary = dict((v, k) for k, v in self._vocabulary.items())

    def prepare_cbow_training_data(self, window_size, save=False) -> tuple:
        """Creats training data for CBOW model in format ([x[i - window_size], x[i - window_size +1], ....,
        x[i-1], x[i+1], x[i+2], ...., x[i+window_size]],x[i]) and maps every word to unique integer.
        Returns shape of x_train, length of y_train and length of the vocabulary.

        Args:
            window_size: size of training sample/2
            save: saves training data to rtaining_data.csv if True. May take extremely long time
        """
        self._encode_data()
        data = self._preprocessed_data

        training_data = []
        window = SliceableDeque(maxlen=window_size * 2 + 1)
        for i in range(window_size * 2):
            window.append(data[i])

        for i in range(window_size * 2, len(data[window_size * 2 + 1:])):
            window.append(data[i])
            x_train = list(window[:window_size])
            x_train.extend(list(window[window_size + 1:]))
            y_train = window[window_size]
            training_data.append([x_train, y_train])

        if save:
            with io.open('training_data.csv', 'w', newline='', encoding="utf-8") as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(training_data)

        random.shuffle(training_data)
        training_data = np.array(training_data, dtype=object)

        x_train = list(training_data[:, 0])
        y_train = list(training_data[:, 1])

        self._x_train = x_train
        self._y_train = y_train

        return np.shape(x_train), len(y_train), len(self._vocabulary)

    def prepare_language_model_training_data(self, seqeunce_length, save=False) -> tuple:
        """Creates training data for language model of format ([x[i], x[i+1], ..., x[suqeunce_length]],
         x[suqeunce_length+1]) and maps every word to unique integer.
        Returns shape of x_train, length of y_train and length of the vocabulary.

        Args:
            seqeunce_length: size of training sample
            save: saves training data to rtaining_data.csv if True. May take extremely long time
        """
        self._encode_data()
        data = self._preprocessed_data

        training_data = []
        sequence = SliceableDeque(maxlen=seqeunce_length + 1)
        for i in range(seqeunce_length):
            sequence.append(data[i])

        for i in range(seqeunce_length, len(data[seqeunce_length + 1:])):
            sequence.append(data[i])
            x_train = list(sequence[:seqeunce_length])
            y_train = sequence[-1]
            training_data.append([x_train, y_train])

        if save:
            with io.open('training_data.csv', 'w', newline='', encoding="utf-8") as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(training_data)

        random.shuffle(training_data)
        training_data = np.array(training_data, dtype=object)

        x_train = list(training_data[:, 0])
        y_train = list(training_data[:, 1])

        self._x_train = x_train
        self._y_train = y_train
        self._sequnce_len = seqeunce_length

        return np.shape(x_train), len(y_train), len(self._vocabulary)

    def encode_word(self, word) -> int:
        """Returns unique id of word
        """
        return self._vocabulary[word]

    def decode_word(self, code):
        """Returns word by unique id
        """
        return self._decode_vocabulary[code]

    def set_default_cbow_model(self):
        """Sets default keras model for CBOW:
            input -> Embedding-200 ->Bidirectional CuDNNLSTM-256 -> Dropout-0.2 ->BatchNormalization
            Bidirectional CuDNNLSTM-128 -> Dropout-0.2 -> BatchNormalization -> Dense-64-relu-> Dropout-0.2 ->
            BatchNormalization -> output-softmax
            optimizer: Adam(learning_rate=0.001, decay=1e-6)
            Loss: sparse_categorical_crossentropy
            Metrics: accuracy
            """
        model = Sequential()
        model.add(Embedding(len(self._vocabulary), 200, input_length=len(self._x_train[0])))

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(len(self._vocabulary), activation='softmax'))

        opt = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        self._model = model

    def set_default_language_model(self):
        """Sets default keras model for CBOW:
            input -> Bidirectional CuDNNLSTM-256 -> Dropout-0.2 ->BatchNormalization
            Bidirectional CuDNNLSTM-128 -> Dropout-0.2 -> BatchNormalization -> Dense-64-relu-> Dropout-0.2 ->
            BatchNormalization -> output-softmax
            optimizer: Adam(learning_rate=0.001, decay=1e-6)
            Loss: sparse_categorical_crossentropy
            Metrics: accuracy
            """
        model = Sequential()
        model.add(Embedding(len(self._vocabulary), 200, input_length=len(self._x_train[0])))

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(len(self._vocabulary), activation='softmax'))

        opt = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        print(model.summary())
        self._model = model


    def set_model(self, model) -> None:
        self._model = model

    def train_model(self, batch_size=300, epochs=30, verbose=1) -> None:
        """Trains model and saves to '\model'

        """
        x_train = self._x_train
        y_train = self._y_train
        x_validation = x_train[int(len(x_train) * 0.8):]
        x_train = x_train[:int(len(x_train) * 0.8)]
        y_validation = y_train[int(len(y_train) * 0.8):]
        y_train = y_train[:(int(len(y_train) * 0.8))]

        print(tf.version.VERSION)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self._model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_validation, y_validation),
                        verbose=verbose)

        self._model.save('model')
        print("Finish")

    def generate_text(self, first_words, length): # length not more than self._sequnce_len, need fix
        first_words = [self.encode_word(word) for word in first_words]
        input_text = SliceableDeque(maxlen=self._sequnce_len)
        for i in range(self._sequnce_len):
            input_text.append(0)

        for i in range(1, len(first_words) + 1):
            input_text[-i] = first_words[-i]

        text = list(input_text)

        for i in range(length):
            output = self._model.predict(np.array([list(input_text)]))[0]
            word_idx = np.argmax(output)
            input_text.append(word_idx)
            text.append(word_idx)

        text = [self.decode_word(word) for word in text if word != 0]
        text = " ".join(text)
        print(text)
