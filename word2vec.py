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
            self._dataset[key] = [
                lemmatizer.lemmatize(word.lower()).translate(str.maketrans('', '', string.punctuation))
                for word in self._dataset[key]
                if word not in stopwords.words('english')]

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

    def prepare_cbow_training_data(self, window_size, save=False) -> tuple:
        """Creats training data for CBOW model in format ([x[i - window_size], x[i - window_size +1], ....,
        x[i-1], x[i+1], x[i+2], ...., x[i+window_size]],x[i]) and maps every word to unique integer.
        Returns shape of x_train, length of y_train and length of the vocabulary.

        Args:
            window_size: size of training sample/2
            save: saves training data to rtaining_data.csv if True. May take extremely long time
        """
        data = self._preprocessed_data
        vocabulary = set(data)
        vectorizer = CountVectorizer(min_df=0, lowercase=False, tokenizer=lambda txt: txt.split())
        vectorizer.fit(vocabulary)
        data = [vectorizer.vocabulary_[word] for word in data]

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
        self._vocabulary = vectorizer.vocabulary_
        self._decode_vocabulary = dict((v, k) for k, v in self._vocabulary.items())

        return np.shape(x_train), len(y_train), len(vectorizer.vocabulary_)

    def encode_word(self, word) -> int:
        """Returns unique index of word
        """
        return self._vocabulary[word]

    def decode_word(self, code):
        """Returns word by unique index
        """
        return self._decode_vocabulary[code]

    def set_default_cbow_model(self):
        """Sets default keras model for CBOW:
            input -> Embedding-200 ->Bidirectional CuDNNLSTM-256 -> Droput-0.2 ->BatchNormalization
            Bidirectional CuDNNLSTM-128 -> Droput-0.2 -> BatchNormalization -> Dense-64-relu-> Droput-0.2 ->
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
