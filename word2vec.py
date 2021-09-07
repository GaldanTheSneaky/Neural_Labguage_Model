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


def read_nltk_dataset(dataset) -> dict:  # change to read any data
    """Reads nltk dataset

    Args:
        dataset: nltk.corpus.dataset
    """
    books = defaultdict(lambda: defaultdict(lambda: []))
    books_names = dataset.fileids()
    for book_name in books_names:
        books[book_name] = dataset.words(book_name)

    return books


def preprocess_dataset(dataset, save=False) -> list:
    """Cleans up dataset and turns it into single list
    (lemmatizes, drops out stopwords and punctuation)
    (for future dev: add processing options)

    :Args:
        save: saves dataset to file "corpus.csv" if True
    """
    data = []
    lemmatizer = WordNetLemmatizer()
    for key in dataset.keys():
        dataset[key] = [lemmatizer.lemmatize(word.lower()).translate(str.maketrans('', '', string.punctuation))
                        for word in dataset[key]
                        if word not in stopwords.words('english')]

        dataset[key] = list(filter(None, dataset[key]))
        data.extend(dataset[key])
        print(key)

    if save:
        with io.open('corpus.csv', 'w', newline='', encoding="utf-8") as file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            wr.writerow(data)

    return data


def prepare_training_data(data, window_size, save=False) -> tuple:
    """Creats training data for CBOW model in format ([x[i - window_size], x[i - window_size +1], ....,
    x[i-1], x[i+1], x[i+2], ...., x[i+window_size]],x[i]) and maps every word to unique integer.
    Returns x_train, y_train and length of the vocabulary.

    Args:
        window_size: size of training sample/2
        save: saves training data to rtaining_data.csv if True. May take extremely long time
    """
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

    return x_train, y_train, len(vectorizer.vocabulary_)


def train_model(x_train, y_train, vocabulary_size):
    x_validation = x_train[int(len(x_train) * 0.8):]
    x_train = x_train[:int(len(x_train) * 0.8)]
    y_validation = y_train[int(len(y_train) * 0.8):]
    y_train = y_train[:(int(len(y_train) * 0.8))]

    print(tf.version.VERSION)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model = Sequential()
    model.add(Embedding(vocabulary_size, 200, input_length=len(x_train[0])))

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(vocabulary_size, activation='softmax'))

    opt = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train,
              batch_size=300,
              epochs=500,
              validation_data=(x_validation, y_validation),
              verbose=1)

    model.save('model')
    print("Finish")
