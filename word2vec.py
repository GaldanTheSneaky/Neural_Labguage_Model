import nltk
import io
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam


class sliceable_deque(collections.deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return collections.deque.__getitem__(self, index)


def read_nltk_dataset(dataset):  # change to read any data
    books = defaultdict(lambda: defaultdict(lambda: []))
    books_names = dataset.fileids()
    for book_name in books_names:
        books[book_name] = dataset.words(book_name)

    return books


def preprocess_dataset(dataset, save=False):
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


def prepare_training_data(data, window_size, save=False):
    vocabulary = set(data)
    vectorizer = CountVectorizer(min_df=0, lowercase=False, tokenizer=lambda txt: txt.split())
    vectorizer.fit(vocabulary)
    data = [vectorizer.vocabulary_[word] for word in data]

    training_data = []
    window = sliceable_deque(maxlen=window_size * 2 + 1)
    for i in range(window_size * 2):
        window.append(data[i])

    for i in range(len(data[window_size * 2 + 1:])):
        window.append(data[i])
        x_train = list(window[:window_size])
        x_train.extend(list(window[window_size + 1:]))
        y_train = window[window_size]
        training_data.append([x_train, y_train])

    if save:  # takes extremely long time
        with io.open('training_data.csv', 'w', newline='', encoding="utf-8") as file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            wr.writerow(training_data)

    random.shuffle(training_data)
    training_data = np.array(training_data, dtype=object)

    x_train = list(training_data[:, 0])
    y_train = list(training_data[:, 1])

    return x_train, y_train, len(vectorizer.vocabulary_)


def train_model(x_train, y_train, vocabulary_size):
    x_train = [[word/vocabulary_size for word in sequence] for sequence in x_train]
    x_validation = x_train[int(len(x_train) * 0.8):]
    x_train = x_train[:int(len(x_train) * 0.8)]
    y_validation = y_train[int(len(y_train) * 0.8):]
    y_train = y_train[:(int(len(y_train) * 0.8))]

    print(tf.version.VERSION)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, 200, input_length=len(x_train[0])))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(vocabulary_size + 1, activation='softmax'))

    opt = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    model = model.fit(x_train, y_train,
                      batch_size=1000,
                      epochs=20,
                      validation_data=(x_validation, y_validation),
                      verbose=1)

    model.save()
    print("Finish")

