import nltk
import io
import csv
import string
from nltk.corpus import stopwords
from collections import deque
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import collections
import itertools
import tqdm

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
    window = sliceable_deque(maxlen=window_size*2+1)
    for i in range(window_size*2):
        window.append(data[i])

    for i in tqdm.tqdm(range(len(data[window_size*2+1:]))):
        window.append(data[i])
        x_train = list(window[:window_size])
        x_train.extend(list(window[window_size+1:]))
        y_train = window[window_size]
        training_data.append([x_train, y_train])

        if save:
            with io.open('training_data.csv', 'w', newline='', encoding="utf-8") as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(training_data)

    print("FINISHED")





