import language_model
import csv
from tensorflow import keras
from collections import defaultdict
from nltk.corpus import gutenberg
import nltk
import tqdm
import itertools
import numpy as np
from scipy.spatial import distance


def find_sim_by_vector(vocabulary, target_vector, sim_number, distance_func):
    dist_dict = {}
    for key, value in vocabulary.items():
        dist_dict[key] = distance_func(target_vector, value)

    sorted_sim_list = list(sorted(dist_dict.items(), key=lambda item: item[1]))

    return sorted_sim_list[:sim_number]


def find_sim(vocabulary, word, sim_number, distance_func):
    target_vector = vocabulary[word]

    dist_dict = {}
    for key, value in vocabulary.items():
        dist_dict[key] = distance_func(target_vector, value)

    sorted_sim_list = list(sorted(dist_dict.items(), key=lambda item: item[1]))

    return sorted_sim_list[:sim_number]


def main():
    books = defaultdict(lambda: defaultdict(lambda: []))
    books_names = gutenberg.fileids()
    for book_name in books_names:
        books[book_name] = gutenberg.words(book_name)

    model = language_model.LanguageModel()
    #model.read_dataset(books)
    #model.preprocess_dataset(save=True)
    model.load_preprocessed_data()
    model.prepare_language_model_training_data(30)

    #model.set_default_language_model()
    #model.train_model()
    rnn = keras.models.load_model("model")
    model.set_model(rnn)

    first_words = "Now , when I say that I am in the habit of going to sea whenever I begin to grow hazy about the eyes"
    first_words = first_words.split()
    model.generate_text(first_words, 100)


if __name__ == "__main__":
    main()
