import word2vec
import csv
from nltk.corpus import gutenberg
import nltk


def main():
    # data = word2vec.read_nltk_dataset(gutenberg)
    # word2vec.preprocess_dataset(data, save=True)
    with open('corpus.csv', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)[0]

    word2vec.prepare_training_data(data, 5, save=True)



if __name__ == "__main__":
    main()
