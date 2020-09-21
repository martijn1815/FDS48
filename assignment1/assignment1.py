#!/usr/bin/python3
"""
File:       assignment1.py
Author:     FDS Group 48
Date:       September 2020
"""

import sys
import argparse
import pickle
import os
import tarfile
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
import random
import re


def parse_arguments():
    """
    Read arguments from a command line
    :return:    args
    """

    parser = argparse.ArgumentParser(description="Sentiment classification")
    parser.add_argument("-train",
                        metavar="FILE",
                        help="Run sentiment classification training, "
                             "provide the save file name")
    parser.add_argument("-test",
                        metavar="CLASSIFIER",
                        help="Run sentiment classification test, "
                             "provide trained classifier (pickle file)")
    args = parser.parse_args()

    # test compatibility of parameters
    if args.train and args.test:
        raise RuntimeError("-train and -test can not be called at the same time.")
    return args


def save_pickle_file(data, file_name):
    """
    Saves data as a pickle file
    :param data:
    :param file_name:   The desired name for the file (without .pickle)
    """
    with open(file_name +'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    """
    Loads data from a pickle file
    :param file_name:   The desired name for the file (without .pickle)
    :return data:
    """
    with open(file_name + '.pickle', 'rb') as file:
        data = pickle.load(file)
    return data


def clean_data(data_set):
    """
    Data cleaning - removing punctuation from text; removing tags; removing links
    :param data_set:    a panda data frame
    :return data_set:
    """
    stopword_list = stopwords.words('english')  # Define stopwords

    data_set['Text'] = data_set['Text'].str.replace('@[^\s]+', '')
    data_set['Text'] = data_set['Text'].str.replace('[^\w\s]', '')
    data_set['Text'] = data_set['Text'].str.replace('\[.*?\]', '')
    data_set['Text'] = data_set['Text'].str.replace('[‘’“”…]', '')
    data_set['Text'] = data_set['Text'].str.replace('\w*\d\w*', '')

    data_set['Text'] = data_set['Text'].str.lower().str.split()  # lower and split the text
    data_set['Text'] = data_set['Text'].apply(lambda x: [item for item in x if item not in stopword_list])  # remove stopwords

    return data_set


def get_words_in_tweets(tweets):
    all_words= []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = list(wordlist.keys())
    return word_features


def extract_features(file_name, document):
    """
    Loads word_features from pickle file and extracts features of given document
    :param file_name:   string
    :param document:    string
    :return features:   dict
    """
    word_features = load_pickle("word_features_" + file_name)
    document_words = set(document)
    features = {}
    for word in word_features:
        if word in document_words:
            features[word] = True
        else:
            features[word] = False
    return features


def train(file_name):
    # Loading training dataset
    print("Loading dataset:", end=" ")
    data_set = pd.read_csv("training.1600000.processed.noemoticon.csv",
                           encoding='latin-1',
                           names=["Polarity", "Tweet ID", "Date", "Query", "User", "Text"])
    print("Done")

    # Pre-Process Data
    print("Pre-Processing data:", end=" ")
    data_set = data_set.sample(n = 10000)  # Select random sample - otherwise the classifier will be too slow
    data_set = clean_data(data_set)  # Data cleaning

    # Create data that puts tweet texts together with sentiment score
    tweets_and_polarity = []
    for i in range(0, len(data_set)):
        if data_set["Polarity"].iloc[i] == 0 or data_set["Polarity"].iloc[i] == 4:
            if data_set["Polarity"].iloc[i] == 0:
                tweet_polarity = "negative"
            else:
                tweet_polarity = "positive"
            tweets_and_polarity.append((data_set['Text'].iloc[i], tweet_polarity))
    print("Done")

    # Feature extraction
    print("Feature extraction:", end=" ")
    word_features = get_word_features(get_words_in_tweets(tweets_and_polarity))
    save_pickle_file(word_features, "word_features_" + file_name)
    final_data_set = [(extract_features(file_name, tweet), pol_score) for (tweet, pol_score) in tweets_and_polarity]

    # Split training and test set
    training_set = final_data_set[int(len(final_data_set) / 10):]
    test_set = final_data_set[:int(len(final_data_set) / 10)]
    print("Done")

    # Train and Save Classifier
    print("Training Classifier:", end=" ")
    model = nltk.NaiveBayesClassifier.train(training_set)
    save_pickle_file(model, file_name)
    print("Done\n")

    # Test Classifier
    model.show_most_informative_features()
    acc = nltk.classify.accuracy(model, test_set)
    print("\nAccuracy:", acc)


def test(file_name):
    classifier = load_pickle(file_name)

    # Testing
    tweet_positive = 'I like Larry'
    print(tweet_positive)
    print(classifier.classify(extract_features(file_name, tweet_positive.split())))



if __name__ == "__main__":
    args = parse_arguments()
    if args.train:
        train(args.train)
    elif args.test:
        test(args.test)
    else:
        print("No parameters given. Use '-h' for help.")
