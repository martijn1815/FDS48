#!/usr/bin/python3
"""
File:       assignment1.py
Author:     FDS Group 48
Date:       September 2020
"""

import argparse
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import tarfile
import json
from geopy.geocoders import Nominatim

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    parser.add_argument("-run",
                        metavar="CLASSIFIER",
                        help="Run sentiment classification on twitter data, "
                             "provide trained classifier (pickle file)")
    parser.add_argument("-data",
                        metavar="CLASSIFIER",
                        help="Run sentiment classification on twitter data with cleaned data, "
                             "provide trained classifier (pickle file)")
    args = parser.parse_args()

    # test compatibility of parameters
    if args.train and args.test:
        raise RuntimeError("-train and -test can not be called at the same time.")
    if args.train and args.run:
        raise RuntimeError("-train and -run can not be called at the same time.")
    if args.test and args.run:
        raise RuntimeError("-test and -run can not be called at the same time.")
    if args.data and not args.run:
        raise RuntimeError("-data can only be called together with -run.")
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

    data_set['text'] = data_set['text'].str.replace('@[^\s]+', '')
    data_set['text'] = data_set['text'].str.replace('[^\w\s]', '')
    data_set['text'] = data_set['text'].str.replace('\[.*?\]', '')
    data_set['text'] = data_set['text'].str.replace('[‘’“”…]', '')
    data_set['text'] = data_set['text'].str.replace('\w*\d\w*', '')

    data_set['text'] = data_set['text'].str.lower().str.split()  # lower and split the text
    data_set['text'] = data_set['text'].apply(lambda x: [item for item in x if item not in stopword_list])  # remove stopwords

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


def city_state_country(coord):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geolocator.reverse(coord, exactly_one=True)
        address = location.raw['address']
        #city = address.get('city', '')
        state = address.get('state', '')
        #country = address.get('country', '')
        return state
    except:
        return 'NULL'


def train(file_name):
    """
    Train and Test a sentiment classifier using NaiveBayes and sentiment140 data.
    :param file_name:   string
    """
    # Loading training dataset
    print("Loading dataset:", end=" ")
    data_set = pd.read_csv("training.1600000.processed.noemoticon.csv",
                           encoding='latin-1',
                           names=["Polarity", "Tweet ID", "Date", "Query", "User", "text"])
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
            tweets_and_polarity.append((data_set['text'].iloc[i], tweet_polarity))
    print("Done")

    # Feature extraction
    print("Feature extraction:", end=" ")
    word_features = get_word_features(get_words_in_tweets(tweets_and_polarity))
    save_pickle_file(word_features, file_name + "_word_features")
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
    """
    Small test sentiment classification example
    :param file_name:   string, file name of a pickle-file containing a trained classifier
    """
    model = load_pickle(file_name)
    # Testing
    tweet_positive = 'I like Larry'
    print(tweet_positive)
    print(model.classify(extract_features(file_name, tweet_positive.split())))


def classify_tweets(file_name, data_file):
    if not data_file:
        # Load Data
        print("Loading dataset:", end=" ")
        tar = tarfile.open("geotagged_tweets_20160812-0912.tar")
        files = tar.getmembers()
        f = tar.extractfile(files[0])
        #data = pd.read_json(f, lines=True)
        # Efficient way of loading json to panda:
        # https://medium.com/@ram.parameswaran22/a-relatively-faster-approach-for-reading-json-lines-file-into-pandas-dataframe-90b57353fd38
        lines = f.read().splitlines()
        df_inter = pd.DataFrame(lines[:10])
        df_inter.columns = ['json_element']
        df_inter['json_element'].apply(json.loads)
        data = pd.json_normalize(df_inter['json_element'].apply(json.loads))
        print("Done")
        #print(list(data))  # Column titles

        print("Cleaning dataset:", end=" ")
        # Only included wanted columns
        data = data[['id', 'text', 'source', 'place.country_code', 'place.bounding_box.coordinates', 'coordinates',
                     'lang', 'user.followers_count', 'user.friends_count']]
        #print('\n', list(data))  # Column titles

        # Exclude tweets outside of US (location) - filter on country code US (Country where tweet is posted)
        data = data[data['place.country_code'] == "US"]
        data = data[data['lang'] == 'en']

        # Get states
        pd.set_option('display.max_columns', None)
        data['latitude'] = data['place.bounding_box.coordinates'].apply(lambda x: x[0][0][0])
        data['longitude'] = data['place.bounding_box.coordinates'].apply(lambda x: x[0][0][1])
        data['coordinates'] = data['longitude'].astype(str) + ", " + data['latitude'].astype(str)
        #print(data.head())
        data['state'] = data['coordinates'].apply(city_state_country)

        # Determine tags and hashs (some tweets dont have tags only hash)
        data['tags_mention'] = data.text.str.findall(r'(?<![@\w])@(\w{1,25})').apply(','.join)
        data['hash_mention'] = data.text.str.findall(r"#(\w+)").apply(','.join)
        data['tags_hash'] = data['hash_mention'] + data['tags_mention']
        # Lower cases in tags and hashes
        data['tags_hash'] = data['tags_hash'].apply(lambda x: x.lower())

        # See if either trum or clinton mentioned in hash or tags
        data['Trump'] = data['tags_hash'].str.contains(r'trump|maga', na=False)
        data['Clinton'] = data['tags_hash'].str.contains(r'clinton|imwithher|hillary', na=False)

        # Filter bots (hyperlinks in text)
        # Removing dubious sources:
        sources = ('<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>',
                   '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>',
                   '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>',
                   '<a href="https://twitter.com/download/android" rel="nofollow">Twitter for Android Tablets</a>',
                   '<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>',
                   '<a href="http://instagram.com" rel="nofollow">Instagram</a>')
        data = data[data.source.isin(sources)]
        # Removing people with no friends and no followers
        data = data[(data['user.followers_count'] > 0) & (data['user.friends_count'] > 0)]

        save_pickle_file(data, file_name + "_twitter_data")
        print("Done")

    else:
        print("Loading dataset:", end=" ")
        data = load_pickle(data_file)
        print("Done")

    print("Size dataset:", len(data))
    #print("Columns", list(data))  # Column titles

    # Pre-Process Data
    print("Classifying data:", end=" ")
    data = clean_data(data)  # Data cleaning

    # Apply classifier to column
    model = load_pickle(file_name)
    data['classification_polarity'] = data['text'].apply(lambda x: model.classify((extract_features(file_name, x))))
    save_pickle_file(data, "twitter_data_classified")
    print("Done")
    #print("Columns", list(data))  # Column titles
    #print(data.head())

    # Get data per state and save to CSV
    print("Save data per state to CSV:", end=" ")
    states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
              'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
              'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
              'Maine' 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
              'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
              'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
              'North Carolina', 'North Dakota', 'Ohio',
              'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
              'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah',
              'Vermont', 'Virginia', 'Washington', 'West Virginia',
              'Wisconsin', 'Wyoming']

    with open('sentiment_polarity_states.csv', 'w') as f:
        f.write("state,"
                "trump_pos,trump_neg,trump_ratio,"
                "clinton_pos,clinton_neg,clinton_ratio,"
                "total_pos,total_neg,total_ratio\n")
        for state in states:
            trump_pos = len(data[(data['classification_polarity'] == 'positive') &
                                 (data['Trump'] == 'True') &
                                 (data['state'].str.lower() == state.lower())])
            trump_neg = len(data[(data['classification_polarity'] == 'negative') &
                                 (data['Trump'] == 'True') &
                                 (data['state'].str.lower() == state.lower())])
            trump_ratio = 0 if (trump_pos + trump_neg) == 0 else trump_pos / (trump_pos + trump_neg)
            clinton_pos = len(data[(data['classification_polarity'] == 'positive') &
                                   (data['Clinton'] == 'True') &
                                   (data['state'].str.lower() == state.lower())])
            clinton_neg = len(data[(data['classification_polarity'] == 'negative') &
                                   (data['Clinton'] == 'True') &
                                   (data['state'].str.lower() == state.lower())])
            clinton_ratio =  0 if (clinton_pos + clinton_neg) == 0 else clinton_pos / (clinton_pos + clinton_neg)
            total_pos = len(data[(data['classification_polarity'] == 'positive') &
                                 (data['state'].str.lower() == state.lower())])
            total_neg = len(data[(data['classification_polarity'] == 'negative') &
                                 (data['state'].str.lower() == state.lower())])
            total_ratio = 0 if (total_pos + total_neg) == 0 else total_pos / (total_pos + total_neg)
            f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(state,
                                                                       trump_pos,
                                                                       trump_neg,
                                                                       trump_ratio,
                                                                       clinton_pos,
                                                                       clinton_neg,
                                                                       clinton_ratio,
                                                                       total_pos,
                                                                       total_neg,
                                                                       total_ratio))
        trump_pos = len(data[(data['classification_polarity'] == 'positive') &
                             (data['Trump'] == 'True')])
        trump_neg = len(data[(data['classification_polarity'] == 'negative') &
                             (data['Trump'] == 'True')])
        trump_ratio =  0 if (trump_pos + trump_neg) == 0 else trump_pos / (trump_pos + trump_neg)
        clinton_pos = len(data[(data['classification_polarity'] == 'positive') &
                               (data['Clinton'] == 'True')])
        clinton_neg = len(data[(data['classification_polarity'] == 'negative') &
                               (data['Clinton'] == 'True')])
        clinton_ratio =  0 if (clinton_pos + clinton_neg) == 0 else clinton_pos / (clinton_pos + clinton_neg)
        total_pos = len(data[data['classification_polarity'] == 'positive'])
        total_neg = len(data[data['classification_polarity'] == 'negative'])
        total_ratio =  0 if (total_pos + total_neg) == 0 else total_pos / (total_pos + total_neg)
        f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}".format("Total",
                                                                 trump_pos,
                                                                 trump_neg,
                                                                 trump_ratio,
                                                                 clinton_pos,
                                                                 clinton_neg,
                                                                 clinton_ratio,
                                                                 total_pos,
                                                                 total_neg,
                                                                 total_ratio))
    print("Done")


if __name__ == "__main__":
    args = parse_arguments()
    if args.train:
        train(args.train)
    elif args.test:
        test(args.test)
    elif args.run:
        if args.data:
            classify_tweets(args.run, args.data)
        else:
            classify_tweets(args.run, None)
    else:
        print("No parameters given. Use '-h' for help.")
