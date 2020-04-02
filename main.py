import pandas as pd
import nltk.classify

from nltk.tokenize import word_tokenize
from random import shuffle
from classification import precision_recall

def get_features(df):
    features = list()
    for row in df.itertuples():
        category = row[1]
        tokens = word_tokenize(row[2])
        # room for preprocessing data

        features.append((tokens, category))

    return features


def train_bayes(train_feats):
    classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)
    
    return classifier


def split_features(features):
    train_feats, test_feats = [], []
    shuffle(features)
    cutoff = int(len(features) * 0.9)
    train_feats, test_feats = features[:cutoff], features[cutoff:]

    return train_feats, test_feats

def evaluation(classifier, test_features, categories):
    accuracy = nltk.classify.accuracy(classifier, test_feats)
    #precisions, recalls = precision_recall(classifier, test_feats)
    #f_measures = calculate_f(precisions, recalls)
    print("Accuracy: {0}".format(accuracy))
    #print(precisions)
    


def main():
    df = pd.read_csv('data/spam.csv',encoding = "ISO-8859-1")
    categories = ['ham','spam']
    unwanted = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    for item in unwanted:
        df.pop(item)
    features = get_features(df)

    # ruimte voor high info features

    train_feats, test_feats = split_features(features)
    classifier = train_bayes(train_feats)
    #evaluation(classifier, test_feats, categories) werkt nog niet

if __name__ == '__main__':
    main()
