import pandas as pd
import numpy as np
import nltk.classify
import collections
import random

from nltk.metrics import precision, recall
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.svm import LinearSVC
from sklearn import svm
from prettytable import PrettyTable


# Lowering the words improves our accuracy for Naive Bayes
# Lowering words did not have influence on the SVM

# Removing stopwords lowers the accuracy for Naive Bayes
# Removing stopwords also lowered the accuracy for SVM

# Porter stemmer seems to slightly improves accuracy for Naive Bayes
# Porter stemmer also slightly improced accuracy for SVM

# Lancaster stemmer seems to improve accuracy for Naive Bayes more than the Porter Stemmer
# Lancaster stemmer improved accuracy significantly for SVM, more than Porter

# Bigrams tend to lower the accuracy for Naive Bayes
# Bigrams also lowered accuracy for SVM

# We hebben een random state bij het shuffelen, zodat we elke keer dezelfde
# Dev en Test set hebben en zo de Test set niet gebruiken tot de uiteindelijke
# test zodat deze de hele tijd apart wordt gehouden

# Heb een screenshot van de Most Informative Features voor het stemmen
# (Staan met en zonder stemming op git)

# Parameter tuning:
# The Linear SVM had the best results
# Increasing the C value resulted in the same accuracy
# Decreasing the C value below 0.4 decreased the accuracy
# Thus, C => 0.05 is the best value.


# Results:

# Accuracy
# Naive Bayes:  0.933572710951526   &&  0.9301075268817204
# SVM:      0.9784560143626571  &&  0.9838709677419355
# SVM duidelijk hoger, dit moet dus statistisch getest worden

# Precision
# [HAM]     Naive Bayes scoort hoger op ham (bijna 1!), SVM scoort ook erg hoog.
# [SPAM]    SVM scoort veel hoger op spam, Naive Bayes erg laag (0.65 - 0.7).
# Dus:  Naive Bayes beter in 'ham' herkennen, maar niet precies met 'spam'.
#   SVM is veel beter in 'spam' herkennen, en is ook erg goed in 'ham'.
#   Precision: True Positive / True Positive + False Positive

# Recall
# [HAM]     SVM duidelijk hoger (bijna 1), maar Naive Bayes ook hoog (ongeveer 0.925)
# [SPAM]    Naive Bayes scoort hoger op spam (dus heeft bijna alle spam als spam gemarkeert),
#       de SVM scoort lager op spam.
# Dus:      SVM is beter in ham markeren als ham, maar laat daarmee ook meer spam door.
#       Naive Bayes laat minder ham door, maar houdt daarmee wel meer spam tegen.
#       --> naive bayes is dus 'veiliger' als je alle spam weg wilt hebben ondanks
#       dat daarmee ook nuttige info weg gaat
#   Recall: True Positive / True Positive + False Negative

# F-measure
# [HAM]     SVM duidelijk hoger, Naive Bayes ook wel hoog.
# [SPAM]    SVM echt stukken hoger, Naive Bayes een stuk lager door de lage precision.
# Dus:      De F-score laat inderdaad zien dat Naive Bayes gemiddeld minder goed presteert,
#       maar het ligt er aan wat je belangrijk vindt. Als geen 'ham' mailtjes wilt missen
#       en daarmee ook meer spam accepteert, is de SVM beter. Stel je wilt zo min mogelijk
#       spam, en je vindt het ok dat je daardoor vaker 'ham' mailtjes mist, is Naive Bayes
#       de betere classifier.


# Used from assignment 1
def bag_of_words(tokens):
    return dict([(token, True) for token in tokens])


def get_features(df):
    features = list()
    lancaster = LancasterStemmer()
    porter = PorterStemmer()
    for row in df.itertuples():
        category = row[1]

        # lower
        words = row[2]
        words = words.lower()

        # Tokenize
        tokens = word_tokenize(words)

        # Stemming the data
        # stemmed_tokens = [porter.stem(token) for token in tokens]
        stemmed_tokens = [lancaster.stem(token) for token in tokens]

        # List all stopwords
        # stop_words = set(stopwords.words('english'))

        # Create bag of words without stop_words
        # no_stopwords = bag_of_words(set(stemmed_tokens) - set(stop_words))

        # Find bigrams
        # bigram_finder = BigramCollocationFinder.from_words(stemmed_tokens)
        # bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 200)

        # bag with stopwords
        # bag = bag_of_words(tokens)
        bag = bag_of_words(stemmed_tokens)
        # bag = bag_of_words(stemmed_tokens + bigrams)

        # Append the preprocessed row to the feature list
        features.append((bag, category))
        # features.append((no_stopwords, category))

    return features


def split_features(features):
    random.seed(4)
    random.shuffle(features)
    cutoff = int(len(features) * 0.8)
    train_feats, temp_feats = features[:cutoff], features[cutoff:]

    cutoff = int(len(temp_feats) * 0.5)
    dev_feats, test_feats = temp_feats[:cutoff], temp_feats[cutoff:]

    return train_feats, dev_feats, test_feats


def train_bayes(train_feats):
    classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)

    return classifier


def train_svm(train_feats):
    classifier = nltk.classify.SklearnClassifier(LinearSVC(C=1))
    classifier.train(train_feats)

    return classifier


# Used from assignment 1
def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls


# Used from assignment 1
def calculate_f(precisions, recalls):
    f_measures = {}
    for x in precisions:
        if precisions[x] is not None:
            uppervalue = 2 * precisions[x] * recalls[x]
            lowervalue = precisions[x] + recalls[x]
            f_measures[x] = round(uppervalue/lowervalue, 6)
        else:
            f_measures[x] = "NA"

    return f_measures


def evaluation(classifier, test_features, categories):
    accuracy = nltk.classify.accuracy(classifier, test_features)
    precisions, recalls = precision_recall(classifier, test_features)
    f_measures = calculate_f(precisions, recalls)

    print("\nAccuracy: {0}".format(accuracy))
    table = PrettyTable(['Category', 'Precision', 'Recall',
                 'F-measure'])
    for category in categories:
        table.add_row([category, round(precisions[category], 5),
                   round(recalls[category], 5),
                   round(f_measures[category], 5)])
    print(table)


def analysis(classifier):
    print("\nAnalysis")
    classifier.show_most_informative_features(10)


def main():
    df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    categories = ['ham', 'spam']
    unwanted = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    for item in unwanted:
        df.pop(item)

    features = get_features(df)

    # Split features into train, dev and test set
    train_feats, dev_feats, test_feats = split_features(features)

    print("Length train set: {0}".format(len(train_feats)))
    print("Length development set: {0}".format(len(dev_feats)))
    print("Length test set: {0}".format(len(test_feats)))
    print("Total number of features: {0}".format(len(features)))

    # Evaluation on dev set Naive Bayes
    print("\nDevelopment set evaluation of the Naive Bayes algorithm")
    bayes_classifier = train_bayes(train_feats)
    analysis(bayes_classifier)
    evaluation(bayes_classifier, dev_feats, categories)

    # Evaluation on dev set SVM
    print("\nDevelopment set evaluation of the Support Vector Machine")
    svm_classifier = train_svm(train_feats)
    evaluation(svm_classifier, dev_feats, categories)

    # Evaluation on test set
    print("\n\nFinal evaluation of Naive Bayes on the test set")
    evaluation(bayes_classifier, test_feats, categories)
    print("\n\nFinal evaluation of SVM on the test set")
    evaluation(svm_classifier, test_feats, categories)


if __name__ == '__main__':
    main()
