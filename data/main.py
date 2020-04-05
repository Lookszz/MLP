import pandas as pd
import nltk.classify
import collections

from nltk.metrics import precision, recall
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from string import punctuation
from random import shuffle




# vanuit antionio zn code gepakt: moet nog omgetypt worden
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


# uit eigen code --> ook nog omtypen
def calculate_f(precisions, recalls):
	f_measures = {}
	for x in precisions:
                if precisions[x] is not None:
                        uppervalue = 2 * precisions[x] * recalls[x]
                        lowervalue = precisions[x] + recalls[x]
                        f_measures[x] = round(uppervalue/lowervalue, 6)
                else:
                        f_measures[x] = "NA"
	#TODO calculate the f measure for each category using as input the precisions and recalls
	return f_measures


def get_features(df):
    features = list()
    for row in df.itertuples():
        category = row[1]
        tokens = word_tokenize(row[2])

        #lower
        [token.lower() for token in tokens]

        # remove stopwords
        #stop_words = set(stopwords.words('english'))
        #no_stopwords = []
        #for token in tokens:
            #if token not in stop_words:
                #no_stopwords.append(token)

        # remove punctuation
        no_punct = []
        for token in tokens:
            if token not in punctuation:
                no_punct.append(token)

        # no numbers
        no_integers = [token for token in no_punct if not (token.isdigit() or token[0] == '-' and token[1:].isdigit())]

        # stem
        lancaster = LancasterStemmer()
        lancaster_list = [lancaster.stem(token) for token in no_integers]

        # create bag of words
        bag = dict([(token, True) for token in lancaster_list])
        for item in bag:
            print(item)

        
        # room for preprocessing data

        features.append((bag, category))

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
    print("\nEvaluation")
    accuracy = nltk.classify.accuracy(classifier, test_features)
    precisions, recalls = precision_recall(classifier, test_features)
    f_measures = calculate_f(precisions, recalls)
    print("Accuracy: {0}".format(accuracy))

    for category in categories:
        print("\nCategory: {0}".format(category))
        print("Precision: {0}".format(precisions[category]))
        print("Recall: {0}".format(recalls[category]))
        print("F-measure: {0}".format(f_measures[category]))


def analysis(classifier):
	print("\nAnalysis")
	classifier.show_most_informative_features(10)    


def main():
    df = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
    categories = ['ham','spam']
    unwanted = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    for item in unwanted:
        df.pop(item)
    features = get_features(df)
    
    # ruimte voor high info features

    train_feats, test_feats = split_features(features)
    print("Length train set: {0}".format(len(train_feats)))
    print("Length test set: {0}".format(len(test_feats)))
    print("Total number of features: {0}".format(len(train_feats)+len(test_feats)))
    classifier = train_bayes(train_feats)
    evaluation(classifier, test_feats, categories)
    analysis(classifier)

if __name__ == '__main__':
    main()
