#!/usr/bin/python3

# Basic classifiction functionality with naive Bayes. File provided for the assignment on classification (IR course 2019/20)


import nltk.classify
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words, bag_of_non_stopwords, bag_of_words_in_set
from classification import precision_recall
from sklearn.svm import LinearSVC
from sklearn import svm
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from random import shuffle
from os import listdir # to read files
from os.path import isfile, join # to read files
import sys

def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

# return all the filenames in a folder
def get_filenames_in_folder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f))]



# reads all the files that correspond to the input list of categories and puts their contents in bags of words
def read_files(categories):
	feats = list ()
	#porter = PorterStemmer()
	lancaster = LancasterStemmer()
	print("\n##### Reading files...")
	for category in categories:
		files = get_filenames_in_folder('Volkskrant/' + category)
		num_files=0
		for f in files: 

			data = open('Volkskrant/' + category + '/' + f, 'r', encoding='UTF-8').read()
			#data = data.lower()
			#data = porter.stem(data)
			tokens = word_tokenize(data)
			lancaster_list = [lancaster.stem(token) for token in tokens]

			#bag = bag_of_words(tokens)
			#ww=high_information([(bag, category)], [category])
			#bag = bag_of_words(lancaster_list)

			#bag = bag_of_non_stopwords(tokens)
			bag = bag_of_non_stopwords(lancaster_list)
			feats.append((bag, category))
			#print len(tokens)
			num_files+=1
			#if num_files>=50: # you may want to de-comment this and the next line if you're doing tests (it just loads N documents instead of the whole collection so it runs faster
				#break
		
		print ("  Category %s, %i files read" % (category, num_files))

	print("  Total, %i files read" % (len(feats)))
	return feats



# splits a labelled dataset into two disjoint subsets train and test
def split_train_test(feats, split=0.9):
	train_feats = []
	test_feats = []
	#print (feats[0])

	shuffle(feats) # randomise dataset before splitting into train and test
	cutoff = int(len(feats) * split)
	train_feats, test_feats = feats[:cutoff], feats[cutoff:]	

	print("\n##### Splitting datasets...")
	print("  Training set: %i" % len(train_feats))
	print("  Test set: %i" % len(test_feats))
	return train_feats, test_feats



# TODO function to split the dataset for n fold cross validation
def split_folds(feats, folds=10):
	shuffle(feats) # randomise dataset before splitting into train and test
	
	# divide feats into n cross fold sections
	nfold_feats = []
	for n in range(folds):
		#TODO for each fold you need 1/n of the dataset as test and the rest as training
                nfolds = chunks(feats, 10)
                test_feats = nfolds[n]
                nfolds.remove(test_feats)
                train_feats = [item for sublist in nfolds for item in sublist]
                nfold_feats.append((train_feats, test_feats))

	print("\n##### Splitting datasets...")
	return nfold_feats



# trains a classifier
def train(train_feats):
    #classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)
    classifier = nltk.classify.SklearnClassifier(LinearSVC())
    #classifier = nltk.classify.SklearnClassifier(LinearSVC(C=10))
    classifier.train(train_feats)
    return classifier
	# the following code uses the classifier with add-1 smoothing (Laplace)
	# You may choose to use that instead
	#from nltk.probability import LaplaceProbDist
	#classifier = nltk.classify.NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)



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



# prints accuracy, precision and recall
def evaluation(classifier, test_feats, categories):
	#print ("\n##### Evaluation...")
	#print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
	print(nltk.classify.accuracy(classifier, test_feats))
	return nltk.classify.accuracy(classifier, test_feats)
	'''
	precisions, recalls = precision_recall(classifier, test_feats)
	f_measures = calculate_f(precisions, recalls)  

	print(" |-----------|-----------|-----------|-----------|")
	print(" |%-11s|%-11s|%-11s|%-11s|" % ("category","precision","recall","F-measure"))
	print(" |-----------|-----------|-----------|-----------|")
	for category in categories:
		if precisions[category] is None:
			print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
		else:
			print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], f_measures[category]))
	print(" |-----------|-----------|-----------|-----------|")
	'''
	




# show informative features
def analysis(classifier):
	print("\n##### Analysis...")
	classifier.show_most_informative_features(10)




# obtain the high information words
def high_information(feats, categories):
	#print("\n##### Obtaining high information words...")

	labelled_words = [(category, []) for category in categories]


	#1. convert the formatting of our features to that required by high_information_words
	from collections import defaultdict
	words = defaultdict(list)
	all_words = list()
	for category in categories:
		words[category] = list()

	for feat in feats:
		category = feat[1]
		bag = feat[0]
		for w in bag.keys():
			words[category].append(w)
			all_words.append(w)
#		break

	labelled_words = [(category, words[category]) for category in categories]
	#print(labelled_words)

	#2. calculate high information words
	high_info_words = set(high_information_words(labelled_words))
	#print(high_info_words)
	#high_info_words contains a list of high-information words. You may want to use only these for classification.
	# You can restrict the words in a bag of words to be in a given 2nd list (e.g. in function read_files)
	# e.g. bag_of_words_in_set(words, high_info_words)

	#print("  Number of words in the data: %i" % len(all_words))
	#print("  Number of distinct words in the data: %i" % len(set(all_words)))
	#print("  Number of distinct 'high-information' words in the data: %i" % len(high_info_words))

	return high_info_words



def main():
	# read categories from arguments. e.g. "python3 assignment_classification.py BINNENLAND SPORT KUNST"
	categories = list()
	for arg in sys.argv[1:]:
		categories.append(arg)


	# load categories from dataset
	feats = read_files(categories)
	high_info_words = high_information(feats, categories)
	high_info_feats = []
	for x in feats:
            high_info_dict = {}
            for y in x[0]:
                if y in high_info_words:
                    high_info_dict[y] = True
            high_info_feats.append((high_info_dict, x[1]))       


	train_feats, test_feats = split_train_test(high_info_feats)
	# TODO to use n folds you'd have to call function split_folds and have the subsequent lines inside a for loop
	nfold_feats = split_folds(high_info_feats)
	mean_acc = []
	#classifier = train(train_feats)
	#evaluation(classifier, test_feats, categories)
	#analysis(classifier)
	for train_feats, test_feats in nfold_feats:
                classifier = train(train_feats)
                mean_acc.append(evaluation(classifier, test_feats, categories))
	print(sum(mean_acc) / len(mean_acc))
if __name__ == '__main__':
	main()

