#sam.py for sentiment analysis module

import nltk
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
import statistics
from nltk.tokenize import word_tokenize
from statistics import mode
class VoteClassifier( ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify( features )
			votes.append(v)

		return statistics.mode(votes)
	def confidence( self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify( features )
			votes.append(v)

		choice_votes = votes.count(statistics.mode(votes))
		conf = choice_votes / len(votes)
		return conf

###
wf = open( "./pickle_jar/word_features5k.pickle", "rb")
word_features = pickle.load(wf)
wf.close()

random.shuffle( word_features )
def find_features( document ):
		if type(document) == str:
			document = word_tokenize( document )
		words = set( document )
		features = {}
		for w in word_features:
			features[w] = (w in words)

		return features


fs = open( "./pickle_jar/featuresets7058.pickle", "rb")
featuresets = pickle.load(fs)
fs.close()

random.shuffle( featuresets )
training_set = featuresets[:5000]

testing_set = featuresets[5000:]






mnb_p = open( "./pickle_jar/mnb_classifier.pickle", "rb")
MNB_classifier = pickle.load( mnb_p)
mnb_p.close()



bnb_p = open( "./pickle_jar/bnb_classifier.pickle", "rb")
BNB_classifier = pickle.load( bnb_p)
bnb_p.close()



sgdc_p = open( "./pickle_jar/sgdc_classifier.pickle", "rb")
SGDClassifier_classifier = pickle.load( sgdc_p)
sgdc_p.close()



lsvc_p = open( "./pickle_jar/lsvc_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(lsvc_p)
lsvc_p.close()







lr_p = open( "./pickle_jar/lr_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load( lr_p)
lr_p.close()

voted_classifier =  VoteClassifier(
									SGDClassifier_classifier,
									 
									 LinearSVC_classifier,
									 MNB_classifier,
									 BNB_classifier,
									LogisticRegression_classifier		
				 	)

def sentiment(sentence):
	feats = find_features(sentence)
	return voted_classifier.classify( feats )
