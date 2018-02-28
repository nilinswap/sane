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
#documents = [ ( list(movie_reviews.words(fileid)), category ) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
#print(documents[0])
documents = []
file_ob = open( "./short_reviews/pos_comm.txt", "r")
pos_st = file_ob.read()
#print(pos_st[:100])
file_ob.seek(0)
#print( pos_lis[0])
pos_lis = file_ob.readlines()
#print( pos_lis[0])
pos_lisa = [ (item.rstrip().lower().split(' '), 'pos') for item in pos_lis ]
file_ob.close()
#print(pos_lisa[5])

file_ob = open( "./short_reviews/neg_comm.txt", "r")
neg_st = file_ob.read()
file_ob.seek(0)
neg_lis = file_ob.readlines()
neg_lisa = [ (item.rstrip().lower().split(' '),'neg') for item in neg_lis ]
file_ob.close()
#print(neg_lisa[5])
documents = pos_lisa + neg_lisa
#print("here", documents[-1])
random.shuffle(documents)

#print(documents[3])
short_neg_words = word_tokenize(neg_st)
short_pos_words = word_tokenize(pos_st)
all_words = short_pos_words + short_neg_words
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print( all_words["stupid"])
word_features = list( all_words.keys())[:5000]
random.shuffle( word_features )
def find_features( document ):

		words = set(document)
		features = {}
		for w in word_features:
			features[w] = (w in words)

		return features

#print((find_features( movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle( featuresets )
training_set = featuresets[:5000]

testing_set = featuresets[5000:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)
'''save_classifier = open( "naivebayes.pickle", "wb")
pickle.dump( classifier, save_classifier)
save_classifier.close()
'''
"""classifier_f = open( "naivebayes.pickle", "rb")
classifier = pickle.load( classifier_f )
classifier_f.close()
"""
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("original NBS accuracy", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train( training_set)
print("MNB NBS accuracy", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train( training_set)
print("BNB NBS accuracy", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)



'''GNB_classifier = SklearnClassifier(GaussianNB())
GNB_classifier.train( training_set)
print("GNB NBS accuracy", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

'''
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

voted_classifier = VoteClassifier(classifier,
									SGDClassifier_classifier,
									 NuSVC_classifier,
									 LinearSVC_classifier,
									 MNB_classifier,
									 BNB_classifier,
									LogisticRegression_classifier		
				 	)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %", voted_classifier.confidence(testing_set[0][0]))
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %", voted_classifier.confidence(testing_set[1][0]))
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %", voted_classifier.confidence(testing_set[2][0]))
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %", voted_classifier.confidence(testing_set[3][0]))
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %", voted_classifier.confidence(testing_set[4][0]))
