import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer( train_text )
tokenized = custom_sent_tokenizer.tokenize(sample_text)
#print(tokenized)

'''sample_lis = ['swapnil', 'swapniller', 'swapnilled', 'swapnilation']
ps_obj = nltk.stem.PorterStemmer()
print( [ps_obj.stem(word) for word in sample_lis])
''' #stemming

'''lmtizr_obj = nltk.stem.WordNetLemmatizer()
print(lmtizr_obj.lemmatize("better",'a'))''' # lemmatizer example

'''example_sent = " this is to show the use of stop words in practice. These are words like a, the etc which are more of like fillers"
stop_word_set = set( nltk.corpus.stopwords.words("english"))
print( [ word for word in nltk.tokenize.word_tokenize(example_sent) if word not in stop_word_set ])''' # stopword example

def process_content():
	try:
		dic = {}
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag( words )
			#print(tagged[:3])
			#chunkGram = r"""chhunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""#chunking, much more than video is give in actual website
			'''chunkGram = r"""chhunk: {<.*>+}
																			}<R*|PRP|NN>+{""" #chinking	
												chunkParser = nltk.RegexpParser( chunkGram )
												chunked = chunkParser.parse( tagged )
									
												print(chunked)
												chunked.draw()'''
			named_ne = nltk.ne_chunk(tagged)

			#named_ne = nltk.ne_chunk(tagged, binary=True)
			print(named_ne)
			named_ne.draw()

		'''	
			for item in tagged:
				
					if item[0] not in dic:
						dic[item[0]] = set([])
					dic[item[0]].add(item[1])

		for key in dic.keys():
			#if len(dic[key])>1:
			
			if len(dic[key])>2:
				print(key,dic[key])
		'''
	except Exception as e:
		print(str(e))

syns = nltk.corpus.wordnet.synsets("program")

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())

#exqmples
print(syns[0].examples())

synonyms = []
antonyms = []
for syn in nltk.corpus.wordnet.synsets("good"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())
print( set(synonyms))
print(set(antonyms))

w1 = nltk.corpus.wordnet.synset("boat.n.01") # or equivalently nltk.corpus.wordnet.synsets("boat")[0]
w2 = nltk.corpus.wordnet.synset("ship.n.01")
print(w1.wup_similarity(w2))
#process_content()