from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.wordnet import information_content

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger

from nltk.collocations import *

from itertools import chain
from math import *

import nltk
import re
import time
import csv
from perceptron import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



import csv

tagger = PerceptronTagger()
ps = PorterStemmer()
wnl = WordNetLemmatizer()
brown_ic = wordnet_ic.ic('ic-brown.dat')

def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def penalty(tag):
    if tag != 'n':
        return 0.75
    return 1

def wordSimilarity(ss1, ss2):
    '''
    ss1 and ss2 are synset object of word 1 and word 2 respectively.
    '''
    a = 0.1
    b = 0.45
    #print("ss1 : {}, ss2 : {}".format(ss1, ss2))
    if ss1._pos != ss2._pos:
        return 0

    l = ss1.shortest_path_distance(ss2, simulate_root=False)
    if l is None:
        l = 0
        h = 0
    else:
        lcs = ss1.lowest_common_hypernyms(ss2, simulate_root=False, use_min_depth=False)
        h = lcs[0].max_depth()
    f1 = exp(-a*l)
    f2 = (exp(b*h) - exp(-b*h))/(exp(b*h) + exp(-b*h))

    return penalty(ss1._pos)*f1*f2



def mostSimilarWord(ss1, synsetList, threshold):
    similarityScore = 0
    mostSimilarWord = None
    stop = set(stopwords.words('english'))
    if ss1 is None:
        return mostSimilarWord, similarityScore
    for ss2 in synsetList:
        if ss2 in stop or synsetList[ss2] is None:
            continue
        s = wordSimilarity(ss1, synsetList[ss2])
        if s > similarityScore:
            mostSimilarWord = synsetList[ss2]
            similarityScore = s
    if similarityScore >= threshold:
        #return mostSimilarWord, similarityScore*information_content(ss1, brown_ic)*information_content(mostSimilarWord, brown_ic)
        return mostSimilarWord, similarityScore
    return mostSimilarWord, similarityScore



def tokenize(sentence):
    tokens = {}
    i = 0
    for word in re.split('[^A-Za-z]', sentence):
        # Ignore short words
        if len(word) < 3 or word in tokens.keys():
            continue
        tokens[word.lower()] = i
        i += 1
    return tokens



def tokenizeSynset(sentence):
    tokenSynset = {}
    context_sentence = re.split('[^A-Za-z]', sentence)
    for word in set(context_sentence):
        # Ignore short words
        if len(word) < 3 or word in tokenSynset.keys():
            continue
        tokenSynset[ps.stem(word.lower())] = lesk(context_sentence, word.lower())
    return tokenSynset



def lesk(context_sentence, word, pos=None, stem=True, hyperhypo=True):
    max_overlaps = 0
    lesk_sense = None

    for ss in wn.synsets(word):
        if pos and ss.pos is not pos:
            continue

        lesk_dictionary = []
        lesk_dictionary += ss.definition().split()
        lesk_dictionary += ss.lemma_names()

        if hyperhypo == True:
            lesk_dictionary += list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))
        if stem == True:
            lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
            context_sentence = [ps.stem(i) for i in context_sentence]
        overlaps = set(lesk_dictionary).intersection(context_sentence)

        if len(overlaps) > max_overlaps:
            lesk_sense = ss
            max_overlaps = len(overlaps)
    return lesk_sense



def wordOrderSimilarity(T1, T2):
    r1 = []
    r2 = []
    T = {}
    stop = set(stopwords.words('english'))

    for t in T1:
        if t in T or t in stop:
            continue
        T[t] = T1[t]
        
    for t in T2:
        if t in T or t in stop:
            continue
        T[t] = T2[t]

    for word in T:
        if word in T1:
            r1.append(T1[word]+1)
        else:
            r1.append(0)
        if word in T2:
            r2.append(T2[word]+1)
        else:
            r2.append(0)

    num = 0
    den = 0
    for i in range(len(r1)):
        num += (r1[i] - r2[i])**2
        den += (r1[i] + r2[i])**2
    score = 1 - sqrt(num)/sqrt(den)
    return score



def mostSimilarWord2(ss1, synsetList, threshold):
    similarityScore = 0
    mostSimilarWord = None
    stop = set(stopwords.words('english'))
    if ss1 is None:
        return mostSimilarWord, similarityScore
    for ss2 in synsetList:
        if ss2 in stop or synsetList[ss2] is None:
            continue
        s = wordSimilarity(ss1, synsetList[ss2])
        if s > similarityScore:
            mostSimilarWord = ss2
            similarityScore = s
    if similarityScore >= threshold:
        #return mostSimilarWord, similarityScore*information_content(ss1, brown_ic)*information_content(mostSimilarWord, brown_ic)
        return mostSimilarWord, similarityScore
    return mostSimilarWord, similarityScore



def wordOrderSimilarity2(s1, s2):
    r1 = []
    r2 = []
    str1 = [ps.stem(w) for w in re.split('[^A-Za-z]', s1) if w != '']
    str2 = [ps.stem(w) for w in re.split('[^A-Za-z]', s2) if w != '']
    T1 = tokenizeSynset(s1)
    T2 = tokenizeSynset(s2)
    T = {}
    stop = set(stopwords.words('english'))
    questions = set(['what', 'who', 'when', 'how', 'why', 'where'])

    for t in T1:
        if t in T or t in stop:
            continue
        T[t] = T1[t]
        
    for t in T2:
        if t in T or t in stop:
            continue
        T[t] = T2[t]

    for word in T:
        if word in T1:
            r1.append(str1.index(word)+1)
        elif word in stop or T[word] is None:
            r1.append(0)
        else:
            mostSimSs, simScore = mostSimilarWord2(T[word], T1, 0.8)
            if mostSimSs is not None:
                r1.append(str1.index(mostSimSs)+1)
            else:
                r1.append(0)
            
        if word in T2:
            r2.append(str2.index(word)+1)
        elif word in stop or T[word] is None:
            r2.append(0)
        else:
            mostSimSs, simScore = mostSimilarWord2(T[word], T2, 0.8)
            if mostSimSs is not None:
                r2.append(str2.index(mostSimSs)+1)
            else:
                r2.append(0)

    num = 0
    den = 0
    for i in range(len(r1)):
        num += (r1[i] - r2[i])**2
        den += (r1[i] + r2[i])**2
    score = 1 - sqrt(num)/sqrt(den)
    return score



def checkCollocation(w1, w2, finder):
    word_filter = lambda *w: w1 not in w and w2 not in w
    finder.apply_ngram_filter(word_filter)
    result = finder.nbest(bigram_measures.likelihood_ratio, 1)
    if len(result) > 0:
        return 1
    return 0


def semanticSimilarity(T1, T2):
    #T1 = tokenizeSynset(sent1)
    #T2 = tokenizeSynset(sent2)
    T = {}
    stop = set(stopwords.words('english'))
    
    for t in T1:
        if t in T or t in stop:
            continue
        T[t] = T1[t]
        
    for t in T2:
        if t in T or t in stop:
            continue
        T[t] = T2[t]
        
    s1 = []
    s2 = []
    for word in T:
        if word in T1:
            s1.append(1)
        else:
            mostSimSs, simScore = mostSimilarWord(T[word], T1, 0.8)
            s1.append(simScore)
        if word in T2:
            s2.append(1)
        else:
            mostSimSs, simScore = mostSimilarWord(T[word], T2, 0.8)
            s2.append(simScore)

    num = 0
    den1 = 0
    den2 = 0
    for i in range(len(s1)):
        num += s1[i] * s2[i]
        den1 += s1[i]**2
        den2 += s2[i]**2
    return num / (sqrt(den1) * sqrt(den2))


def nonMatchingWord(T1, T2):
    stop = set(stopwords.words('english'))
    T = {}
    
    for k in set(T1.keys()) - set(T2.keys()):
        if k not in T and k not in stop:
            T[k] = T1[k]
            
    for k in set(T2.keys()) - set(T1.keys()):
        if k not in T and k not in stop:
            T[k] = T2[k]
    cnt = 0
    for k in T:
        if T[k] is None or T[k]._pos == 'n':
            cnt += 1
        else:
            cnt += 0.5
    return -exp(-cnt)


def extractFeatures(X):
    stop = set(stopwords.words('english'))
    questions = set(['what', 'who', 'when', 'how', 'why', 'where'])
    stop = stop - questions
    #total_word = set(tokens1.keys()).union(set(tokens2.keys())) - stop

    features = []

    for x in X:
        T1 = tokenizeSynset(x[0])
        T2 = tokenizeSynset(x[1])
        #wordOrderScore = wordOrderSimilarity(tokenize(x[0]), tokenize(x[1]))
        wordOrderScore = wordOrderSimilarity2(x[0], x[1])
        semanticScore = semanticSimilarity(T1, T2)
        #nonMatchScore = nonMatchingWord(T1, T2)
        features.append([semanticScore, wordOrderScore])

    return features



X_raw = []
Y = []

filename = 'sample.csv'
csvfile = open(filename,encoding='utf-8')
reader = csv.reader(csvfile)
reader.__next__()  # Ignoring the header
for row in reader:
    X_raw.append([row[3], row[4]])
    Y.append(int(row[5]))

#bigram_measures = nltk.collocations.BigramAssocMeasures()
#finder = BigramCollocationFinder.from_words(wn.words())
#finder.apply_freq_filter(2)

X = extractFeatures(X_raw)


model = perceptron(rate = 0.1, n_iter = 100)
model.fit(X, Y)

Y_predict = list(model.predict(X))

feature1 = []
feature2 = []
for i in range(len(Y_predict)):
    if Y[i] == 0:
        feature1.append(X[i][0])
        feature2.append(X[i][1])

plt.plot(feature1, feature2, "ro")

feature1 = []
feature2 = []
for i in range(len(Y_predict)):
    if Y[i] == 1:
        feature1.append(X[i][0])
        feature2.append(X[i][1])
        
plt.plot(feature1, feature2, "bx")

plt.show()
