from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.wordnet import information_content

from nltk.stem import PorterStemmer
from itertools import chain
import nltk
import re
import time
from math import *
from itertools import chain
from nltk.corpus import wordnet

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger

tagger = PerceptronTagger()
ps = PorterStemmer()
wnl = WordNetLemmatizer()
brown_ic = wordnet_ic.ic('ic-brown.dat')


#dictionary for tagging parts of speech

def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def penalty(tag):
    if get_wordnet_pos(tag) != 'n':
        return 0.5
    return 1

def wordSimilarity(ss1, ss2):
    '''
    ss1 and ss2 are synset object of word 1 and word 2 respectively.
    '''
    a = 0.2
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
        tokenSynset[word.lower()] = lesk(context_sentence, word.lower())
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



def similarity(wordList1, wordList2, total_word, exclusion):

    # Combined set of distinct words minus the stop words.
    total_word = wordList1.union(wordList2) - exclusion

    temp = set()
    for word in (wordList1 - exclusion):
        temp.add(word)
        for ss in wn.synsets(word):
            temp = temp.union(set(ss.lemma_names()))

    temp = set([ps.stem(i) for i in temp])

    match = 0
    tags = tagger.tag(list(wordList2 - exclusion))
    for wordss in tags:
        word = wordss[0]
        if(wordss[1][0].lower() in ['v', 'n']):
            word = wnl.lemmatize(word, wordss[1][0].lower())
        if ps.stem(word) in temp:
            match += 1

    acc = match/len(total_word)
    return acc



def wordOrderSimilarity(T1, T2):
    r1 = []
    r2 = []
    #T = set(T1.keys()).union(set(T2.keys()))
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
            #r1.append(T1.index(word)+1)
            r1.append(T1[word]+1)
        else:
            #mostSimSs, simScore = mostSimilarWord(T[word], T1, 0.6)
            #r1.append(T1.index(mostSimSs)+1)
            r1.append(0)
        if word in T2:
            #r2.append(T2.index(word)+1)
            r2.append(T2[word]+1)
        else:
            #mostSimSs, simScore = mostSimilarWord(T[word], T2, 0.6)
            #r2.append(T2.index(mostSimSs)+1)
            r2.append(0)

    num = 0
    den = 0
    for i in range(len(r1)):
        num += (r1[i] - r2[i])**2
        den += (r1[i] + r2[i])**2
    score = 1 - sqrt(num)/sqrt(den)
    return score



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
        #print(word)
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



def predict(sent1, sent2, threshold, alpha, beta, gamma):
    stop = set(stopwords.words('english'))
    questions = set(['what', 'who', 'when', 'how', 'why', 'where'])
    stop = stop - questions
    
    tokens1 = tokenize(sent1)
    tokens2 = tokenize(sent2)

    #tokens1 = tokenizeSynset(sent1)
    #tokens2 = tokenizeSynset(sent2)

    total_word = set(tokens1.keys()).union(set(tokens2.keys())) - stop

    wordSimilarity = similarity(set(tokens1.keys()), set(tokens2.keys()), total_word, stop)
    wordOrderScore = wordOrderSimilarity(tokens1, tokens2)
    semanticScore = semanticSimilarity(tokenizeSynset(sent1), tokenizeSynset(sent2))
    #semanticScore = semanticSimilarity(sent1, sent2)

    result = alpha*semanticScore + beta*wordOrderScore + gamma*wordSimilarity
    #result = delta*wordSimilarity + (1 - delta)*wordOrderScore

    if result < threshold:
        return 0, result
    return 1, result



'''
threshold = 0.5
delta = 0.2

sent1 = 'Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?'
sent2 = "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?"
start_time = time.time()
res, score = predict(sent1, sent2, threshold, delta)
print('Predict : {}, score : {}, run time : {} seconds'.format(res, score, time.time() - start_time))

sent1 = "What is the step by step guide to invest in share market in india?"
sent2 = "What is the step by step guide to invest in share market?"
start_time = time.time()
res, score = predict(sent1, sent2, threshold, delta)
print('Predict : {}, score : {}, run time : {} seconds'.format(res, score, time.time() - start_time))


sent1 = "How do I read and find my YouTube comments?"
sent2 = "How can I see all my Youtube comments?"
start_time = time.time()
res, score = predict(sent1, sent2, threshold, delta)
print('Predict : {}, score : {}, run time : {} seconds'.format(res, score, time.time() - start_time))
'''

