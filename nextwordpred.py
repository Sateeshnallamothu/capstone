#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:12:27 2017

@author: qanv
"""
#https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
import pandas as pd
##########################3
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import glob
import pickle as pkl
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import unicodedata
import re, string
from gensim import corpora, models
from gensim.models import word2vec
#import gensim
import os
from numpy.random import randn
import pandas as pd
import datetime
import math,re, string
from random import randint
 
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report,roc_curve, f1_score,auc
from sklearn.decomposition import PCA
from datetime import datetime
import pickle 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import naive_bayes
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import LineTokenizer, WhitespaceTokenizer,sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, Counter
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer() 
#############
 
def clean_text(review,sw_flag = False, stem_flag= False):
    ## split text into sentences, remove numbers, special char etc and convert to tokens
    ## if sw_flat is true, remove stop words
    ## if stem_flag is true, stem the words
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import LineTokenizer, WhitespaceTokenizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    import re, string
    
    sentences = tokenizer.tokenize(review)
    #sentences = LineTokenizer().tokenize(review)
    tokens = []
    for sentence in sentences:
        if (len(sentence) > 0) :
            line_str = ''
            for char in sentence:
                char = char.replace('-',' ')
                if (char in string.ascii_letters) or char == ' ':
                    line_str += char
            ## remove non char and non numbers
            line_str = re.sub('[^a-zA-Z0-9\']',' ',line_str)
            ## convert to lower and split words
            words = line_str.lower().split()
            if sw_flag:
                words = [w for w in words if not w in stopwords]
            if stem_flag:
                stemmedwords = [p_stemmer.stem(i) for i in words]
                words = [w for w in stemmedwords ]
            ## append to tokens
            [tokens.append(w) for w in words if len(w) > 1]         
    return tokens

def get_idx(vocab):
    word_index = dict((word,idx) for idx, word in enumerate(vocab))
    index_word = dict((idx,word) for word, idx in word_index.items())
    return word_index,index_word

fnames = ['C:\\Users\\SateeshSwathi\\Documents\\Rscripts\\Capstone\\final\\en_US\\en_US.blogs.txt',\
      'C:\\Users\\SateeshSwathi\\Documents\\Rscripts\\Capstone\\final\\en_US\\en_US.news.txt',\
      'C:\\Users\\SateeshSwathi\\Documents\\Rscripts\\Capstone\\final\\en_US\\en_US.twitter.txt']
#cmtdata = open(fn,'r',encoding='utf8').readlines() 

dictionary = defaultdict(dict)
dictionary['<unk>'] = 0
dict_ix = 0
lines_in_token = []
all_tokens=[]
for fn in fnames:
    with open(fn,'r',encoding='utf8') as f:
        for cnt,line in enumerate(f):
            #if cnt > 10000:
            #    break
            #if cnt < 10000:
            if random.random() > .95:    ## select 15% of the file
                ## create dictionary
                line_in_token = []
                words=clean_text(line)
                for w in words:
                    idx = dictionary.get(w)
                    if idx is None:
                        dict_ix += 1
                        dictionary[w] = dict_ix 
                        idx = dict_ix
                    line_in_token.append(idx)  
                    all_tokens.append(idx)
                #print(words)
                lines_in_token.append(line_in_token)
idx2word = dict((idx,word) for word, idx in dictionary.items())        
#comments = [l.strip().lower() for l in cmtdata]
#dfc=pd.DataFrame(comments,columns=['original_txt'])
#dfc.insert(dfc.shape[1],'clean_txt',dfc.original_txt.apply(clean_text))
import sys
sys.getsizeof(all_tokens) 
 
from nltk.util import ngrams
#bigrams = list(zip(all_tokens, all_tokens[1:]))
#trigrams = list(zip(all_tokens,all_tokens[1:],all_tokens[2:]))
dist_ugs=Counter(w for w in all_tokens)

#bigramcounts=Counter(w for w in bigrams)
bgs=nltk.bigrams(all_tokens)
dist_bgs=nltk.FreqDist(bgs)
#dist_bgs2 = dict((key,value) for key,value in dist_bgs.items() if value > 1)
tgs=nltk.trigrams(all_tokens)
dist_tgs=nltk.FreqDist(tgs)
fgs=nltk.ngrams(all_tokens,4)
dist_fgs=nltk.FreqDist(fgs)
del all_tokens
prob_table_bi = defaultdict(dict)
for key,value in dist_bgs.items():
    prob_table_bi[key[0]][key[1]] = dist_bgs[key]/dist_ugs[key[0]]
del dist_bgs 
kn=nltk.KneserNeyProbDist(dist_tgs)
prob_table_kn2 = defaultdict(dict)
for gram in kn.samples():
    prob_table_kn2[gram[:2]][gram[2]] = kn.prob(gram)  
del kn
### keep only 5 words per combination. this logic will convert the nexted dict to list.
for key,value in prob_table_kn2.items():
    prob_table_kn2[key] = sorted(value.items(),key=lambda x:x[1],reverse=True)[:4]
for key,value in prob_table_bi.items():
    prob_table_bi[key] = sorted(value.items(),key=lambda x:x[1],reverse=True)[:4]
#prob_table_f = defaultdict(dict)
#for key,value in dist_fgs.items():
#    trykey = key[:3]
#    prob_table_f[trykey][key[3:]] = dist_fgs[key]/dist_tgs[trykey]

del dist_tgs
del dist_fgs
#import gc
import pickle
import os
import json
os.chdir('C:\\Users\\SateeshSwathi\\Documents\\Python Scripts\\')
#ngram_dict = [dictionary,idx2word,prob_table_kn2,prob_table_bi]
pickle.out = open("ngramdict.pickle","wb")
pickle.dump((dictionary,idx2word,prob_table_kn2,prob_table_bi,dist_ugs),pickle.out)
pickle.out.close()
with open("ngramdict.json","w") as jf:
    json.dump(dictionary,jf)
#with open("ngramkn2.json","w") as jf:
#    json.dump(prob_table_kn2,jf)
del dictionary,idx2word,prob_table_kn2,prob_table_bi,dist_ugs
####  load data
dictionary,idx2word,prob_table_kn2,prob_table_bi = pickle.load(open("ngramdict.pickle","rb"))

### convert the dict to csv file
with open('kn2dict.csv', 'w') as f:
    for key,value in prob_table_kn2.items():
        row_lst =[]
        row_lst.append(key[0])
        row_lst.append(key[1])
        for k,v in value:
             row_lst.append(k)
             row_lst.append(v) 
        row = ','.join(map(str,row_lst))
        f.write(row+'\n')
with open('bidict.csv', 'w') as f:
    for key,value in prob_table_bi.items():
        row_lst =[]
        row_lst.append(key)
        for k,v in value:
             row_lst.append(k)
             row_lst.append(v) 
        row = ','.join(map(str,row_lst))
        f.write(row+'\n')      
with open('idx2worddict.csv', 'w') as f:
    for key,value in idx2word.items():
        row_lst =[]
        row_lst.append(key)
        row_lst.append(value)
        row = ','.join(map(str,row_lst))
        f.write(row+'\n') 
with open('dictdict.csv', 'w') as f:
    for key,value in dictionary.items():
        row_lst =[]
        row_lst.append(key)
        row_lst.append(value)
        row = ','.join(map(str,row_lst))
        f.write(row+'\n')
with open('ugdict.csv', 'w') as f:
    for key,value in dist_ugs.items():
        row_lst =[]
        row_lst.append(key)
        row_lst.append(value)
        row = ','.join(map(str,row_lst))
        f.write(row+'\n')
        
w1, w2 =  'have', 'some'
w1, w2 = 'some', 'self'
w1, w2 = 'self','editing'
w1, w2 = 'editing','for'
w1, w2 = 'for','my'
w1, w2 = 'my','grandchildren'
w1, w2 = 'grandchildrenx','who'
w1, w2 = 'i','would'
w1, w2 = 'about','his'
w1, w2 = 'monkeys', 'this'  #
w1, w2 = 'reduce','your'  #
w1, w2 = 'take','a'
w1, w2 = 'settle','the'
w1, w2 = 'in','each'  #
w1, w2 = 'to','the' #
w1, w2 ='from','playing' #
w1, w2 ='adam','sandler'  #
import os
os.chdir('C:\\Users\\SateeshSwathi\\Documents\\Python Scripts\\')
import getnextword as gn
gn.pred_next(w1,w2)

    
    

#fourgrams=nltk.collocations.QuadgramCollocationFinder.from_words(all_tokens)
#for fourgram, freq in fourgrams.ngram_fd.items():  
#       print (fourgram, freq)

#https://github.com/MeghanaBandaru/2017-summer-internship/blob/master/quad_prob_model_test.py

##  counts P(Wi/Wi-2 Wi-1) = Count(Wi-2 Wi-1 Wi) / Count(Wi-2 Wi-1)  Markove law with trigrams

for key in dist_fgs:
    trikey=key[:3]
    dist_fgs[key] = dist_fgs[key]/dist_tgs[trikey]  ### this will give probability of each quadgram

prob_table_fgs = defaultdict(dict)
prob_table_tgs = defaultdict(dict)
prob_table_bgs = defaultdict(dict)
prob_table_ugs = defaultdict(dict)

# using quadgram table, contruct a probability table for each three words in the quardgram. 
## e.g fourgram word will be converted to tri:{word1:c, word2:c... wordn:c}

for quad,val in dist_fgs.items():
    trikey=quad[0:3]
    prob_table_fgs[trikey][quad[3]] = val

for tri,val in dist_tgs.items():
    bikey = tri[:2]
    #if dist_bgs[bikey] == 0:
    #    print(tri,dist_bgs[bikey],bikey)
    dist_tgs[tri] = dist_tgs[tri]/dist_bgs[bikey]
    prob_table_tgs[bikey][tri[2]] = dist_tgs[tri]

for bi, val in dist_bgs.items():
    unikey = bi[0]
    dist_bgs[bi] = dist_bgs[bi]/dist_ugs[unikey]
    prob_table_bgs[unikey][bi[1]] = dist_bgs[bi]    


list(map(lambda x:x[0],sorted(prob_table_bgs['happy'].items(),key=lambda x:x[1],reverse=True)))

##  with nltk and Kneser Ney smooting.
gut_ngrams = (ngram for sent in comments for ngram in ngrams(sent.split(),3,pad_left=True,
                                                             pad_right=True,
                                                             right_pad_symbol='EOS',
                                                             left_pad_symbol='BOS'))
 
freq_dist = nltk.FreqDist(gut_ngrams)  ### use only trigrams
kneser_ney = nltk.KneserNeyProbDist(freq_dist)

prob_sum = 0
for i in kneser_ney.samples():
    if i[0] == "keep" and i[1] == "up":
        prob_sum += kneser_ney.prob(i)
        print ("{0}:{1}".format(i, kneser_ney.prob(i)))
print (prob_sum)

## convert the prob into dict of dict with 
prob_table_kn = defaultdict(dict)
for gram in kneser_ney.samples():
    prob_table_kn[gram[:2]][gram[2]] = kneser_ney.prob(gram)
list(map(lambda x:x[0],sorted(prob_table_kn['you','for'].items(),key=lambda x:x[1],reverse=True)))   
## check this for Kneser-Ney https://github.com/smilli/kneser-ney/blob/master/kneser_ney.py

all_review_tokens =[tkns for sen in reviewtokens for tkns in sen]
dist_trigram = nltk.FreqDist(ngrams(all_review_tokens,3))
kn=nltk.KneserNeyProbDist(dist_trigram)
prob_table_kn2 = defaultdict(dict)
for gram in kn.samples():
    prob_table_kn2[gram[:2]][gram[2]] = kn.prob(gram)  
############## w2v embedding

# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count     keep all words..                      
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-1   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model_c = word2vec.Word2Vec(dfc.clean_txt, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model_c.init_sims(replace=True)
model_c_vocablen = len(model_c.wv.vocab)
model_cvocab = model_c.wv.vocab
model_c.wv.syn0.shape
word_index, index_word = get_idx(list(model_cvocab))
def copy_w2v(model,word2idx,embedding_dim):
    vocab_size = len(word2idx)
    embedding_matrix = np.empty((len(word2idx),embedding_dim),dtype=np.float64)
    for ix,w in enumerate(word2idx):
        embedding_matrix[ix,:] = list(model[w])
    return embedding_matrix

embedding_matrix = copy_w2v(model_c,word_index,embedding_dim=num_features)
## convert the text/tokens into dictionary index,

txt_to_dict = []
for tokens in dfc.clean_txt:
    for t in tokens:
        ix = word_index.get(t)
        if ix is not None:
            txt_to_dict.append(ix)

max_seq_words = 5
sequences = []
next_word = []
for i in range(0,len(txt_to_dict) - max_seq_words):
    sequences.append(txt_to_dict[i: i+max_seq_words])
    next_word.append(txt_to_dict[i+max_seq_words])

seqdf=np.array(pd.DataFrame(sequences)) 
nwdf=np.empty((len(next_word),num_features))
for ix,tkn in enumerate(next_word):
    nwdf[ix,:]=embedding_matrix[tkn]      
#xre=np.reshape(seqdf,(len(seqdf),max_seq_len,1))

###  here you go  https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

rnn_comments = Sequential()
rnn_comments.add(Embedding(len(word_index),num_features,
                           weights=[embedding_matrix],
                           input_length=max_seq_words,
                           input_shape=seqdf.shape[1:]))
rnn_comments.add(LSTM(512,return_sequences=False))
rnn_comments.add(Dense(model_c.wv.syn0.shape[1]))
rnn_comments.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
rnn_comments.compile(loss='categorical_crossentropy', optimizer=optimizer)
#rnn_comments.compile(loss='mse', optimizer='sgd')

## lets fit
rnn_comments.fit(seqdf,nwdf,batch_size=128,epochs=2)

start_tkn = random.randint(0,len(txt_to_dict) - max_seq_words -1)
test_sen = txt_to_dict[start_tkn:start_tkn+max_seq_words]
testdf = np.empty((1,len(test_sen)))

rnn_prob=rnn_comments.predict(testdf,verbose=0)[0]
topn = 2;
most_similar_words = model_c.most_similar( positive=[ rnn_prob], negative=[], topn=topn)
most_similar_words
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #probas = np.random.multinomial(1, preds, 1)
    #return np.argmax(probas)
    return preds
    
diversity=.8
next_index = sample(rnn_prob, diversity)