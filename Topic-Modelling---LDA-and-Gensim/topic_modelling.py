#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:45:05 2020

@author: apple
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('words')

words=set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

data=pd.read_excel('instagram_tagged_new.xls')

data=data.sort_values(by='Show')

show_names=data['Show']
show_names=list(set(show_names))
show_names=show_names

captions=[]

for show in show_names:
    for index,value in data.iterrows():
        if value['Show']==show and value['Final_Tagging']=='Owned' and str(value['Caption'])!='nan':
            captions.append(value['Caption'])
            
caption_df=pd.DataFrame(captions,columns=['captions'])

word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem #read

my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^`{|}~•'
latin = 'ãâ°¢¬ëœ€š¤ï¿½‹¯£º«ƒ…¸‚„‚¥¾å“'
#my_punctuation = "(!$%&\'()*+,-./:;<=>?[\\]^`’”“‘{|}~•…¸‚„‚)"
#latin = '(ãâ°¢¬ëœ€š¤ï¿½‹¯£º«ƒ¥¾åðŸ˜‚‡¯²Ž¥ð¿ÿžÿ±™¦¶‰µ)'

def clean_tweet(tweet, bigrams=False):
    tweet2=tweet.split()
    for x in range(0,len(tweet2)-1):
        if tweet2[x][0].isupper() and tweet2[x+1][0].isupper():
            tweet2[x]=tweet2[x]+'_'+tweet2[x+1]
    tweet=' '.join(w for w in tweet2 if len(w)>3)
    tweet=tweet.lower()
    tweet=re.sub('['+ my_punctuation +']+',' ', tweet)
    tweet=re.sub('['+ latin +']+','', tweet)
    tweet=re.sub('\s+',' ', tweet)
    
    #tweet=re.sub('([0-9]+)','', tweet)
    tweet=re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    tweet_token_list=[word for word in tweet.split(' ') if word not in stop_words]
    tweet_token_list=[word_rooter(word) if '@' not in word else word for word in tweet_token_list]
    if bigrams:
        tweet_token_list=tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1] 
        for i in range(len(tweet_token_list)-1)]
        
    tweet=' '.join(tweet_token_list)
    return tweet

caption_df['clean_cap']=caption_df['captions'].apply(clean_tweet)
                      
vectorizer = CountVectorizer(max_df=0.9, min_df=2, token_pattern='\w+|\$[\d\.]+|\S+')

tf=vectorizer.fit_transform(caption_df['clean_cap']).toarray()

tf_feature_names=vectorizer.get_feature_names()

no_of_topics=5

#model=LatentDirichletAllocation(n_components=no_of_topics, random_state=0)
model = NMF(n_components=no_of_topics, random_state=0, alpha=.1, l1_ratio=.5)

model.fit(tf)

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

no_top_words=8
final_df=display_topics(model, tf_feature_names, no_top_words)

###############################################################################

pip install gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet')
stemmer=PorterStemmer()


my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^`{|}~•'
latin = 'ãâ°¢¬ëœ€š¤ï¿½‹¯£º«ƒ…¸‚„‚¥¾å“'

def clean_tweet(tweet):
    if str(tweet) != 'nan':
        tweet=tweet.lower()
        tweet=re.sub('['+ my_punctuation +']+',' ', tweet)
        tweet=re.sub('['+ latin +']+','', tweet)
        tweet=re.sub('\s+',' ', tweet)
        tweet=re.sub('@nickelodeon','', tweet)
        
        #tweet=re.sub('([0-9]+)','', tweet)
        tweet=re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
        return tweet

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

tweet2=lemmatize_stemming(tweet)

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 3:
            result.append(lemmatize_stemming(token))
    return result


caption_df['clean_cap']=caption_df['captions'].apply(clean_tweet)

processed_docs=caption_df['clean_cap'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
count=0
for k,v in dictionary.iteritems():
    print(k,v)
    count+=1
    if count>10:
        break

dictionary.filter_extremes(no_below=2, no_above=0.9, keep_n=60)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

from gensim import corpora, models
tfidf=models.TfidfModel(bow_corpus)
corpus_tfidf=tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


lda_model=gensim.models.LdaMulticore(bow_corpus, num_topics=5,id2word=dictionary, passes=2, workers=2)

for idx,topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))