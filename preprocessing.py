# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:30:51 2017

@author: Karolis
"""
import re
import glob as gb
import pandas as pd

import nltk
from nltk.tokenize import TweetTokenizer
stemmer =  nltk.SnowballStemmer('english')
lemmatizer = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english') 


def clean(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def preprocess(imdb_movie_reviews_root):
    labels = {'pos' : 1, 'neg' : 0}
    
    preprocessed = {}
    
    for s in ['train', 'test']:
        if s not in preprocessed:  
            preprocessed[s] = {'review' : [], 'sentiment' : []}
        
        for l in  labels.keys():
            path = 'aclImdb/%s/%s' % (s, l)
            
            for file_path in gb.glob('%s/*' % path):
                with open(file_path, 'r', encoding='utf8') as file:
                    txt = file.read()
                    preprocessed[s]['review'].append(txt)
                    preprocessed[s]['sentiment'].append(labels[l])
                    
    
    return pd.DataFrame(preprocessed['train']), \
           pd.DataFrame(preprocessed['test'])
           
           
def tokenize(text):    
    # Split into sentences first then tokenize each sentence
    # TweetTokenizer is used to preserve the emoticons in the text
    tokens = [word for sentence in nltk.sent_tokenize(text)
              for word in TweetTokenizer().tokenize(clean(sentence))]
    
    # Remove stopwords
    tokens = [token for token in tokens 
              if not token.lower() in stopwords]

    return tokens           

           
def tokenize_n_stem(text):
    return [stemmer.stem(token) for token in tokenize(text)]  
    
    
def tokenize_n_lemmatize(text):
    return [lemmatizer.lemmatize(token) for token in tokenize(text)]  