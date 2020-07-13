# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:58:09 2020

@author: Jason
"""

import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import regex
#nltk.download('stopwords')
tf = TfidfVectorizer()

def clean_text(article_text, num_sent):
    # Removing all special characters in sentences and stopwords
    # Removing square brackets and extra spaces
    # Eliminate duplicate whitespaces using wildcards
    num_sent = num_sent
    article_text = regex.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = regex.sub(r'\s+', ' ', article_text)
    
    clean_article_text = regex.sub('[^a-zA-z]', ' ', article_text)
    clean_article_text = regex.sub(r'\s+', ' ', clean_article_text)
    sentence_list = nltk.sent_tokenize(article_text)
    word_list = nltk.word_tokenize(clean_article_text)
    tldr(sentence_list, num_sent)
    

def tldr(sentence_list, num_sent):
    num_sent = num_sent
    # TF-IDF
    X = tf.fit_transform(sentence_list).toarray()
    
    # create df for article
    text_df = pd.DataFrame(X,columns=tf.get_feature_names(),index=sentence_list)
    
    # find cosine similarity for all sentences
    similarities = cosine_similarity(X, X)
    
    # creating dataframe for our similarities
    similarities_df = pd.DataFrame(similarities, columns=sentence_list, index=sentence_list)
    
    # New column for similaritiy totals
    similarities_df['total'] = similarities_df.sum(axis=1)
    similarities_df2 = similarities_df['total'].sort_values(ascending=False)
    summary(similarities_df2, num_sent)

def summary(similarities, num_sent):
    num_sent = num_sent
    # Printing out the top X most similar sentences
    text = []
    text_str = ''
    for i in range(0,num_sent):
        text.append(similarities.index[i])
        text_str = ' '.join([str(elem) for elem in text])
    print(text_str)
    
def main():
    # Enter Text
    # Enter number of sentences
    text = input('Enter text: ')
    num_sent = int(input('Enter number of sentences: '))
    clean_text(text,num_sent)

if __name__ == '__main__':
    main()
    
    

