# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:38:48 2020

@author: Jason
"""
# TLDR
# Receive text
# Clean Text
# Tokenize Sentences, Words
# Weighted frequencies

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
import networkx as nx
nltk.download('stopwords')
tf = TfidfVectorizer()

article_text = '''So, keep working. Keep striving. Never give up. Fall down seven times, get up eight.
Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. 
So, keep moving, keep growing, keep learning. See you at work.'''

# Removing all special characters in sentences and stopwords
# Removing square brackets and extra spaces
# Eliminate duplicate whitespaces using wildcards
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

clean_article_text = re.sub('[^a-zA-z]', ' ', article_text)
clean_article_text = re.sub(r'\s+', ' ', clean_article_text)
sentence_list = nltk.sent_tokenize(article_text)
    

lm = WordNetLemmatizer()
corpus = []
for i in sentence_list:
    text = re.sub('[^a-zA-z]', ' ', i)
    text = text.lower()
    text = text.split()
    
    text = [lm.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)

X = tf.fit_transform(corpus).toarray()

# create df for article
text_df = pd.DataFrame(X,columns=tf.get_feature_names(),index=corpus)

# find cosine similarity for all sentences
similarities = cosine_similarity(X, X)
# creating dataframe for our similarities
similarities_df = pd.DataFrame(similarities, columns=corpus, index=corpus)
# New column for similaritiy totals
similarities_df['total'] = similarities_df.sum(axis=1)
similarities_df2 = similarities_df['total'].sort_values(ascending=False)

# Printing out the top X most similar sentences
text = []
text_str = ''
for i in range(len(similarities_df2)):
    text.append(similarities_df2.index[i])
    text_str = ' '.join([str(elem) for elem in text])
    

# create list to hold avg cosine similarities for each sentence
avgs = []
for i in similarities:
    avgs.append(i.mean())

# find index values of the sentences to be used for summary
top_idx = np.argsort(avgs)

sort = sorted(top_idx)

