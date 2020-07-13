# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 00:20:02 2020

@author: Jason
"""
from flask import Flask,render_template,url_for,request
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
import flask
import regex
nltk.download('stopwords')
tf = TfidfVectorizer()

# initialize the flask app
app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# set up the main route
@app.route('/', methods=['POST'])
def main_1():
    if flask.request.method == 'POST':
        article_text=request.form['article_text']
        article_text = regex.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = regex.sub(r'\s+', ' ', article_text)          
        clean_article_text = regex.sub('[^a-zA-z]', ' ', article_text)
        clean_article_text = regex.sub(r'\s+', ' ', clean_article_text)
        sentence_list = nltk.sent_tokenize(article_text)
        word_list = nltk.word_tokenize(clean_article_text)           
        X = tf.fit_transform(sentence_list).toarray()
        text_df = pd.DataFrame(X,columns=tf.get_feature_names(),index=sentence_list)
        similarities = cosine_similarity(X, X)            
        similarities_df = pd.DataFrame(similarities, columns=sentence_list, index=sentence_list)
        similarities_df['total'] = similarities_df.sum(axis=1)
        similarities_df2 = similarities_df['total'].sort_values(ascending=False)
        text = []
        text_str = ''
        for i in range(0,3):
            text.append(similarities_df2.index[i])
            text_str = ' '.join([str(elem) for elem in text])            
    return render_template('index.html',text_str=text_str)

if __name__ == '__main__':
    app.run(debug=True)        
    
