# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

input_text = st.text_input("Enter text for Sentimental Analysis:")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>', '', text)
        
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
        
    # Tokenize text
    words = nltk.word_tokenize(text)
        
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
        
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
        
    # Join words to form text
    text = ' '.join(words)
    return text

Stopwords = (set(nltk.corpus.stopwords.words("english")))
Stopwords.remove('not')
Stopwords.remove('down')
Stopwords.remove("more")
Stopwords.remove("under")
domain_words=['finnish','russian','finland','russia','swedish','firm','eighteen','months','taking','total','square',
              'eur','million','announcement','day','earlier','glaston','net','third','quarter','dropped','mln','euro',
              'period','april','baltic','countries','eur mn','last','year','million','state',
              'office','msft','orcl','goog','crm','adbe','aapl','afternoon','esi','billion','eurm','third','quarter',
              'half','annually','annualy','first','second','nine','helsinki','omx','year','month','day','indian','india','third'
              ,'fourth','mn','mln','in','eur','euro','months','goods','one','the', 'of', 'in', 'to', 'and', 'a','eur', 'for',
              's', 'is', 'on', 'from', 'will', 'company', 'as', 'mn', 'its', 'with', 'by', 'be', 'has', 'at','it', 'said', 
              'million', 'net', 'year', 'm', 'that', 'was', 'group', 'an', 'mln','new', 'are', 'quarter','this', 'oyj','also',
              'have', 'which', 'first', 'euro', 'today', 'been', 'about', 'helsinki', 'per','total', 'after', 'nokia', 'bank', 
              'based', 'were', 'we', 'than', 'some','or', 'other', 'all', 'one', 'hel' ,'our', 'plc', 'now', 'last', 'their',
              'second', 'ceo', 'pct', 'january', 'into', 'aapl', 'would', 'eurm', 'out', 'part', 'oy','i','september', 'usd',
              'two', 'third','earlier', 'can', 'time', 'billion','had', 'omx','us', 'russia', 'may','annual', 'day', 'both', 
              'tsla','while', 'before','months', 'number', 'march', 'october', 'euros',
              'they','through', 'april']
Stopwords.update(domain_words)
def Text_Processing(Text): 
    Processed_Text = list()
    Lemmatizer = WordNetLemmatizer()

    # Tokens of Words
    Tokens = nltk.word_tokenize(Text)

    for word in Tokens:
        if word not in Stopwords:            
            Processed_Text.append(Lemmatizer.lemmatize(word))            
    return(" ".join(Processed_Text))


vectorizer = pickle.load(open('C:/Users/HP/Desktop/tf_idf_model.pkl','rb'))
model = pickle.load(open('C:/Users/HP/Desktop/mnb_model.plk','rb'))

if st.button('Analyze'):

    # 1. preprocess
    cleaned_text = clean_text(input_text)
    processed_text=Text_Processing(cleaned_text)
    
    # 2. vectorize
    vector_input = vectorizer.transform([processed_text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header('Negative Statement')
    elif result == 1:
        st.header('Neutral statement')
    elif result == 2:
        st.header('Positive statement')
    

