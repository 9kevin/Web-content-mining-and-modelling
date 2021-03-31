import streamlit as st
import pandas as pd
import re
import string
from html.parser import HTMLParser
from nltk.corpus import stopwords
import sd_algorithm

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import nltk


sd = sd_algorithm.SDAlgorithm()

# Loading the model
pickle_in = open("newspaper_model.sav", "rb")
model = pickle.load(pickle_in)

# Loading the tokenizer
pickle_n = open("tokenizer.pickle", "rb")
tokenizer = pickle.load(pickle_n)

# Function to get news content from url
def get_content(url):
    sd.url = url
    text = sd.analyze_page()
    return text

# Function turning a list into a string
def listToString(s): 
    str1 = " "   
    return (str1.join(s))

# Function to clean texts
def clean_texts(text):
  # Removing html characters
  text = HTMLParser().unescape(text)
  # Removing urls and hashtags
  text = re.sub(r'https?:\/\/.\S+', "", text)
  text = re.sub(r'#', '', text)
  text = re.sub(r'^RT[\s]+', '', text)
  # Contradiction replacement
  dictionary={"'s":" is","n't":" not","'m":" am","'ll":" will",
           "'d":" would","'ve":" have","'re":" are"}
  for key,value in dictionary.items():
      if key in text:
          text = text.replace(key, value)
  # Convert to lower case
  text = text.lower()
  # Removing stopwords
  nltk.download('stopwords')
  stopwords_eng = stopwords.words('english') 
  text_tokens = text.split()
  text_list=[]
  for word in text_tokens:
      if word not in stopwords_eng:
          text_list.append(word)
  # Remove punctuations
  clean_text = []
  for word in text_list:
      if word not in string.punctuation:
          clean_text.append(word)

  # Turning the list of words into a single string
  clean_text = listToString(clean_text)
  return clean_text

def predict_news(url):
    content = get_content(url)
    content = clean_texts(listToString(content[3]))
    encoded_text = tokenizer.texts_to_sequences([content])
    max_length = 2
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    y_pred = model.predict(padded_text)
    return y_pred


st.title("Classify News Category")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h3 style="color:white;">Insert the link to the article, and I'll do the rest of the magic..</h3>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

result = ""
r = ""
url = st.text_input("Enter the URL of the article", "")

if st.button("Classify news category"):
    result = predict_news(url)
    if result == [0]:
        r = 'Business'
    elif result == [1]:
        r = 'Entertainment'
    elif result == [2]:
        r = 'Politics'
    elif result == [3]:
        r = 'Sports'
st.success('The predicted category of the article is: {}'.format(r))
