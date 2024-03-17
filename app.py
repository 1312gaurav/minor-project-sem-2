import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')

# Concatenate 'author' and 'title' with a space in between
news_df['content'] = news_df['author'] + " " + news_df['title']

X = news_df['content'].values
y = news_df['label'].values

# Stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Applying stemming function
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title('Fake News Detector')
st.title("Welcome to the Fake News Detector")
input_text = st.text_input("Enter news article")

# Function for prediction
def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]
X
# Check if input_text is not empty
if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('This is a FAKE NEWS')
    else:
        st.write('This is a REAL NEWS')
