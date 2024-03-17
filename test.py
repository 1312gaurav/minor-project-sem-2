# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model

# Download NLTK resources
nltk.download('stopwords')

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')

# Concatenate 'author' and 'title' with a space in between
news_df['content'] = news_df['author'] + " " + news_df['title']

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

X = news_df['content'].values
y = news_df['label'].values

# Vectorize data
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Set page title and icon
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Header and Introduction
st.title("Welcome to the Fake News Detector")
st.write("This tool is designed to help you identify and evaluate the credibility of news articles.")

# Navigation Menu
menu_options = ["Home", "About", "Get Started", "Contact"]
menu_choice = st.sidebar.selectbox("Navigation Menu", menu_options)


# Function for prediction
def prediction(input_text):
    # Preprocess input text
    stemmed_input_text = stemming(input_text)

    # Transform preprocessed text using the same vectorizer
    input_data = vector.transform([stemmed_input_text])

    # Predict using the trained model
    prediction = model.predict(input_data)

    return prediction[0]


# About Section
if menu_choice == "About":
    st.header("About the Fake News Detector")
    st.write(
        "The Fake News Detector is an initiative to combat misinformation and enhance information credibility."
        "Our model is trained to analyze news articles and provide users with insights into the likelihood of the "
        "news being fake or genuine."
    )
    st.write(
        "**Key Features:**\n"
        "- Real-time analysis of news articles\n"
        "- Transparent credibility scores with explanations\n"
        "- Continuous improvement through user feedback"
    )

# Get Started Section
elif menu_choice == "Get Started":
    st.header("Get Started")
    st.markdown(
        "## How It Works\n"
        "1. **Input Article:** Paste the text of a news article you want to analyze.\n"
        "2. **Click the Button:** Our model will process the text and provide a prediction.\n"
        "3. **Interpret Results:** Understand the credibility score and key insights."
    )

    # Input text field
    input_text = st.text_area("Enter news article")

    # Check if input_text is not empty
    if input_text:
        # Prediction and display result
        pred = prediction(input_text)
        if pred == 1:
            st.warning('This is a FAKE NEWS')
        else:
            st.success('This is a REAL NEWS')

# Contact Section
elif menu_choice == "Contact":
    st.header("Contact Us")
    st.write(
        "We appreciate your interest in the Fake News Detector.\n"
        "If you have any questions, feedback, or suggestions, please feel free to reach out to us at ["
        "gauravbhandari016@gmail.com]."
    )

# Home Section (default)
else:
    st.markdown(
        "## Why Detect Fake News?\n"
        "The spread of misinformation can have serious consequences.\n"
        "Our fake news detector is here to assist you in distinguishing between trustworthy and unreliable news "
        "sources."
    )
    st.markdown(
        "## How It Works\n"
        "1. **Input Article:** Paste the text of a news article you want to analyze.\n"
        "2. **Click the Button:** Our model will process the text and provide a prediction.\n"
        "3. **Interpret Results:** Understand the credibility score and key insights."
    )
    st.button("Get Started")

# Footer
st.markdown(
    "***\n"
    "Developed by GAURAV BHANDARI | [GitHub Repository](https://github.com/1312gaurav)"
)
