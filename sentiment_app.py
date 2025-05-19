import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# App UI
st.title("Text Sentiment App")
st.write("Type something to check its sentiment:")

user_input = st.text_input("Your sentence")

if user_input:
    result = sentiment_model(user_input)[0]
    st.write(f"**Sentiment:** {result['label']}")
    st.write(f"**Confidence:** {round(result['score'] * 100, 2)}%")
