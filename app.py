
import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("language_vectorizer.pkl")

# App title
st.title("AI Language Identifier")

# Description
st.write("Enter a sentence to identify its language.")

# User input
user_text = st.text_area("Enter Your Sentence:")

# Predict button
if st.button("Identify Language"):

    if user_text.strip() == "":
        st.warning("Please enter a sentence.")

    else:
        # Convert text into numerical features
        text_features = vectorizer.transform([user_text])

        # Predict language
        prediction = model.predict(text_features)

        # Display result
        st.success(f"Predicted Language: {prediction[0]}")
