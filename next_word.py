import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(page_title="Next Word Predictor", page_icon="🤖", layout="centered")

# Load files
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

model = load_model("lstm_model.h5")


# Prediction function
def predict_next_word(text):

    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return ""


# Predict multiple words
def generate_text(seed_text, next_words=5):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        prediction = model.predict(token_list, verbose=0)

        predicted_index = np.argmax(prediction)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# UI Title
st.title("🤖 LSTM Next Word Prediction")
st.write("Enter a sentence and the model will predict the next words.")

# User input
input_text = st.text_input("Enter your text")

num_words = st.slider("Number of words to generate", 1, 10, 3)

# Button
if st.button("Generate Prediction"):

    if input_text.strip() == "":
        st.warning("Please enter some text")

    else:

        with st.spinner("Predicting..."):

            result = generate_text(input_text, num_words)

        st.success("Prediction Complete")

        st.subheader("Generated Text")
        st.write(result)