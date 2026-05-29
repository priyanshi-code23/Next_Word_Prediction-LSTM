import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max_len
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Create index_to_word dictionary
word_index = tokenizer.word_index

index_to_word = {}

for word, index in word_index.items():
    index_to_word[index] = word


def predictor(model, tokenizer, text, max_len):

    text = text.lower()

    seq = tokenizer.texts_to_sequences([text])[0]

    seq = pad_sequences(
        [seq],
        maxlen=max_len,
        padding="pre"
    )

    pred = model.predict(seq, verbose=0)

    pred_index = np.argmax(pred)

    next_word = index_to_word.get(pred_index, "")

    return next_word


def generate_text(model, tokenizer, seed_text, max_len, n_words):

    generated = seed_text

    for _ in range(n_words):

        next_word = predictor(
            model,
            tokenizer,
            generated,
            max_len
        )

        if next_word == "":
            break

        generated += " " + next_word

    return generated


# Streamlit UI
st.title("Quote Generator")

st.write("Enter starting words and generate a quote.")

seed_text = st.text_input(
    "Enter Seed Text",
    "are you a"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=5,
    max_value=30,
    value=10
)

if st.button("Generate Quote"):

    result = generate_text(
        model,
        tokenizer,
        seed_text,
        max_len,
        num_words
    )

    st.success(result)