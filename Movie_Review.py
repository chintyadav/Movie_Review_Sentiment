import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_model.h5")

model = load_model()

# ------------------------------
# Load tokenizer
# ------------------------------
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 200  # same as training


# ------------------------------
# Prediction function
# ------------------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    pred = model.predict(padded)[0][0]
    sentiment = "Positive 😀" if pred >= 0.5 else "Negative 😡"
    return sentiment, float(pred)


# ------------------------------
# Custom CSS for Dark Theme
# ------------------------------
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cfcfcf;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        background-color: #1c1c1c;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff914d);
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #ff914d, #ff4b4b);
    }
    .result-box {
        background-color: #111111;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 0px 15px rgba(255,75,75,0.4);
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Streamlit UI
# ------------------------------
st.markdown("<h1 class='title'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Type a movie review below and let the AI tell you if it’s Positive or Negative.</p>", unsafe_allow_html=True)

# User input
review = st.text_area("Enter your movie review:")

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        sentiment, score = predict_sentiment(review)
        st.markdown(
            f"""
            <div class="result-box">
                <h2>Prediction Result</h2>
                <p><b>Sentiment:</b> {sentiment}</p>
                <p><b>Confidence Score:</b> {score:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
