import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Resume Classifier")
st.write("Upload your resume and get the predicted job category!")

# File uploader
uploaded_file = st.file_uploader("Upload your resume (.txt)", type=["txt"])

if uploaded_file is not None:
    # Read file content
    text = uploaded_file.read().decode("utf-8")

    # Transform and predict
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)

    st.subheader("Prediction Result:")
    st.write("This resume is classified as:", "🧑‍💻 Tech" if prediction[0] == 1 else "💼 Non-Tech")
