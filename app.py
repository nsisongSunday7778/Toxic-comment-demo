import re
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def lamming(content):
    lemma_content = re.sub('[^a-zA-Z]', ' ', str(content))
    lemma_content = lemma_content.lower()
    lemma_content = lemma_content.split()
    lemma_content = [lemmatizer.lemmatize(word) for word in lemma_content if word not in stop_words]
    lemma_content = ' '.join(lemma_content)
    return lemma_content


@st.cache_resource
def load_artifacts():
    model = joblib.load("toxic_rf_model.pkl")
    vectorizer = joblib.load("toxic_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()


def predict_toxicity(comment):
    cleaned = lamming(comment)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return bool(prediction)


st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️")
st.title("🛡️ Toxic Comment Classifier")
st.write("Type a comment below and the model will predict whether it's toxic.")

user_comment = st.text_area("Enter a comment:", height=100)

if st.button("Check Toxicity"):
    if user_comment.strip() == "":
        st.warning("Please type a comment first.")
    else:
        result = predict_toxicity(user_comment)
        if result:
            st.error("🚨 This comment is predicted TOXIC.")
        else:
            st.success("✅ This comment is predicted NOT toxic.")

st.caption("Model: Random Forest (class-weight) · Trained on Kaggle toxic comments dataset")
