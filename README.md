# Toxic-comment-demo
Toxic comment classifier demo

A machine learning web app that predicts whether a comment is toxic, trained on 4,000 human-annotated comments scraped from Reddit, Twitter/X, and YouTube.
🔗 Live demo: https://toxic-comment-demo-zapqvsrscr4rwgnzzu6hwt.streamlit.app/
What this is
Toxic comments — harassment, hate speech, abusive language — don't scale to manual moderation. This project trains and compares 9 model/imbalance-handling combinations (Logistic Regression, Naive Bayes, and Random Forest, each with baseline, class-weight balancing, and SMOTE oversampling) to find the best classifier for catching them, then deploys the winning model as a live, interactive demo.
Model in production: Random Forest with class-weight balancing — best F1-score (0.517) and best overall balance between catching toxic comments and avoiding false alarms, out of all 9 variants tested.
How it works
Text is cleaned: non-alphabetic characters stripped, lowercased, English stopwords removed, and each word reduced to its dictionary root form (lemmatization).
The cleaned text is vectorized with a bag-of-words CountVectorizer fitted during training.
The vectorized comment is passed to the trained Random Forest classifier, which predicts Toxic / Not Toxic.
Files in this repo
File
Purpose
app.py
Streamlit web app — loads the saved model and serves predictions
requirements.txt
Python dependencies for deployment
toxic_rf_model.pkl
Trained Random Forest classifier (class-weight balanced)
toxic_vectorizer.pkl
Fitted CountVectorizer used at training time
Run it locally
Bash
Key finding
The training data was imbalanced (~74% not-toxic, ~26% toxic) — meaning a model that always guessed "not toxic" would already score ~76% accuracy while catching zero real toxic comments. Addressing that imbalance (class-weight balancing beat SMOTE on this dataset) roughly doubled recall across every algorithm tested. Full methodology, results tables, and confusion-matrix-level analysis are in the technical report.
Limitations — read before trusting this in production
This is a bag-of-words classifier: it responds to which words were common in toxic training examples, not to meaning or context. In practice this means:
It reliably catches explicit threats and slurs.
It can miss subtle hostility, sarcasm, or exclusion that uses no obvious "trigger" words.
It can occasionally over-flag harmless sentences that happen to contain a strong signal word (e.g. "I could kill for some jollof rice").
This is a learning/demo project, not a production-grade moderation system.
This isn't finished
Training doesn't end at production — it continues. Every comment someone types into the live demo is a small piece of real-world evidence about where the model actually breaks, in ways the original 800-comment test set can't fully capture. Future improvements under consideration: TF-IDF instead of raw counts, hyperparameter tuning, cross-validation, and a larger/more diverse training set.
Author
Built by Nsisong Sunday — first end-to-end ML deployment, trained in Google Colab and deployed via Streamlit Community Cloud.
