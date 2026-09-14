# AI Comment Moderator

An AI-powered comment moderation system that analyzes comments and classifies them into different categories.

## Features

* 🤖 AI-based comment classification
* 🛡️ Toxic comment detection
* 🚫 Spam detection
* 📢 Promotional comment detection
* 💬 Normal comment detection
* 🇧🇩 Supports English, Bangla, and Banglish comments
* 📊 Confidence score for predictions
* 📘 Streamlit web interface
* 🔵 Facebook Page integration for comment moderation

## Moderation Categories

The system currently classifies comments into:

* **NORMAL** — Safe and normal comments
* **TOXIC** — Abusive, hateful, threatening, or harmful comments
* **SPAM** — Spam or scam-related comments
* **PROMO** — Promotional or advertising comments

## AI Model

The moderation model uses a combination of:

* Word-level TF-IDF features
* Character-level TF-IDF features
* Logistic Regression
* Balanced class weighting

Combining word and character features helps the model handle different writing styles, including Bangla, Banglish, English, spelling variations, and informal comments.

## Dataset

The project was trained using multiple datasets containing:

* English comments
* Bengali comments
* Banglish comments
* Toxic/hate speech
* Spam
* Promotional messages

The datasets were combined and processed to create the final moderation dataset.

## Project Structure

```text
AI_Comment_Moderator/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── comment_moderator_model.joblib
│   ├── word_tfidf_vectorizer.joblib
│   └── char_tfidf_vectorizer.joblib
│
└── .streamlit/
    └── secrets.toml
```

> `.streamlit/secrets.toml` contains private credentials and should never be uploaded to GitHub.

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Facebook Integration

The project is being developed to allow Facebook Page owners to connect their Pages and use AI to moderate comments.

The planned system will allow authorized Page owners to:

1. Connect their Facebook account.
2. Select a Facebook Page.
3. Retrieve Page comments.
4. Analyze comments using the AI model.
5. Take moderation actions on harmful comments.

## Future Development

Planned improvements include:

* Real-time Facebook comment moderation
* Facebook Webhooks
* Automatic moderation
* Multi-user support
* Business dashboard
* Secure token management
* Cloud deployment
* Subscription-based service

## Disclaimer

This project is under active development. AI predictions may not always be correct, so automated moderation decisions should be tested carefully before being used in production.
