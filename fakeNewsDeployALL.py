import streamlit as st
import joblib
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Initialize the Porter Stemmer
portStemmer = PorterStemmer()

# Function to preprocess text for classical models
def stemming(content):
    content = str(content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [portStemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Streamlit App Title
st.title('Fake News Detection')

# List of available models
models = {
    "Multinomial Naive Bayes": "MultinomialNB_pipeline.pkl",
    "Logistic Regression": "LogisticRegression_pipeline.pkl",
    "Decision Tree": "DecisionTreeClassifier_pipeline.pkl",
    "Random Forest": "RandomForestClassifier_pipeline.pkl",
    "LSTM": "fake_news_lstm_model.keras"
}

# Dropdown for model selection
selected_model_name = st.selectbox(
    "Choose a Model",
    list(models.keys())
)

# Input field for the news title
title = st.text_input('Enter News Title:')

# LSTM-specific initialization
MAX_LEN = 1000
if selected_model_name == "LSTM":
    # Load the LSTM model and tokenizer
    lstm_model = load_model(models[selected_model_name])
    with open('LSTMtokenizer.pkl', 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)

# Predict button logic
if st.button('Predict'):
    if title:
        if selected_model_name == "LSTM":
            # Preprocess the input title for the LSTM model
            sequences = lstm_tokenizer.texts_to_sequences([title])
            padded_seq = sequence.pad_sequences(sequences, maxlen=MAX_LEN)

            # Make prediction using the LSTM model
            prediction = lstm_model.predict(padded_seq)

            # Interpret the prediction
            if prediction[0][0] >= 0.5:
                st.write(f"Prediction ({selected_model_name}): Real News")
            else:
                st.write(f"Prediction ({selected_model_name}): Fake News")
        else:
            # Load the selected classical model pipeline
            selected_model_file = models[selected_model_name]
            pipeline = joblib.load(selected_model_file)

            # Preprocess the input title for classical models
            processed_title = stemming(title)

            # Make predictions using the selected model
            prediction = pipeline.predict([processed_title])

            # Display the prediction
            if prediction == 1:
                st.write(f"Prediction ({selected_model_name}): Fake News")
            else:
                st.write(f"Prediction ({selected_model_name}): Real News")
    else:
        st.write("Please enter a news title to predict.")
