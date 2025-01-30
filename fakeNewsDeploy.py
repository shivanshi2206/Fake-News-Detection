import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# List of available models
models = {
    "Multinomial Naive Bayes": "MultinomialNB_pipeline.pkl",
    "Logistic Regression": "LogisticRegression_pipeline.pkl",
    "Decision Tree": "DecisionTreeClassifier_pipeline.pkl",
    "Random Forest": "RandomForestClassifier_pipeline.pkl"
}

# Initialize the Porter Stemmer
portStemmer = PorterStemmer()

# Function to preprocess the title (same as in the training script)
def stemming(content):
    content = str(content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [portStemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Streamlit App
st.title('Fake News Detection')

# Dropdown for model selection
selected_model_name = st.selectbox(
    "Choose a Model",
    list(models.keys())
)

# Input field for the news title
title = st.text_input('Enter News Title:')

# Load the selected model dynamically
selected_model_file = models[selected_model_name]
pipeline = joblib.load(selected_model_file)

# When the button is clicked
if st.button('Predict'):
    if title:
        # Preprocess the input title
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
