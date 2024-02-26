import numpy as numpy
import pandas as pd 
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json


# Import model
loaded_vectorizer = pickle.load(open('model/vectorizer.pickle', 'rb'))
loaded_model = pickle.load(open('model/predict.model', 'rb'))


def classify_utterance(df):
    
    global loaded_vectorizer, loaded_model
    
    # Make a prediction
    prediction = loaded_model.predict(loaded_vectorizer.transform([df]))
    return prediction[0]

# Load data from JSON file into a DataFrame
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data_list = [json.loads(line) for line in file]
    df = pd.json_normalize(data_list)
    df = df.dropna()
    df = df.head(100)
    return df

# Upload File
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())


def main():
    st.title("Craigslist Category Prediction Apps")
    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])

    if uploaded_file is not None:
        st.success("File successfully uploaded!")

        save_path = "data/test-model.json"  
        save_uploaded_file(uploaded_file, save_path)
        st.success(f"File saved at: {save_path}")

        # Generate Prediction
        with st.spinner("Generating predictions..."):
            json_file_path = 'data/test-model.json'
            df = load_data_from_json(json_file_path)
            df['prediction'] = df['heading'].apply(classify_utterance)

        st.success("Predictions generated successfully!")
        st.write("Predicted categories Top 100:")
        st.write(df)
    
    
    
if __name__ == '__main__':
    main()