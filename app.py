import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_dataset():
    data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
    return data


def train_model(data):
    x = np.array(data["Text"])
    y = np.array(data["language"])
    cv = CountVectorizer()
    X = cv.fit_transform(x)
    model = MultinomialNB()
    model.fit(X, y)
    return cv, model


def detect_language(user_input, cv, model):
    user_data = cv.transform([user_input]).toarray()
    output = model.predict(user_data)
    return output[0]


def main():
    st.title("Language Detection App")

    
    data = load_dataset()

    
    cv, model = train_model(data)

    
    user_input = st.text_area("Enter a Text:")

    if st.button("Detect Language"):
        
        detected_language = detect_language(user_input, cv, model)
        st.success(f"Detected Language: {detected_language}")

    
    st.header("Batch Language Detection")
    st.write("Upload a CSV file with a 'text' column for batch language detection.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        
        batch_data = pd.read_csv(uploaded_file)

        
        batch_data = batch_data.dropna(subset=['text'])

        if not batch_data.empty:
            
            batch_data["Detected Language"] = batch_data["text"].apply(lambda x: detect_language(x, cv, model))

            
            st.write("Updated DataFrame with Detected Languages:")
            st.write(batch_data)

            
            st.download_button(
                label="Download Updated CSV",
                data=batch_data.to_csv(index=False),
                file_name="updated_language_detection.csv",
                key="download_button"
            )
        else:
            st.warning("No valid data found in the 'text' column.")

if __name__ == "__main__":
    main()
