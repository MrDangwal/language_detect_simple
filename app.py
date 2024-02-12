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

    # Load the dataset
    data = load_dataset()

    # Train the model
    cv, model = train_model(data)

    # User input text
    user_input = st.text_area("Enter a Text:")

    if st.button("Detect Language"):
        # Detect language
        detected_language = detect_language(user_input, cv, model)
        st.success(f"Detected Language: {detected_language}")

    # Batch Language Detection
    st.header("Batch Language Detection")
    st.write("Upload a CSV or Excel file for batch language detection.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.type == "application/vnd.ms-excel":
            batch_data = pd.read_excel(uploaded_file, engine='openpyxl')  # Specify the engine for Excel files
        else:
            batch_data = pd.read_csv(uploaded_file)

        # Show all columns in the file
        st.write("Columns in the uploaded file:")
        selected_column = st.selectbox("Select the 'text' column:", batch_data.columns)

        if st.button("Proceed"):
            if selected_column in batch_data.columns:
                # Fill blanks in the selected column with "Not Available"
                batch_data[selected_column].fillna("Not Available", inplace=True)

                # Add a new column for detected languages
                batch_data["Detected Language"] = batch_data[selected_column].apply(lambda x: detect_language(x, cv, model))

                # Display the updated dataframe with detected languages
                st.write("Updated DataFrame with Detected Languages:")
                st.write(batch_data)

                # Download the updated CSV file
                st.download_button(
                    label="Download Updated File",
                    data=batch_data.to_csv(index=False) if uploaded_file.type != "application/vnd.ms-excel" else None,
                    file_name="updated_language_detection.csv",
                    key="download_button"
                )
            else:
                st.warning("No valid 'text' column found in the selected column.")

if __name__ == "__main__":
    main()
