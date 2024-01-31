import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load models
nb_model = pickle.load(open("nb_models.pickle", "rb"))
svm_model = pickle.load(open("svm_models.pickle", "rb"))
rf_model = pickle.load(open("rf_models.pickle", "rb"))
dt_model = pickle.load(open("dt_models.pickle", "rb"))
lr_model = pickle.load(open("lr_models.pickle", "rb"))

# Function to predict sentiment for a given aspect
def predict_sentiment(model, aspect, text):
    # Implement feature extraction based on your model requirements
    vectorizer = CountVectorizer()
    X = vectorizer.transform(text)

    # Predict sentiment
    sentiment = model.predict(X)

    return sentiment

# Function to display prediction results in a table
def display_results(predictions):
    st.table(predictions)

# Function to display prediction results in a plot
def display_plot(predictions):
    for aspect, result in predictions.items():
        plt.figure(figsize=(8, 6))
        plt.bar(result.keys(), result.values())
        plt.title(f"Sentiment Summary for {aspect}")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Sentiment Prediction App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Aspect selection
        selected_aspect = st.selectbox("Select Aspect", df.columns[3:])

        # Model selection
        selected_model = st.selectbox("Select Model", ["nb", "svm", "rf", "dt", "lr"])
        model = None

        if selected_model == "nb":
            model = nb_model
        elif selected_model == "svm":
            model = svm_model
        elif selected_model == "rf":
            model = rf_model
        elif selected_model == "dt":
            model = dt_model
        elif selected_model == "lr":
            model = lr_model

        # Predict sentiment
        predictions = predict_sentiment(model, selected_aspect, df['text'])

        # Display results in a table
        st.subheader("Prediction Results (Table)")
        display_results(predictions)

        # Display results in a plot
        st.subheader("Prediction Results (Plot)")
        display_plot(predictions)

if __name__ == "__main__":
    main()
