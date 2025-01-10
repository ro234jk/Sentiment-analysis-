# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:21:55 2025

@author: adars
"""

# -*- coding: utf-8 -*-
"""
Perform Sentiment Analysis on Indian Stock Market News Articles

This script trains a Naive Bayes classifier for sentiment analysis and applies it to
predict sentiments of news articles. The overall market sentiment (bullish or bearish)
is also calculated based on the predicted sentiments.

@author: Adarsh Pal
"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime, timedelta

# Function to train the Naive Bayes classifier
def train_sentiment_classifier():
    """Train and save the Naive Bayes classifier and vectorizer."""
    data = {
        "News_Article": [
            "Stock market surges after positive economic data release.",
            "Market sees slight decline after rate hike concerns.",
            "Investors remain cautious ahead of Fed meeting.",
            "Rising inflation impacts market sentiment.",
            "Tech stocks rally on optimistic growth projections.",
        ] * 20,  # Simulate larger dataset
        "Sentiment": [1, -1, 0, -1, 1] * 20
    }
    df_train = pd.DataFrame(data)

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    X_vec_train = vectorizer.fit_transform(df_train['News_Article'])

    # Apply TF-IDF transformation
    tfidf = TfidfTransformer()
    X_tfidf_train = tfidf.fit_transform(X_vec_train)

    # Train the Naive Bayes classifier
    nb_clf = MultinomialNB()
    nb_clf.fit(X_tfidf_train, df_train['Sentiment'])

    # Save the classifier and vectorizer
    with open("nb_clf_indian_stock", "wb") as clf_file:
        pickle.dump(nb_clf, clf_file)
    with open("vectorizer_indian_stock", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("Classifier and vectorizer trained and saved successfully!")

# Function to perform sentiment analysis on news articles
def analyze_sentiment():
    """Perform sentiment analysis on news articles."""
    try:
        # Load the dataset of past 100 days' news articles
        file_path = "Indian_Stock_News_Past_100_Days.csv"
        df_news = pd.read_csv(file_path)

        # Load the trained classifier and vectorizer
        nb_clf = pickle.load(open("nb_clf_indian_stock", 'rb'))
        vectorizer = pickle.load(open("vectorizer_indian_stock", 'rb'))

        # Transform the news articles
        X_test = df_news['News_Article']
        X_vec_test = vectorizer.transform(X_test)
        X_vec_test = X_vec_test.todense()

        # Predict sentiments
        df_news['Sentiment'] = nb_clf.predict(X_vec_test)

        # Aggregate sentiment to calculate overall market view
        sentiment_summary = df_news['Sentiment'].value_counts()

        # Determine overall market view
        if sentiment_summary.get(1, 0) > sentiment_summary.get(-1, 0):
            market_view = "Bullish"
        elif sentiment_summary.get(-1, 0) > sentiment_summary.get(1, 0):
            market_view = "Bearish"
        else:
            market_view = "Neutral"

        # Save the results
        df_news.to_csv("Indian_Stock_News_Analyzed.csv", index=False)

        print("Sentiment analysis completed!")
        print("Sentiment Summary:", sentiment_summary.to_dict())
        print("Overall Market View:", market_view)
    except FileNotFoundError as e:
        print(f"Error: {e}\nEnsure the dataset and trained files are available.")

# Main execution
if __name__ == "__main__":
    # Train the classifier (run once to save the model)
    train_sentiment_classifier()

    # Perform sentiment analysis
    analyze_sentiment()
