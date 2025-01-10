### Sentiment Analysis for Indian Stock Market News

#### Overview
This project implements a sentiment analysis system to evaluate the sentiment of Indian stock market news articles. The system uses a Naive Bayes classifier trained on labeled news articles to predict the sentiment (Positive, Neutral, Negative) of unseen articles. Additionally, it determines the overall market sentiment (Bullish, Bearish, Neutral) based on the aggregated sentiment values.

---

#### Features
1. **Training**: Train a Naive Bayes classifier using labeled stock market news data.
2. **Prediction**: Predict the sentiment of news articles for the past 100 days.
3. **Market View**: Aggregate individual sentiments to determine the overall market view (Bullish, Bearish, Neutral).
4. **CSV Integration**: Input and output datasets are seamlessly stored in CSV files for easy integration and analysis.

---

#### Project Structure
- **`sentiment_analysis_indian_stock.py`**: Main Python script for training the model and performing sentiment analysis.
- **`Indian_Stock_News_Past_100_Days.csv`**: Sample input dataset containing news articles for the past 100 days.
- **`Indian_Stock_News_Analyzed.csv`**: Output dataset with predicted sentiment for each article.
- **`nb_clf_indian_stock`**: Serialized Naive Bayes classifier (generated after training).
- **`vectorizer_indian_stock`**: Serialized vectorizer (generated after training).

---

#### Requirements
- **Python**: Version 3.7+
- **Required Libraries**:
  - `pandas`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

---

#### Usage

1. **Train the Model**
   Run the script to train the Naive Bayes classifier. This step saves the classifier and vectorizer for reuse.
   ```bash
   python sentiment_analysis_indian_stock.py --train
   ```

2. **Perform Sentiment Analysis**
   After training, the script performs sentiment analysis on the news dataset (`Indian_Stock_News_Past_100_Days.csv`) and generates a new file with predicted sentiments.
   ```bash
   python sentiment_analysis_indian_stock.py --analyze
   ```

3. **Check Results**
   The output CSV file (`Indian_Stock_News_Analyzed.csv`) contains:
   - **Date**: Date of the article.
   - **News_Article**: The news article text.
   - **Sentiment**: Predicted sentiment (-1: Negative, 0: Neutral, 1: Positive).

   Additionally, the script prints:
   - **Sentiment Summary**: Count of Positive, Neutral, and Negative articles.
   - **Overall Market View**: Bullish, Bearish, or Neutral.

---

#### Example Output
**Sample sentiment summary**:
```plaintext
Sentiment Summary: {1: 60, 0: 30, -1: 10}
Overall Market View: Bullish
```

---

#### Future Enhancements
- Incorporate **real-time data scraping** for stock market news.
- Extend sentiment analysis to include specific **sectors** or **industries**.
- Use **advanced models** like Transformer-based architectures (e.g., BERT, RoBERTa) for improved accuracy.
- Add **visualizations** to display sentiment trends over time.
- Develop a **web-based UI** for user interaction.

---

#### License
This project is licensed under the **MIT License**.

---

#### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

---

#### Author
**Adarsh Pal**

For questions or collaborations, feel free to reach out.

