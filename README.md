# 🍴 Amazon Fine Food Sentiment Analyzer

An NLP-based sentiment analysis application that analyzes Amazon Fine Food customer reviews and classifies them as **Positive, Neutral, or Negative**.

The project uses **TF-IDF and a tuned Linear SVM** and provides an interactive **Streamlit dashboard** for both individual review analysis and product-level sentiment analysis.

---

## 🚀 Features

- Analyze sentiment of an individual customer review
- Analyze sentiment for a complete product using its Product ID
- View Positive, Neutral, and Negative review percentages
- View overall product sentiment
- Filter reviews by sentiment
- View individual reviews with their predicted sentiment
- Interactive Streamlit dashboard

---

## 🧠 Machine Learning Approach

```text
Customer Review
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Tuned Linear SVM
      ↓
Positive / Neutral / Negative

Models Tested
Model	Accuracy
Logistic Regression	88.92%
Multinomial Naive Bayes	79.15%
Linear SVM	88.96%
Tuned Linear SVM	87.83%

The Tuned Linear SVM was selected as the final model because it provided better balance across the sentiment classes, particularly improving Neutral-class recall.

📊 Dataset

Amazon Fine Food Reviews

The original dataset contains 568,454 customer reviews.

Important columns used:

ProductId
Text
Score

Sentiment mapping:

Score	Sentiment
1–2	Negative
3	Neutral
4–5	Positive

Dataset:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

🛠️ Tech Stack
Python
Pandas
Scikit-learn
NLTK
Streamlit
Joblib
Git & GitHub
📁 Project Structure
Amazon-Fine-Food-Sentiment-Analyzer/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── Models/
│   ├── sentiment_svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── sentiment_dataset/
    └── product_reviews.csv
▶️ Run Locally

Clone the repository:

git clone <YOUR_REPOSITORY_URL>
cd Amazon-Fine-Food-Sentiment-Analyzer

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

💼 Business Use Case

The application is designed as an internal tool for e-commerce companies.

Employees such as product managers or customer-insights teams can enter a Product ID and quickly understand how customers feel about that product without manually reading hundreds of reviews.

The dashboard can help identify:

Products receiving negative feedback
Overall customer sentiment
Distribution of customer opinions
Individual negative reviews requiring attention

🔮 Future Improvements
Aspect-based sentiment analysis
Sentiment trends over time
Identification of common customer complaints
Transformer-based models such as BERT
More advanced product-level insights


👩‍💻 Author
Kasak 