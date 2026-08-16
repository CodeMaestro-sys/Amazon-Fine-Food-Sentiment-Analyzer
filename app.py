import joblib
import pandas as pd
import streamlit as st
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer

best_svm = joblib.load("Models/sentiment_svm_model.pkl")
tfidf = joblib.load("Models/tfidf_vectorizer.pkl")

# Load dataset

df = pd.read_csv("data/product_reviews.csv")

# Streamlit configuration

st.set_page_config(
    page_title="Amazon Fine Food Sentiment Analyzer",
    page_icon="🍴",
    layout="wide"
)

st.title("🍴 Amazon Fine Food Sentiment Analyzer")

# Text preprocessing

def preprocessing(text):

    text = text.lower()

    for ch in string.punctuation:
        text = text.replace(ch, "")

    words = text.split()

    stemmed_words = []

    stop_words = set(stopwords.words("english"))

    negations = {"no", "not", "never", "nor"}

    custom_words = stop_words - negations

    for word in words:
        if word not in custom_words:
            stemmed_words.append(word)

    lemmatized_words = []

    lm = WordNetLemmatizer()

    for word in stemmed_words:
        lemmatized_words.append(lm.lemmatize(word))

    return " ".join(lemmatized_words)


# Sidebar

option = st.sidebar.selectbox(
    "Choose Analysis",
    ["Review Analyzer", "Product Analyzer"]
)

# REVIEW ANALYZER

if option == "Review Analyzer":

    st.header("Analyze an Individual Review")

    review = st.text_area(
        "Enter your review:",
        placeholder="Write your food/product review here...",
        height=150
    )

    if st.button("Analyze Review"):

        if not review.strip():

            st.warning("Please enter a review.")

        else:

            cleaned_review = preprocessing(review)

            review_tfidf = tfidf.transform([cleaned_review])

            prediction = best_svm.predict(review_tfidf)[0]

            st.success(f"Predicted Sentiment: {prediction}")


# PRODUCT ANALYZER

else:

    st.header("Product-Level Sentiment Analysis")

    st.write(
        "Enter a Product ID to analyze customer reviews associated "
        "with that product."
    )

    product_id = st.text_input(
        "Enter Product ID:",
        placeholder="Example: B001E4KFG0"
    )

    if st.button("Analyze Product"):

        if not product_id.strip():

            st.warning("Please enter a Product ID.")

        else:

            product_reviews = df[
                df["ProductId"].astype(str).str.strip()
                == product_id.strip()
            ].copy()

            if product_reviews.empty:

                st.error("Product ID not found.")

            else:

                texts = product_reviews["Text"].fillna("").apply(preprocessing)

                review_vectors = tfidf.transform(texts)

                predictions = best_svm.predict(review_vectors)

                product_reviews["Predicted_Sentiment"] = predictions


                sentiment_counts = (
                    product_reviews["Predicted_Sentiment"]
                    .value_counts()
                )

                total_reviews = len(product_reviews)
                positive_count = sentiment_counts.get("Positive", 0)
                neutral_count = sentiment_counts.get("Neutral", 0)
                negative_count = sentiment_counts.get("Negative", 0)

                st.subheader("Product Summary")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Reviews",
                    total_reviews
                )

                col2.metric(
                    "Positive",
                    f"{positive_count / total_reviews * 100:.1f}%"
                )

                col3.metric(
                    "Neutral",
                    f"{neutral_count / total_reviews * 100:.1f}%"
                )

                col4.metric(
                    "Negative",
                    f"{negative_count / total_reviews * 100:.1f}%"
                )

                if positive_count > max(
                    neutral_count,
                    negative_count
                ):

                    overall_sentiment = "Positive"

                elif negative_count > max(
                    positive_count,
                    neutral_count
                ):

                    overall_sentiment = "Negative"

                else:

                    overall_sentiment = "Neutral"


                st.subheader("Overall Sentiment")

                if overall_sentiment == "Positive":
                    st.success("Overall Sentiment: Positive")

                elif overall_sentiment == "Negative":
                    st.error("Overall Sentiment: Negative")

                else:
                    st.warning("Overall Sentiment: Neutral")

                chart_data = pd.DataFrame({
                    "Sentiment": [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ],
                    "Reviews": [
                        positive_count,
                        neutral_count,
                        negative_count
                    ]
                })

                st.subheader("Sentiment Distribution")

                st.bar_chart(
                    chart_data.set_index("Sentiment")
                )

                st.subheader("Product Reviews")

                selected_sentiment = st.selectbox(
                    "Filter Reviews",
                    [
                        "All",
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]
                )

                if selected_sentiment != "All":

                    display_reviews = product_reviews[
                        product_reviews["Predicted_Sentiment"]
                        == selected_sentiment
                    ]

                else:

                    display_reviews = product_reviews


                st.dataframe(
                    display_reviews[
                        ["Text", "Predicted_Sentiment"]
                    ],
                    use_container_width=True
                )