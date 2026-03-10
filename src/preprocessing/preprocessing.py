import pandas as pd
import spacy
from textblob import TextBlob
import sys
sys.path.append('..')   # add parent folder

# load data
df_books = pd.read_csv("data/raw/books_data.csv", nrows=100000)
df_reviews = pd.read_csv("data/raw/Books_rating.csv", nrows=100000)

# -----------------------------
# NLP PREPROCESSING
# -----------------------------

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    return " ".join(tokens)

df_reviews["clean_review"] = df_reviews["text"].apply(clean_text)

df_reviews["sentiment"] = df_reviews["clean_review"].apply(
    lambda x: TextBlob(x).sentiment.polarity
) * 2.5 + 2.5

df = df_reviews.merge(df_books[["Title", "authors", "publishedDate", "categories"]], on="Title", how="left")

df[['authors','categories']] = df[['authors','categories']].apply(lambda x: x.str.replace('[', '',).str.replace(']', '',).str.replace("'", ''))

# -----------------------------
# Save File
# -----------------------------

df.to_csv("data/processed/processed_reviews.csv", index=False)
