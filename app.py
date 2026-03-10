import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss


st.set_page_config(layout="wide")

st.title("📚 Book Reviews Explorer")

# load data
df = pd.read_csv("data/processed/processed_reviews.csv", nrows=1000)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Filters")

min_reviews = st.sidebar.slider(
    "Minimum number of reviews",
    1, 100, 10
)

use_sentiment = st.sidebar.toggle("Use sentiment analysis rank",value=False)


# -----------------------------
# AGGREGATIONS
# -----------------------------

# avg score per book
book_stats = (
    df.groupby("Title")
    .agg(
        avg_score=("score","mean"),
        avg_sentiment=("sentiment", "mean"),
        n_reviews=("score","count")
    )
)

# avg score per author
author_stats = (
    df.groupby("authors")
    .agg(
        avg_score=("score","mean"),
        n_reviews=("score","count"),
        avg_sentiment=("sentiment", "mean"),
        n_books=("Title", "nunique")
    )
)

# avg score per genre
genre_stats = (
    df.groupby("categories")
    .agg(
        avg_score=("score","mean"),
        n_reviews=("score","count"),
        avg_sentiment=("sentiment", "mean"),
        n_books=("Title", "nunique")
    )
)

book_stats = book_stats.query("n_reviews >= @min_reviews")
author_stats = author_stats.query("n_reviews >= @min_reviews")
genre_stats = genre_stats.query("n_reviews >= @min_reviews")

if use_sentiment:
    filter_custom = ["avg_sentiment","avg_score","n_reviews"]
else:
    filter_custom = ["avg_score","avg_sentiment","n_reviews"]

book_stats = book_stats.sort_values(
        by=filter_custom,
        ascending=[False,False,False]
    )

author_stats = author_stats.sort_values(
        by=filter_custom,
        ascending=[False,False,False]
    )

genre_stats = genre_stats.sort_values(
        by=filter_custom+["n_books"],
        ascending=[False,False,False,False]
    )


# -----------------------------
# TOP BOOKS
# -----------------------------

st.subheader("🏆 Best Rated")

top_books = book_stats.head(10)
top_authors = author_stats.head(20)
top_genres = genre_stats.head(10)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        top_books,
        x="avg_sentiment" if use_sentiment else "avg_score",
        y=top_books.index,
        orientation="h",
        title="Top books"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.bar(
        top_genres,
        x="avg_sentiment" if use_sentiment else "avg_score",
        y=top_genres.index,
        orientation="h",
        title="Top genres"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AUTHOR ANALYSIS
# -----------------------------

st.subheader("✍️ Authors")

fig = px.scatter(
    top_authors,
    x="avg_sentiment" if use_sentiment else "avg_score",
    y="n_reviews",
    hover_name=top_authors.index
)
# fig = px.histogram(
#     top_authors,
#     x=top_authors.index,
#     y="avg_sentiment" if use_sentiment else "avg_score",
#     nbins=20
# )

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# SCORE DISTRIBUTION
# -----------------------------

st.subheader("⭐ Total Reviews By Book")

top_n_scores = book_stats.sort_values(by='n_reviews',ascending=False).head(10)

fig = px.histogram(
    top_n_scores,
    x=top_n_scores.index,
    y="n_reviews",
    nbins=20
)


st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TOP REVIEWERS
# -----------------------------

st.subheader("⭐ Top reviewers")

top_reviewers = df.groupby("profileName").agg(
    n_reviews=("score", "count"),
    avg_score=("score", "mean"),
    avg_sentiment=("sentiment", "mean")
)

# top_reviewers = top_reviewers.query("n_reviews >= @min_reviews")

top_n_scores = top_reviewers.sort_values(by="n_reviews", ascending=False).head(20)

# fig = px.histogram(
#     top_n_scores,
#     x=top_n_scores.index,
#     y="n_reviews",
#     nbins=20
# )
fig = px.scatter(
    top_n_scores,
    x="avg_score" if not use_sentiment else "avg_sentiment",
    y="n_reviews",
    hover_name=top_n_scores.index
)

st.plotly_chart(fig, use_container_width=True)

# # -----------------------------
# # REVIEW SEARCH
# # -----------------------------

# st.subheader("🔎 Search reviews")

# query = st.text_input("Search text")

# if query:

#     results = df[
#         df["text"].str.contains(query, case=False)
#     ].head(10)

#     st.dataframe(
#         results[["Title","score","text"]]
#     )

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------

st.subheader("🔎 Search reviews")

model = SentenceTransformer('all-MiniLM-L6-v2')

df["embedding"] = df["clean_review"].apply(
    lambda x: model.encode(x)
)

embeddings = np.vstack(df["embedding"].values)

index = faiss.IndexFlatL2(384)
index.add(embeddings)

def search_reviews(query, top_k=10):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, top_k)
    return D, I

query = st.text_input("Search reviews")

if query:

    D, I = search_reviews(query, top_k=10)

    rows = []
    for distance, idx in zip(D[0], I[0]):
        rows.append({
            "Similarity with query": distance,
            "review": df.iloc[idx]["text"],
            "user": df.iloc[idx]["profileName"],
            "Title": df.iloc[idx]["Title"],
            "Score": df.iloc[idx]["score"] if not use_sentiment else df.iloc[idx]["sentiment"]
        })

    results = pd.DataFrame(rows)

    st.dataframe(results)
