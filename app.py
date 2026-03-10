import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss


st.set_page_config(layout="wide")

st.title("📚 Book Reviews Explorer")

# --- Data Load / Preprocessing -------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(nrows: int = 1000) -> pd.DataFrame:
    df = pd.read_csv("data/processed/processed_reviews.csv", nrows=nrows)
    df["publishedYear"] = pd.to_datetime(df.get("publishedDate", None), errors="coerce").dt.year
    return df


def _extract_genres(cat: str) -> list[str]:
    if pd.isna(cat) or not isinstance(cat, str):
        return []
    parts = [p.strip() for p in cat.split(",") if p.strip()]
    return parts


df = load_data(nrows=10000)

# build genre list for filters
all_genres = sorted({g for cats in df["categories"].dropna().unique() for g in _extract_genres(cats)})

# -----------------------------
# SIDEBAR / FILTERS
# -----------------------------

st.sidebar.header("Filters")

min_reviews = st.sidebar.slider("Minimum number of reviews", 1, 100, 10)
use_sentiment = st.sidebar.toggle("Use sentiment analysis rank", value=False)

selected_genres = st.sidebar.multiselect("Genres", all_genres)

published_year_min = int(df["publishedYear"].min(skipna=True))
published_year_max = int(df["publishedYear"].max(skipna=True))
selected_years = st.sidebar.slider(
    "Published year range",
    published_year_min,
    published_year_max,
    (published_year_min, published_year_max),
)

# -----------------------------
# FILTERED DATA FOR DASHBOARD
# -----------------------------

df_filtered = df.copy()

if selected_genres:
    mask = np.zeros(len(df_filtered), dtype=bool)
    for genre in selected_genres:
        mask |= df_filtered["categories"].str.contains(genre, case=False, na=False)
    df_filtered = df_filtered[mask]

if selected_years:
    df_filtered = df_filtered[
        df_filtered["publishedYear"].between(selected_years[0], selected_years[1], inclusive="both")
    ]

# -----------------------------
# METRICS ROW
# -----------------------------

st.subheader("📊 Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total reviews", f"{len(df_filtered):,}")
col2.metric("Average rating", f"{df_filtered['score'].mean():.2f}")
col3.metric("Average sentiment", f"{df_filtered['sentiment'].mean():.2f}")
col4.metric("Unique books", f"{df_filtered['Title'].nunique():,}")

# -----------------------------
# AGGREGATIONS
# -----------------------------

# avg score per book
book_stats = (
    df_filtered.groupby("Title")
    .agg(
        avg_score=("score","mean"),
        avg_sentiment=("sentiment", "mean"),
        n_reviews=("score","count")
    )
)

# avg score per author
author_stats = (
    df_filtered.groupby("authors")
    .agg(
        avg_score=("score","mean"),
        n_reviews=("score","count"),
        avg_sentiment=("sentiment", "mean"),
        n_books=("Title", "nunique")
    )
)

# avg score per genre
genre_stats = (
    df_filtered.groupby("categories")
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
    sort_cols = ["avg_sentiment", "avg_score", "n_reviews"]
else:
    sort_cols = ["avg_score", "avg_sentiment", "n_reviews"]

book_stats = book_stats.sort_values(by=sort_cols, ascending=[False, False, False])
author_stats = author_stats.sort_values(by=sort_cols, ascending=[False, False, False])
genre_stats = genre_stats.sort_values(
    by=sort_cols + ["n_books"], ascending=[False, False, False, False]
)

# -----------------------------
# TOP BOOKS + GENRES (CHARTS)
# -----------------------------

st.subheader("🏆 Top Books & Genres")
top_books = book_stats.head(10)
top_genres = genre_stats.head(10)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        top_books,
        x="avg_sentiment" if use_sentiment else "avg_score",
        y=top_books.index,
        orientation="h",
        title="Top books",
        template="plotly_white",
        color="avg_sentiment" if use_sentiment else "avg_score",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(fig, width='stretch')

with col2:
    fig = px.bar(
        top_genres,
        x="avg_sentiment" if use_sentiment else "avg_score",
        y=top_genres.index,
        orientation="h",
        title="Top genres",
        template="plotly_white",
        color="avg_sentiment" if use_sentiment else "avg_score",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(fig, width='stretch')

# -----------------------------
# AUTHOR ANALYSIS
# -----------------------------

st.subheader("✍️ Authors")

fig = px.scatter(
    author_stats.head(30),
    x="avg_sentiment" if use_sentiment else "avg_score",
    y="n_reviews",
    size="n_books",
    hover_name=author_stats.head(30).index,
    template="plotly_white",
    color="avg_sentiment" if use_sentiment else "avg_score",
    color_continuous_scale="Plasma",
)
fig.update_layout(xaxis_title="Score" if not use_sentiment else "Sentiment")
st.plotly_chart(fig, width='stretch')

# -----------------------------
# SCORE DISTRIBUTION
# -----------------------------

st.subheader("⭐ Most Reviewed Books")

top_n_scores = book_stats.sort_values(by="n_reviews", ascending=False).head(10)
fig = px.bar(
    top_n_scores,
    x=top_n_scores.index,
    y="n_reviews",
    title="Top books by number of reviews",
    template="plotly_white",
    color="n_reviews",
    color_continuous_scale="Turbo",
)
fig.update_layout(xaxis_title="Book", yaxis_title="Review count")
st.plotly_chart(fig, width='stretch')


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

fig = px.histogram(
    top_n_scores,
    x=top_n_scores.index,
    y="n_reviews",
    nbins=20
)
# fig = px.scatter(
#     top_n_scores,
#     x="avg_score" if not use_sentiment else "avg_sentiment",
#     y="n_reviews",
#     hover_name=top_n_scores.index
# )

st.plotly_chart(fig, width='stretch')

# -----------------------------
# SEARCH + DOWNLOAD
# -----------------------------

st.subheader("🔎 Explore reviews")

search_title = st.text_input("Search book title")

if search_title:
    matches = df_filtered[df_filtered["Title"].str.contains(search_title, case=False, na=False)].sort_values(by="sentiment" if use_sentiment else "score", ascending=False)
    st.write(f"Found {len(matches):,} matching reviews")
    st.dataframe(matches[["Title", "profileName", "score", "sentiment", "categories", "text"]].head(50))

csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download filtered dataset",
    data=csv,
    file_name="filtered_reviews.csv",
    mime="text/csv",
)

# -----------------------------
# SEMANTIC SEARCH (optional)
# -----------------------------

st.subheader("🧠 Semantic search (experimental)")

@st.cache_data(show_spinner=False)
def build_semantic_index(df: pd.DataFrame):
    model_local = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.vstack(df["clean_review"].apply(lambda x: model_local.encode(x)).values)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    return model_local, idx

model, index = build_semantic_index(df_filtered)

query = st.text_input("Search reviews with semantic similarity")

if query:
    query_vector = model.encode([query])
    D, I = index.search(query_vector, 10)
    results = []
    for distance, idx in zip(D[0], I[0]):
        row = df_filtered.iloc[idx]
        results.append(
            {
                "Similarity": distance,
                "Title": row["Title"], 
                "profileName": row["profileName"],
                "Review": row["text"],
                "Score": row["score"],
                "Sentiment": row["sentiment"],
            }
        )
    st.dataframe(pd.DataFrame(results))

