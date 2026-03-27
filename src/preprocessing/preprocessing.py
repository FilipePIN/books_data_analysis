import pandas as pd
import spacy
from textblob import TextBlob
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import itertools

sys.path.append('..')   # add parent folder

# -----------------------------
# WORKER FUNCTIONS (top-level for pickling on Windows)
# -----------------------------

TASK_SIZE = 300   # texts per task sent to each worker (keep small to limit tok2vec memory)

def init_nlp_worker():
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_chunk(texts):
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=50):
        tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
        cleaned.append(" ".join(tokens))
    return cleaned

def sentiment_chunk(texts):
    return [TextBlob(text).sentiment.polarity * 2.5 + 2.5 for text in texts]

def chunked(iterable, size):
    """Split a list into chunks of given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    OUTPUT_PATH = "data/processed/processed_reviews.csv"
    BOOKS_PATH  = "data/raw/books_data.csv"
    REVIEWS_PATH = "data/raw/Books_rating.csv"
    CSV_CHUNK_SIZE = 100_000   # rows read from disk at a time

    n_workers = max(1, cpu_count() - 2)

    # Load books once (smaller file)
    df_books = pd.read_csv(BOOKS_PATH, usecols=["Title", "authors", "publishedDate", "categories"])

    # Count total rows for progress bar (fast scan)
    total_rows = sum(1 for _ in open(REVIEWS_PATH, encoding="utf-8")) - 1
    print(f"Using {n_workers} workers for {total_rows} reviews...")

    # Remove previous output if exists so we can append cleanly
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    first_batch = True

    with Pool(n_workers, initializer=init_nlp_worker) as nlp_pool, \
         Pool(n_workers) as sent_pool, \
         tqdm(total=total_rows, desc="Processing reviews", unit="rows") as pbar:

        reader = pd.read_csv(REVIEWS_PATH, chunksize=CSV_CHUNK_SIZE)

        for df_chunk in reader:
            # --- NLP ---
            texts = df_chunk["text"].fillna("").str.lower().tolist()
            tasks = list(chunked(texts, TASK_SIZE))

            nlp_results = list(nlp_pool.imap(clean_chunk, tasks))
            df_chunk["clean_review"] = list(itertools.chain.from_iterable(nlp_results))

            # --- Sentiment ---
            clean_texts = df_chunk["clean_review"].tolist()
            sent_tasks  = list(chunked(clean_texts, TASK_SIZE))

            sent_results = list(sent_pool.imap(sentiment_chunk, sent_tasks))
            df_chunk["sentiment"] = list(itertools.chain.from_iterable(sent_results))

            # --- Merge with books ---
            df_chunk = df_chunk.merge(
                df_books[["Title", "authors", "publishedDate", "categories"]],
                on="Title",
                how="left"
            )

            for col in ["authors", "categories"]:
                if col in df_chunk.columns:
                    df_chunk[col] = (
                        df_chunk[col]
                        .astype(str)
                        .str.replace("[", "", regex=False)
                        .str.replace("]", "", regex=False)
                        .str.replace("'", "", regex=False)
                    )

            # --- Write incrementally ---
            df_chunk.to_csv(
                OUTPUT_PATH,
                mode="a",
                header=first_batch,
                index=False
            )
            first_batch = False

            pbar.update(len(df_chunk))

    print(f"Done! Output saved to {OUTPUT_PATH}")
