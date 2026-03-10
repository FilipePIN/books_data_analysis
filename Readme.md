# Book Reviews Dashboard

A simple Streamlit app that visualizes book reader reviews and ratings from a dataset. The app loads processed review data and shows charts and tables to help explore reader sentiment and rating distributions.

## ✅ Features

- Displays book review statistics and distributions
- Interactive charts and tables powered by Streamlit
- Uses cleaned/processed review data from `data/processed/processed_reviews.csv`

## 🚀 Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the preprocessing script in `src/preprocessing/preprocessing.py`

2. Start the app:

```bash
streamlit run app.py
```

3. Open the app in your browser:

http://localhost:8501

## 📂 Project Structure

- `app.py` - Streamlit application entry point
- `src/preprocessing/preprocessing.py` - Data preprocessing script
- `data/raw/` - Raw datasets
- `data/processed/processed_reviews.csv` - Cleaned dataset used by the app

---
