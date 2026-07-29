"""
Prepare-everything step: load + clean the raw Letterboxd CSVs, then train and
save the BPE tokenizer on the cleaned review text.

Usage:
    python src/process_data.py
    python src/process_data.py path/to/a.csv path/to/b.csv --vocab-size 2000
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd

from tokenizer import Tokenizer

# Candidate column names, in priority order — the Kaggle dumps don't all use
# the same headers, so we sniff for whichever exists (same approach as
# baseline.py, plus a title column baseline.py doesn't need).
TEXT_CANDIDATES = ["review_text", "review", "text", "content", "comment", "body"]
RATING_CANDIDATES = ["rating", "star_rating", "stars", "score", "user_rating"]
TITLE_CANDIDATES = ["movie", "title", "movie_title", "film", "name"]


def parse_rating(value):
    """Convert a rating to a 0.5-5.0 float.

    The riyosha Kaggle CSVs store ratings as star glyphs - '★★★★★', '★★★½',
    '½' - not numbers. Handle those, plus already-numeric values.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    if "★" in s or "½" in s:
        return s.count("★") + (0.5 if "½" in s else 0.0)
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pick_column(df, candidates, kind):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise SystemExit(
        f"Could not find a {kind} column. Looked for {candidates}.\n"
        f"Columns actually in the file: {list(df.columns)}\n"
        f"Edit the *_CANDIDATES list at the top of process_data.py if needed."
    )


def load_and_clean_data(paths):
    """Load, clean, and combine the raw CSVs into parallel titles/ratings/reviews lists."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        text_col = _pick_column(df, TEXT_CANDIDATES, "review-text")
        rating_col = _pick_column(df, RATING_CANDIDATES, "rating")
        title_col = _pick_column(df, TITLE_CANDIDATES, "title")
        frames.append(
            df[[title_col, rating_col, text_col]].rename(
                columns={title_col: "title", rating_col: "rating", text_col: "text"}
            )
        )
        print(f"  loaded {len(df):>7,} rows from {p}")
    data = pd.concat(frames, ignore_index=True)

    # Clean: parse ratings (handles star glyphs), drop blanks.
    data["rating"] = data["rating"].map(parse_rating)
    data = data.dropna(subset=["text", "rating", "title"])
    data = data[data["text"].astype(str).str.strip().astype(bool)]

    # Normalize scale. Letterboxd is 0.5-5.0, but some scrapers store 1-10.
    if data["rating"].max() > 5:
        print("  detected 1-10 rating scale -> dividing by 2 to get 0.5-5.0")
        data["rating"] = data["rating"] / 2.0
    data = data[(data["rating"] >= 0.5) & (data["rating"] <= 5.0)]
    data = data.reset_index(drop=True)

    titles = data["title"].astype(str).tolist()
    ratings = data["rating"].tolist()
    reviews = data["text"].astype(str).tolist()
    return titles, ratings, reviews


def train_and_save_tokenizer(reviews, vocab_size, save_prefix):
    """Train the BPE tokenizer on the review corpus and persist it to disk.

    Reviews are joined with plain newlines, not special tokens - training
    should never see the rating/title/endoftext markers, only real text.
    """
    corpus = "\n".join(reviews)
    tokenizer = Tokenizer()
    tokenizer.train(corpus, vocab_size)
    tokenizer.save(save_prefix)
    print(f"  trained tokenizer, vocab size {len(tokenizer.vocab)}")
    print(f"  saved to {save_prefix}.model / {save_prefix}.vocab")
    return tokenizer


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="CSV paths (default: data/*.csv)")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument(
        "--out",
        default=os.path.join(data_dir, "tokenizer"),
        help="save_prefix for the tokenizer .model/.vocab files",
    )
    args = parser.parse_args()

    paths = args.paths or sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise SystemExit("No CSVs given and none found in ./data/.")

    print("Loading data...")
    titles, ratings, reviews = load_and_clean_data(paths)
    print(f"Total usable reviews: {len(reviews):,}")

    print("\nTraining tokenizer...")
    train_and_save_tokenizer(reviews, args.vocab_size, args.out)


if __name__ == "__main__":
    main()
