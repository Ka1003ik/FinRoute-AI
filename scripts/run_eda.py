import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_PATH = ROOT / "outputs" / "reports" / "eda_summary.json"

TEXT = "Consumer complaint narrative"
LABEL = "Product"
DATE = "Date received"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        usecols=[TEXT, LABEL, DATE],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    narratives = df[TEXT].str.strip()
    products = df[LABEL].str.strip()
    char_lengths = narratives.str.len()
    word_lengths = narratives.str.split().str.len()
    product_counts = products.value_counts().sort_values(ascending=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    product_counts.sort_values().plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_title("Complaint narratives by product")
    ax.set_xlabel("Number of narratives")
    ax.set_ylabel("Product")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "product_distribution.png", dpi=150)
    plt.close(fig)

    upper_length = word_lengths.quantile(0.99)
    lengths_for_plot = word_lengths[word_lengths <= upper_length]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lengths_for_plot, bins=40, color="#0f766e", edgecolor="white")
    ax.set_title("Narrative length in words (up to the 99th percentile)")
    ax.set_xlabel("Words per narrative")
    ax.set_ylabel("Number of narratives")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "narrative_length.png", dpi=150)
    plt.close(fig)

    dates = pd.to_datetime(df[DATE], errors="coerce", format="mixed")
    monthly_counts = (
        dates.dropna()
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_counts.plot(kind="bar", ax=ax, color="#7c3aed")
    ax.set_title("Narratives by month received")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of narratives")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_narratives.png", dpi=150)
    plt.close(fig)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=30,
    )
    term_matrix = vectorizer.fit_transform(narratives)
    term_counts = term_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    top_terms = sorted(
        zip(terms, term_counts),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    minority_count = int(product_counts.min())
    majority_count = int(product_counts.max())

    report = {
        "rows": int(len(df)),
        "product_category_count": int(product_counts.size),
        "product_counts": {
            str(label): int(count)
            for label, count in product_counts.items()
        },
        "largest_to_smallest_class_ratio": round(
            majority_count / minority_count, 2
        ),
        "narrative_characters": {
            "minimum": int(char_lengths.min()),
            "median": float(char_lengths.median()),
            "mean": round(float(char_lengths.mean()), 1),
            "maximum": int(char_lengths.max()),
        },
        "narrative_words": {
            "minimum": int(word_lengths.min()),
            "median": float(word_lengths.median()),
            "mean": round(float(word_lengths.mean()), 1),
            "maximum": int(word_lengths.max()),
        },
        "duplicate_narrative_rows": int(narratives.duplicated().sum()),
        "date_range": {
            "earliest": str(dates.min().date()),
            "latest": str(dates.max().date()),
        },
        "monthly_counts": {
            str(month): int(count)
            for month, count in monthly_counts.items()
        },
        "top_terms": [
            {"term": str(term), "count": int(count)}
            for term, count in top_terms
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nFigures saved to: {FIGURE_DIR}")
    print(f"Summary saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()