#!/usr/bin/env python3
"""
Native Deodorant: Consumer Language Analysis
============================================

What consumers actually say about Native deodorant on social media, how that
compares with its closest competitor (Lume), how interest has moved over time,
and how far consumer vocabulary sits from Native's own marketing language.

Pipeline
--------
1. Load & clean      Merge social posts (Bluesky, YouTube), drop Amazon reviews,
                     remove blanks and duplicates.
2. Score & tag       VADER sentiment per post + keyword-based theme tagging.
3. Native themes     Theme frequency and mean sentiment for Native.
4. Native vs Lume    Theme share and sentiment, side by side.
5. Search & time     Google Trends interest and Native sentiment over time.
6. Vocabulary gap    Consumer language vs Native brand copy.
7. Findings          Printed summary of headline numbers.

Inputs (in ``--data-dir``, default ``./data``)
----------------------------------------------
corpus_clean.csv        Posts/reviews. Columns: review_text, rating, date, source, brand
trends.csv              Weekly Google Trends interest. Columns: date, Native deodorant,
                        Lume deodorant (0-100 scale)
native_brand_copy.txt   Native marketing copy, one line per row

Outputs (in ``--output-dir``, default ``./output``)
---------------------------------------------------
figures/*.png           Charts (8 in total)
tables/*.csv            Coded corpus, theme table, brand comparison, vocabulary gap

Requirements
------------
    pip install pandas>=2.2 numpy matplotlib scikit-learn vaderSentiment

Usage
-----
    python analysis.py
    python analysis.py --data-dir path/to/data --output-dir path/to/output

Methodology notes and limitations
---------------------------------
* Sentiment is VADER's compound score (-1 to +1). Posts scoring >= +0.05 are
  "positive", <= -0.05 "negative", otherwise "neutral" (VADER's standard cutoffs).
  VADER is a general-purpose lexicon model; it can misread sarcasm and
  domain-specific language.
* Themes are tagged by keyword matching (see ``THEMES``). A post can belong to
  several themes, so theme percentages do not sum to 100. Most keywords match
  as word *prefixes* (e.g. "irritat" catches "irritated" and "irritation"),
  which is broad but can occasionally over-match (e.g. "cap" also matches "capable").
* Theme sentiment is the mean score of every post that mentions the theme, so it
  reflects the whole post, not only the sentence about that theme.
* Social posts are a self-selected sample and are not representative of all
  Native customers. Small theme groups (n < 25) are excluded from brand comparison.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent

# Brand colours and chart styling
NATIVE = "#0D9488"
LUME = "#E08E45"
NEUTRAL = "#C8C8C8"
NEGATIVE = "#C0504D"
INK = "#222222"
GRID = "#DDDDDD"

# VADER's conventional thresholds for classifying the compound score
POSITIVE_CUTOFF = 0.05
NEGATIVE_CUTOFF = -0.05

# A theme needs at least this many posts in BOTH brands to be compared
MIN_THEME_N = 25

# Vocabulary-gap filters: keep a word if it appears in >= this % of consumer
# posts or >= this % of brand-copy lines
MIN_CONSUMER_PCT = 2
MIN_BRAND_PCT = 4

# Keyword dictionary used to tag each post with themes.
# Terms starting with a letter match at the start of a word (prefix match);
# terms starting with a symbol (e.g. "$") match literally.
THEMES = {
    "irritation":      ["rash", "burn", "itch", "sting", "red", "reaction", "sensitive",
                        "irritat", "allergic", "breakout", "break out"],
    "efficacy":        ["work", "stopped", "sweat", "wet", "odor", "odour", "lasted", "last",
                        "all day", "stink", "funk", "dry", "protect", "hours"],
    "scent":           ["smell", "fragrance", "scent", "overpowering", "light", "aroma", "overwhelming"],
    "transition":      ["detox", "adjust", "switch", "first week", "aluminum-free",
                        "aluminum free", "natural deodorant"],
    "price_value":     ["expensive", "worth", "price", "cost", "cheaper", "money", "$", "pricey"],
    "ingredients":     ["natural", "clean", "aluminum", "aluminium", "baking soda",
                        "ingredient", "chemical", "shea", "coconut"],
    "ownership_trust": ["p&g", "procter", "acquired", "sold out", "changed formula",
                        "reformul", "lied", "sue", "used to"],
    "packaging":       ["plastic", "container", "twist", "cap", "leak", "knob", "dial",
                        "crumbl", "glob", "stick won"],
}

# Display labels for charts
THEME_LABELS = {
    "efficacy":        "Efficacy",
    "scent":           "Scent",
    "ingredients":     '"Natural" / ingredients',
    "irritation":      "Irritation",
    "transition":      "Transition",
    "price_value":     "Price / value",
    "ownership_trust": "Ownership / trust",
    "packaging":       "Packaging",
}

# Posts matching this pattern look like brand/influencer promotion rather than
# organic consumer talk. They are excluded from the vocabulary-gap analysis.
PROMO = re.compile(
    r"✨|shop now|ready to switch|keeps you feeling|% off|link in|"
    r"available at|www\.|http|#ad|get yours|use code", re.I)

# Words ignored in the vocabulary-gap analysis: standard English stop words plus
# brand names, category words and filler that say nothing about what people mean.
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "native", "deodorant", "deo", "product", "products", "use", "used", "using",
    "get", "got", "like", "would", "really", "one", "im", "dont", "ive", "also",
    "even", "still", "much", "lume", "day", "days", "make", "makes", "made",
    "ll", "ve", "re", "does", "didnt", "doesnt", "every", "single", "just",
    "don", "doesn", "didn", "wont", "cant", "isnt", "gonna",
}


# =============================================================================
# SHARED HELPERS
# =============================================================================

def apply_style() -> None:
    """Set global matplotlib defaults used by every chart."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def labels_for(themes) -> list[str]:
    """Convert theme keys (e.g. 'price_value') to display labels."""
    return [THEME_LABELS[t] for t in themes]


def save_fig(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    """Save a figure to ``fig_dir`` at print quality and release its memory."""
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes, axis: str = "x") -> None:
    """Add light gridlines behind the bars/lines on the chosen axis."""
    grid_axis = ax.xaxis if axis == "x" else ax.yaxis
    grid_axis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# =============================================================================
# STEP 1: LOAD & CLEAN
# =============================================================================

def load_corpus(path: Path) -> pd.DataFrame:
    """Load the raw corpus and keep clean, unique social posts.

    Amazon reviews are excluded so the analysis covers open social conversation
    (Bluesky and YouTube) only. Duplicates are defined as identical text within
    the same brand.
    """
    df = pd.read_csv(path)
    df = df[df["source"] != "Amazon"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["review_text", "brand"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} social posts "
          f"({df['date'].min():%b %Y} to {df['date'].max():%b %Y})")
    print(df.groupby(["brand", "source"]).size().to_string(), end="\n\n")
    return df


# =============================================================================
# STEP 2: SENTIMENT & THEME CODING
# =============================================================================

def sentiment_label(score: float) -> str:
    """Map a VADER compound score to positive / negative / neutral."""
    if score >= POSITIVE_CUTOFF:
        return "positive"
    if score <= NEGATIVE_CUTOFF:
        return "negative"
    return "neutral"


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER compound score (``vader``) and class label (``sentiment``)."""
    analyzer = SentimentIntensityAnalyzer()
    df["vader"] = df["review_text"].apply(lambda s: analyzer.polarity_scores(s)["compound"])
    df["sentiment"] = df["vader"].apply(sentiment_label)
    return df


def build_theme_pattern(terms: list[str]) -> str:
    """Turn a list of keywords into a single regex.

    * "red" is matched as a whole word, otherwise it would hit "bored", "referred", etc.
    * Terms starting with a letter match at a word boundary (prefix match).
    * Other terms (e.g. "$") are matched literally.
    """
    patterns = []
    for term in terms:
        if term == "red":
            patterns.append(r"\bred\b")
        elif re.match(r"^[a-z]", term):
            patterns.append(r"\b" + re.escape(term))
        else:
            patterns.append(re.escape(term))
    return "|".join(patterns)


def tag_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Add one 0/1 column per theme in ``THEMES``. A post can have several themes."""
    text = df["review_text"].str.lower()
    for theme, terms in THEMES.items():
        df[theme] = text.str.contains(build_theme_pattern(terms), regex=True).astype(int)
    return df


# =============================================================================
# STEP 3: NATIVE THEMES
# =============================================================================

def build_theme_table(nat: pd.DataFrame) -> pd.DataFrame:
    """Per-theme post count, share of Native posts, and mean VADER sentiment."""
    rows = []
    for theme in THEMES:
        sub = nat[nat[theme] == 1]
        if len(sub) == 0:
            continue
        rows.append({
            "theme": theme,
            "n": len(sub),
            "pct_of_reviews": len(sub) / len(nat) * 100,
            "mean_vader": sub["vader"].mean(),
        })
    return pd.DataFrame(rows)


def plot_theme_frequency(nat_table: pd.DataFrame, fig_dir: Path) -> None:
    """Horizontal bars: how often each theme is mentioned."""
    data = nat_table.sort_values("pct_of_reviews")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels_for(data["theme"]), data["pct_of_reviews"], color=NATIVE)
    ax.set_xlabel("% of Native posts mentioning theme")
    ax.set_title("Theme frequency — Native", fontweight="bold")
    style_axes(ax, "x")
    save_fig(fig, fig_dir, "theme_frequency.png")


def plot_sentiment_split(nat: pd.DataFrame, fig_dir: Path) -> None:
    """Bars: share of Native posts that are positive / neutral / negative."""
    share = nat["sentiment"].value_counts(normalize=True) * 100
    cats = ["positive", "neutral", "negative"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["Positive", "Neutral", "Negative"],
           [share.get(c, 0) for c in cats],
           color=[NATIVE, NEUTRAL, NEGATIVE])
    ax.set_ylabel("% of Native posts")
    ax.set_title("Native sentiment split", fontweight="bold")
    style_axes(ax, "y")
    save_fig(fig, fig_dir, "sentiment_split.png")


def plot_sentiment_by_theme(nat_table: pd.DataFrame, fig_dir: Path) -> None:
    """Horizontal bars: mean sentiment of posts within each theme."""
    data = nat_table.sort_values("mean_vader")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels_for(data["theme"]), data["mean_vader"], color=NATIVE)
    ax.set_xlabel("mean sentiment (VADER, −1 to +1)")
    ax.set_title("Sentiment by theme — Native", fontweight="bold")
    style_axes(ax, "x")
    save_fig(fig, fig_dir, "sentiment_by_theme.png")


# =============================================================================
# STEP 4: NATIVE VS LUME
# =============================================================================

def plot_theme_share_vs_lume(nat: pd.DataFrame, lume: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bars: share of posts mentioning each theme, by brand."""
    order = sorted(THEMES, key=lambda t: nat[t].mean())
    nat_share = [nat[t].mean() * 100 for t in order]
    lume_share = [lume[t].mean() * 100 for t in order]

    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y + h / 2, nat_share, height=h, color=NATIVE, label="Native")
    ax.barh(y - h / 2, lume_share, height=h, color=LUME, label="Lume")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_for(order))
    ax.set_xlabel("% of posts mentioning theme")
    ax.set_title("Theme share — Native vs Lume", fontweight="bold")
    style_axes(ax, "x")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, fig_dir, "theme_share_vs_lume.png")


def build_comparison(nat: pd.DataFrame, lume: pd.DataFrame, min_n: int = MIN_THEME_N) -> pd.DataFrame:
    """Mean sentiment per theme for each brand.

    Only themes with at least ``min_n`` posts in *both* brands are kept, so that
    averages are not driven by a handful of posts.
    """
    rows = []
    for theme in THEMES:
        nn, ln = int(nat[theme].sum()), int(lume[theme].sum())
        if nn >= min_n and ln >= min_n:
            rows.append({
                "theme": theme,
                "native_sent": nat.loc[nat[theme] == 1, "vader"].mean(),
                "lume_sent": lume.loc[lume[theme] == 1, "vader"].mean(),
                "native_n": nn,
                "lume_n": ln,
            })
    return pd.DataFrame(rows).sort_values("native_sent").reset_index(drop=True)


def plot_sentiment_vs_lume(comparison: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bars: mean sentiment per theme, by brand, with sample sizes."""
    ylabels = [f"{t}\n(N={n}, L={l})"
               for t, n, l in zip(comparison["theme"], comparison["native_n"], comparison["lume_n"])]
    y = np.arange(len(comparison))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(y + h / 2, comparison["native_sent"], height=h, color=NATIVE, label="Native")
    ax.barh(y - h / 2, comparison["lume_sent"], height=h, color=LUME, label="Lume")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("mean sentiment (VADER compound)")
    ax.set_title(f"Sentiment by theme: Native vs Lume (themes with n>={MIN_THEME_N} in both)",
                 fontweight="bold")
    style_axes(ax, "x")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, fig_dir, "sentiment_by_theme_vs_lume.png")


# =============================================================================
# STEP 5: SEARCH INTEREST & SENTIMENT OVER TIME
# =============================================================================

def plot_search_interest(trends_path: Path, fig_dir: Path) -> None:
    """Line chart of weekly Google Trends interest (0-100) for both brands."""
    tr = pd.read_csv(trends_path, index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(tr.index, tr["Native deodorant"], color=NATIVE, lw=2, label="Native")
    ax.plot(tr.index, tr["Lume deodorant"], color=LUME, lw=2, label="Lume")
    ax.set_ylabel("search interest (0-100)")
    ax.set_title("Search interest over time — Native vs Lume", fontweight="bold")
    style_axes(ax, "y")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, fig_dir, "search_interest.png")


def plot_sentiment_over_time(nat: pd.DataFrame, fig_dir: Path) -> None:
    """Native mean sentiment by quarter, smoothed with a 2-quarter rolling mean.

    Quarterly averaging plus smoothing is used because the weekly post count is
    too small for stable week-level estimates.
    """
    series = (nat.dropna(subset=["date"]).sort_values("date")
                 .set_index("date")["vader"]
                 .resample("QE").mean()          # quarter-end frequency (pandas >= 2.2)
                 .rolling(2).mean().dropna())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(series.index, series.values, color=NATIVE, lw=2)
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_ylabel("mean VADER")
    ax.set_title("Native sentiment over time (quarterly, rolling)", fontweight="bold")
    save_fig(fig, fig_dir, "sentiment_over_time.png")


# =============================================================================
# STEP 6: VOCABULARY GAP (CONSUMER LANGUAGE VS BRAND COPY)
# =============================================================================

def word_set(text: str) -> set[str]:
    """Return the unique, meaningful words in ``text``.

    Strips URLs and @mentions, keeps letters only, lower-cases, and drops stop
    words and very short tokens. A set is used so each word counts once per
    post (document frequency, not raw frequency).
    """
    text = re.sub(r"http\S+|www\.\S+|@\w+", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    return {w for w in text.split() if w not in STOPWORDS and len(w) > 2}


def build_vocabulary_gap(nat: pd.DataFrame, brand_copy_path: Path) -> pd.DataFrame:
    """Compare how often words appear in consumer posts vs Native's own copy.

    For each word: ``cust_pct`` = % of consumer posts containing it,
    ``brand_pct`` = % of brand-copy lines containing it, and
    ``diff`` = cust_pct - brand_pct (positive = consumers' word,
    negative = Native's word).
    """
    # Remove promotional / influencer content so only organic talk is measured
    customer_posts = nat[~nat["review_text"].str.contains(PROMO)]
    with open(brand_copy_path, encoding="utf-8") as f:
        brand_lines = [line.strip() for line in f if line.strip()]
    print(f"Vocabulary gap: {len(customer_posts)} consumer posts vs {len(brand_lines)} brand lines")

    cust_sets = [word_set(t) for t in customer_posts["review_text"]]
    brand_sets = [word_set(t) for t in brand_lines]
    vocab = set().union(*cust_sets) | set().union(*brand_sets)

    rows = []
    for word in vocab:
        cust_pct = sum(word in s for s in cust_sets) / len(cust_sets) * 100
        brand_pct = sum(word in s for s in brand_sets) / len(brand_sets) * 100
        if cust_pct >= MIN_CONSUMER_PCT or brand_pct >= MIN_BRAND_PCT:
            rows.append({"term": word, "cust_pct": cust_pct,
                         "brand_pct": brand_pct, "diff": cust_pct - brand_pct})
    return pd.DataFrame(rows)


def plot_vocabulary_gap(vg: pd.DataFrame, fig_dir: Path, top_n: int = 12) -> None:
    """Back-to-back bars: top consumer-only words (right) vs top brand-only words (left)."""
    cust_top = vg.sort_values("diff", ascending=False).head(top_n)
    brand_top = vg.sort_values("diff").head(top_n)

    c = cust_top.sort_values("cust_pct")
    b = brand_top.sort_values("brand_pct", ascending=False)

    # A blank row separates the two groups; brand values are negated to point left
    labels = list(c["term"]) + [""] + list(b["term"])
    cust_vals = list(c["cust_pct"]) + [0] + [0] * len(b)
    brand_vals = [0] * len(c) + [0] + list(-b["brand_pct"])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(y, cust_vals, color=LUME, label="What customers say (% of consumer posts)")
    ax.barh(y, brand_vals, color=NATIVE, label="What Native says (% of brand lines)")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("<- Native's language        share of corpus (%)        customers' language ->")

    # Symmetric x-axis; tick labels show absolute values (left side is negative)
    xmax = max(cust_top["cust_pct"].max(), brand_top["brand_pct"].max()) * 1.1
    ax.set_xlim(-xmax, xmax)
    ticks = [t for t in ax.get_xticks() if -xmax <= t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.0f}" for t in ticks])
    ax.set_xlim(-xmax, xmax)

    style_axes(ax, "x")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("Vocabulary gap: what customers talk about vs what Native markets",
                 fontweight="bold")
    save_fig(fig, fig_dir, "vocabulary_gap.png")


# =============================================================================
# STEP 7: KEY FINDINGS
# =============================================================================

def print_findings(df: pd.DataFrame, nat: pd.DataFrame, lume: pd.DataFrame,
                   nat_table: pd.DataFrame, comparison: pd.DataFrame,
                   vg: pd.DataFrame, fig_dir: Path) -> None:
    """Print the headline numbers in a compact, shareable summary."""
    pos = (nat["vader"] >= POSITIVE_CUTOFF).mean() * 100
    neg = (nat["vader"] <= NEGATIVE_CUTOFF).mean() * 100

    print("\n" + "=" * 70)
    print("Native consumer-language analysis — key findings")
    print("=" * 70)
    print(f"Corpus: {len(df)} posts | Native n={len(nat)} | Lume n={len(lume)}")
    print(f"Date range: {df['date'].min():%b %Y} to {df['date'].max():%b %Y}")
    print(f"Native sentiment: {pos:.0f}% positive, {neg:.0f}% negative, "
          f"mean VADER {nat['vader'].mean():+.2f}")

    print("\nNative themes (share of posts x mean sentiment):")
    for _, r in nat_table.sort_values("pct_of_reviews", ascending=False).iterrows():
        print(f"  {r['theme']:<16}{r['pct_of_reviews']:>5.0f}%   {r['mean_vader']:+.2f}")

    native_warmer = int((comparison["native_sent"] > comparison["lume_sent"]).sum())
    print(f"\nNative vs Lume (themes with n>={MIN_THEME_N} in both): "
          f"Native warmer on {native_warmer} of {len(comparison)} themes.")

    print("\nVocabulary gap — words consumers use that Native does not:")
    print("  " + ", ".join(vg.sort_values("diff", ascending=False).head(8)["term"]))
    print("Words Native uses that consumers do not:")
    print("  " + ", ".join(vg.sort_values("diff").head(8)["term"]))

    print(f"\nFigures written to: {fig_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native deodorant consumer-language analysis.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="folder containing the input files (default: ./data)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output",
                        help="folder for figures and tables (default: ./output)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    # --- 1. Load & clean ---------------------------------------------------
    df = load_corpus(args.data_dir / "corpus_clean.csv")

    # --- 2. Sentiment & themes --------------------------------------------
    df = score_sentiment(df)
    df = tag_themes(df)
    df.to_csv(table_dir / "corpus_coded.csv", index=False)

    nat = df[df["brand"] == "Native"].copy()
    lume = df[df["brand"] == "Lume"].copy()

    # --- 3. Native themes --------------------------------------------------
    nat_table = build_theme_table(nat)
    nat_table.to_csv(table_dir / "native_theme_table.csv", index=False)
    plot_theme_frequency(nat_table, fig_dir)
    plot_sentiment_split(nat, fig_dir)
    plot_sentiment_by_theme(nat_table, fig_dir)

    # --- 4. Native vs Lume -------------------------------------------------
    plot_theme_share_vs_lume(nat, lume, fig_dir)
    comparison = build_comparison(nat, lume)
    comparison.to_csv(table_dir / "theme_comparison.csv", index=False)
    plot_sentiment_vs_lume(comparison, fig_dir)

    # --- 5. Search interest & sentiment over time -------------------------
    plot_search_interest(args.data_dir / "trends.csv", fig_dir)
    plot_sentiment_over_time(nat, fig_dir)

    # --- 6. Vocabulary gap -------------------------------------------------
    vg = build_vocabulary_gap(nat, args.data_dir / "native_brand_copy.txt")
    vg.sort_values("diff", ascending=False).to_csv(table_dir / "vocabulary_gap.csv", index=False)
    plot_vocabulary_gap(vg, fig_dir)

    # --- 7. Findings -------------------------------------------------------
    print_findings(df, nat, lume, nat_table, comparison, vg, fig_dir)


if __name__ == "__main__":
    main()
