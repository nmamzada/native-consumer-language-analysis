from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CORPUS_CSV = DATA_DIR / "corpus_clean.csv"
TRENDS_CSV = DATA_DIR / "trends.csv"
BRAND_COPY = DATA_DIR / "native_brand_copy.txt"
SOCIAL_CSV = DATA_DIR / "corpus_social.csv"
CODED_CSV = DATA_DIR / "corpus_coded.csv"
THEME_TABLE_CSV = DATA_DIR / "native_theme_table.csv"
COMPARISON_CSV = DATA_DIR / "theme_comparison.csv"

NATIVE = "#0D9488"
LUME = "#E08E45"
NEUTRAL = "#C8C8C8"
NEGATIVE = "#C0504D"
INK = "#222222"
GRID = "#DDDDDD"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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


def labels_for(themes):
    return [THEME_LABELS[t] for t in themes]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
