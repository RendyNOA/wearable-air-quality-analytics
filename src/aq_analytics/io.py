"""Input validation and output helpers."""
from pathlib import Path

import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC DEMO DATA"
LOCAL_LABEL = "LOCAL DATA — review before sharing"


def read_table(path, required):
    """Read a CSV and fail early with a clear message if columns are missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. See docs/DATA_DICTIONARY.md "
            "or run `python -m scripts.generate_demo_data`."
        )
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    return df


def figure_label(data_dir) -> str:
    """Label stamped on every figure.

    Inputs from a folder named `demo` are synthetic. Anything else is treated
    as the user's own data, so a chart shared out of context still says what
    it is.
    """
    return SYNTHETIC_LABEL if Path(data_dir).name == "demo" else LOCAL_LABEL


def save_table(df, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"  > {filename}")


def save_plot(fig, filename, label=None):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if label:
        # Just above the plot area so it never overlaps axis text;
        # bbox_inches="tight" expands the saved image to include it.
        fig.text(0.5, 1.0, label, ha="center", va="bottom", fontsize=8,
                 color="#b00020", fontweight="bold")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  > {filename}")
