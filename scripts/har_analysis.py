"""Human Activity Recognition (HAR) from wearable IMU signals.

Pipeline: raw accelerometer/gyroscope -> sliding windows -> hand-crafted
features -> three classifiers, evaluated with leave-one-subject-out (LOSO).

Why LOSO: windows overlap and windows from the same person look alike, so a
random train/test split lets the model "recognise the person" rather than
the activity and overstates accuracy. LOSO always tests on a person the
model has never seen.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.aq_analytics.io import figure_label, read_table, save_plot, save_table

LABELS = ["sitting", "walking", "running", "cycling"]
FEATURES = ["acc_mag_mean", "acc_mag_std", "acc_mag_range", "acc_dom_freq_hz",
            "gyro_mag_mean", "gyro_mag_std"]
SAMPLING_HZ = 25
WINDOW_S = 3.0
OVERLAP = 2 / 3   # 1 s stride


def dominant_frequency(x: np.ndarray, hz: float) -> float:
    """Frequency (Hz) with the largest spectral power, ignoring the DC component."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / hz)
    if len(power) < 2 or power[1:].max() < 1e-9:
        return 0.0
    return float(freqs[1:][np.argmax(power[1:])])


def make_features(df: pd.DataFrame, hz=SAMPLING_HZ, window_s=WINDOW_S, overlap=OVERLAP) -> pd.DataFrame:
    """Slide a window over each recording session and compute features."""
    window = int(round(hz * window_s))
    stride = max(1, int(round(window * (1 - overlap))))
    rows = []
    for (subject, session, activity), g in df.groupby(["subject_id", "session_id", "activity"]):
        g = g.sort_values("timestamp")
        acc = np.linalg.norm(g[["ax", "ay", "az"]].to_numpy(dtype=float), axis=1)
        gyr = np.linalg.norm(g[["gx", "gy", "gz"]].to_numpy(dtype=float), axis=1)
        for begin in range(0, len(g) - window + 1, stride):
            a, w = acc[begin:begin + window], gyr[begin:begin + window]
            rows.append({
                "subject_id": subject, "session_id": session, "activity": activity,
                "window_start": g["timestamp"].iloc[begin],
                "acc_mag_mean": a.mean(), "acc_mag_std": a.std(), "acc_mag_range": a.max() - a.min(),
                "acc_dom_freq_hz": dominant_frequency(a, hz),
                "gyro_mag_mean": w.mean(), "gyro_mag_std": w.std(),
            })
    if not rows:
        raise ValueError("IMU recordings are shorter than one window.")
    return pd.DataFrame(rows)


def models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0),
    }


def run(data_dir: Path, out_dir: Path):
    required = {"timestamp", "subject_id", "session_id", "activity", "ax", "ay", "az", "gx", "gy", "gz"}
    df = read_table(data_dir / "har_imu.csv", required)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=list(required))
    label = figure_label(data_dir)

    feats = make_features(df)
    save_table(feats, out_dir / "tables" / "har_window_features.csv")
    subjects = sorted(feats["subject_id"].unique())
    if len(subjects) < 2:
        raise ValueError("Leave-one-subject-out needs at least 2 subjects.")

    fold_rows, all_true, all_pred = [], [], {}
    for held_out in subjects:
        train, test = feats[feats["subject_id"] != held_out], feats[feats["subject_id"] == held_out]
        for name, model in models().items():
            model.fit(train[FEATURES], train["activity"])
            pred = model.predict(test[FEATURES])
            fold_rows.append({"model": name, "test_subject": held_out, "n_test_windows": len(test),
                              "accuracy": accuracy_score(test["activity"], pred),
                              "macro_f1": f1_score(test["activity"], pred, labels=LABELS,
                                                   average="macro", zero_division=0)})
            all_pred.setdefault(name, []).extend(pred)
        all_true.extend(test["activity"])

    folds = pd.DataFrame(fold_rows)
    summary = (folds.groupby("model")
               .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                    n_folds=("test_subject", "count"))
               .sort_values("macro_f1_mean", ascending=False).reset_index())
    save_table(folds, out_dir / "tables" / "har_fold_scores.csv")
    save_table(summary, out_dir / "tables" / "har_model_comparison.csv")

    # Feature importance from a forest fitted on all windows (impurity-based; indicative only).
    rf = models()["RandomForest"].fit(feats[FEATURES], feats["activity"])
    save_table(pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_})
               .sort_values("importance", ascending=False),
               out_dir / "tables" / "har_feature_importance.csv")

    best = summary["model"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ConfusionMatrixDisplay.from_predictions(all_true, all_pred[best], labels=LABELS, normalize="true",
                                            values_format=".2f", ax=ax, colorbar=False, xticks_rotation=25)
    ax.set_title(f"{best} | pooled leave-one-subject-out")
    save_plot(fig, out_dir / "figures" / "har_confusion_matrix.png", label)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for name, g in folds.groupby("model"):
        ax.plot(g["test_subject"], g["macro_f1"], marker="o", label=name)
    ax.set(title="Macro F1 per held-out subject", ylabel="Macro F1", ylim=(0, 1.02))
    ax.legend(); ax.grid(alpha=.2)
    save_plot(fig, out_dir / "figures" / "har_f1_per_subject.png", label)
    plt.close(fig)


if __name__ == "__main__":
    run(Path("data/demo"), Path("outputs/demo"))
