"""Sensor calibration against a co-located reference instrument.

Method
------
For every device x parameter, a linear and a quadratic correction
(reference ~ raw) are compared with leave-one-session-out validation:
fit on all sessions but one, predict the held-out session, repeat.
Testing on a whole unseen session is stricter than a random row split,
because neighbouring rows in a time series are nearly identical and a
random split leaks information from training into testing.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.aq_analytics.io import figure_label, read_table, save_plot, save_table

PARAMS = {"pm25": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)", "tvoc": "TVOC (ppb)",
          "eco2": "eCO₂ (ppm, estimated)", "temp": "Temperature (°C)", "rh": "RH (%)"}
MODELS = {"linear": 1, "quadratic": 2}


def _scores(y_true, y_pred, prefix):
    return {
        f"mae_{prefix}": mean_absolute_error(y_true, y_pred),
        f"rmse_{prefix}": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"r2_{prefix}": r2_score(y_true, y_pred),
        f"bias_{prefix}": float(np.mean(y_pred - y_true)),
    }


def leave_one_session_out(data: pd.DataFrame, degree: int) -> pd.DataFrame:
    """Return out-of-session predictions for every row (column `pred`)."""
    parts = []
    for held_out in sorted(data["session_id"].unique()):
        train = data[data["session_id"] != held_out]
        test = data[data["session_id"] == held_out].copy()
        coeff = np.polyfit(train["raw"], train["ref"], degree)
        test["pred"] = np.polyval(coeff, test["raw"])
        parts.append(test)
    return pd.concat(parts)


def run(data_dir: Path, out_dir: Path):
    required = {"timestamp", "device_id", "session_id"}
    required |= {f"{p}_{kind}" for p in PARAMS for kind in ("raw", "ref")}
    df = read_table(data_dir / "calibration.csv", required)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    label = figure_label(data_dir)
    if df["session_id"].nunique() < 2:
        raise ValueError("Leave-one-session-out needs at least 2 sessions.")

    metric_rows, coeff_rows, predictions = [], [], {}
    for device, by_device in df.groupby("device_id"):
        for param in PARAMS:
            data = (by_device[["timestamp", "session_id", f"{param}_raw", f"{param}_ref"]]
                    .rename(columns={f"{param}_raw": "raw", f"{param}_ref": "ref"})
                    .dropna().sort_values("timestamp"))
            for model, degree in MODELS.items():
                cv = leave_one_session_out(data, degree)
                fold_mae = pd.Series([mean_absolute_error(g["ref"], g["pred"])
                                      for _, g in cv.groupby("session_id")])
                metric_rows.append({
                    "device_id": device, "parameter": param, "model": model, "n": len(cv),
                    **_scores(cv["ref"], cv["raw"], "raw"),
                    **_scores(cv["ref"], cv["pred"], "corrected"),
                    "mae_corrected_fold_std": float(fold_mae.std(ddof=1)),
                })
                predictions[(device, param, model)] = cv
                # Final coefficients use all sessions (what you would deploy).
                c = np.pad(np.polyfit(data["raw"], data["ref"], degree), (3 - degree - 1, 0))
                coeff_rows.append({"device_id": device, "parameter": param, "model": model,
                                   "a2": c[0], "a1": c[1], "a0": c[2]})

    metrics = pd.DataFrame(metric_rows)
    save_table(metrics, out_dir / "tables" / "calibration_metrics.csv")
    save_table(pd.DataFrame(coeff_rows), out_dir / "tables" / "calibration_coefficients.csv")

    device = sorted(df["device_id"].unique())[0]

    # 1) Raw vs reference scatter with both fitted curves.
    d = df[df["device_id"] == device]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(d["pm25_raw"], d["pm25_ref"], s=6, alpha=.3, label="Observations")
    xs = np.linspace(d["pm25_raw"].min(), d["pm25_raw"].max(), 200)
    for model, degree in MODELS.items():
        ax.plot(xs, np.polyval(np.polyfit(d["pm25_raw"], d["pm25_ref"], degree), xs), label=f"{model} fit")
    ax.plot(xs, xs, "k--", lw=.8, label="1:1 line")
    ax.set(title=f"PM2.5 raw vs reference | {device}", xlabel="Sensor raw (µg/m³)",
           ylabel="Reference (µg/m³)")
    ax.legend(); ax.grid(alpha=.2)
    save_plot(fig, out_dir / "figures" / "calibration_scatter_pm25.png", label)
    plt.close(fig)

    # 2) Time series of one held-out session (predictions never saw this session).
    cv = predictions[(device, "pm25", "quadratic")]
    session = sorted(cv["session_id"].unique())[-1]
    s = cv[cv["session_id"] == session]
    fig, ax = plt.subplots(figsize=(8.4, 4))
    ax.plot(s["timestamp"], s["ref"], lw=1.8, label="Reference")
    ax.plot(s["timestamp"], s["raw"], alpha=.8, label="Sensor raw")
    ax.plot(s["timestamp"], s["pred"], alpha=.9, label="Sensor corrected (quadratic)")
    ax.set(title=f"PM2.5 | {device} | held-out {session}", xlabel="Time", ylabel="PM2.5 (µg/m³)")
    ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate()
    save_plot(fig, out_dir / "figures" / "calibration_holdout_timeseries_pm25.png", label)
    plt.close(fig)

    # 3) Error before/after, normalised by the reference mean so all parameters fit one chart.
    m = metrics[metrics["device_id"] == device].copy()
    ref_mean = {p: df.loc[df["device_id"] == device, f"{p}_ref"].abs().mean() for p in PARAMS}
    m["scale"] = m["parameter"].map(ref_mean)
    raw = m[m["model"] == "linear"].set_index("parameter")
    x = np.arange(len(PARAMS))
    fig, ax = plt.subplots(figsize=(8.4, 4))
    ax.bar(x - .27, 100 * raw["mae_raw"] / raw["scale"], .27, label="raw")
    for k, model in enumerate(MODELS, start=0):
        mm = m[m["model"] == model].set_index("parameter").loc[list(PARAMS)]
        ax.bar(x + k * .27, 100 * mm["mae_corrected"] / mm["scale"], .27, label=f"corrected ({model})")
    ax.set_xticks(x, [p.upper() for p in PARAMS])
    ax.set(title=f"Held-out error by parameter | {device}", ylabel="MAE as % of reference mean")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    save_plot(fig, out_dir / "figures" / "calibration_error_by_parameter.png", label)
    plt.close(fig)


if __name__ == "__main__":
    run(Path("data/demo"), Path("outputs/demo"))
