"""Static vs dynamic comparison of wearable sensor readings.

Comparisons are made only WITHIN the same site x run x device, never across
runs, because background air differs between runs. Uncertainty of each
difference uses a moving-block bootstrap, which respects autocorrelation
(consecutive 10 s readings are not independent samples).

Interpretation: a difference here is descriptive. Background drift between
the two segments, sensor response time and placement can all contribute, so
it is not by itself proof that movement caused the change.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aq_analytics.io import figure_label, read_table, save_plot, save_table
from src.aq_analytics.stats import block_bootstrap_diff_ci, descriptive, relative_difference

PARAMS = {"pm25": "µg/m³", "pm10": "µg/m³", "tvoc": "ppb", "eco2": "ppm (est.)",
          "temp": "°C", "rh": "%RH"}
BLOCK = 6  # bootstrap block length in samples (6 x 10 s = 1 min)


def run(data_dir: Path, out_dir: Path):
    required = {"timestamp", "site", "run_id", "device_id", "phase", "motion_intensity"} | set(PARAMS)
    df = read_table(data_dir / "movement.csv", required)
    df = df[df["phase"].isin(["static", "dynamic"])].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    label = figure_label(data_dir)

    desc_rows, compare_rows = [], []
    for (site, run_id, device), local in df.groupby(["site", "run_id", "device_id"]):
        for param in PARAMS:
            values = {ph: pd.to_numeric(local.loc[local["phase"] == ph, param], errors="coerce").dropna()
                      for ph in ("static", "dynamic")}
            stats = {}
            for ph, v in values.items():
                stats[ph] = descriptive(v)
                desc_rows.append({"site": site, "run_id": run_id, "device_id": device,
                                  "parameter": param, "phase": ph, **stats[ph]})
            if stats["static"]["n"] < 2 or stats["dynamic"]["n"] < 2:
                continue
            lo, hi = block_bootstrap_diff_ci(values["static"], values["dynamic"], block=BLOCK)
            sm, dm = stats["static"]["mean"], stats["dynamic"]["mean"]
            compare_rows.append({
                "site": site, "run_id": run_id, "device_id": device, "parameter": param,
                "mean_static": sm, "mean_dynamic": dm, "delta": dm - sm,
                "delta_ci95_low": lo, "delta_ci95_high": hi,
                "delta_pct": relative_difference(dm, sm),
                "cv_static_pct": stats["static"]["cv_pct"], "cv_dynamic_pct": stats["dynamic"]["cv_pct"],
            })
    compare = pd.DataFrame(compare_rows)
    if compare.empty:
        raise ValueError("No run has both static and dynamic data to compare.")
    save_table(pd.DataFrame(desc_rows), out_dir / "tables" / "movement_descriptives.csv")
    save_table(compare, out_dir / "tables" / "movement_static_vs_dynamic.csv")

    # 1) Forest plot: PM2.5 difference with 95% CI for every run x device.
    pm = compare[compare["parameter"] == "pm25"].sort_values(["device_id", "site", "run_id"]).reset_index(drop=True)
    names = pm["device_id"] + " | " + pm["site"] + " " + pm["run_id"]
    fig, ax = plt.subplots(figsize=(7, .45 * len(pm) + 1.2))
    y = np.arange(len(pm))[::-1]
    ax.errorbar(pm["delta"], y, xerr=[pm["delta"] - pm["delta_ci95_low"], pm["delta_ci95_high"] - pm["delta"]],
                fmt="o", capsize=3)
    ax.axvline(0, ls="--", lw=.9, color="grey")
    ax.set_yticks(y, names)
    ax.set(title="PM2.5: dynamic − static (mean, 95% block-bootstrap CI)", xlabel="Difference (µg/m³)")
    ax.grid(axis="x", alpha=.2)
    save_plot(fig, out_dir / "figures" / "movement_pm25_difference_ci.png", label)
    plt.close(fig)

    # 2) Time series of one run: shows the background drift across both phases.
    first = df.groupby(["site", "run_id", "device_id"]).size().index[0]
    one = df[(df["site"] == first[0]) & (df["run_id"] == first[1]) & (df["device_id"] == first[2])]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for phase, g in one.groupby("phase"):
        ax.plot(g["timestamp"], g["pm25"], marker=".", ms=3, lw=.8, label=phase)
    ax.set(title=f"PM2.5 time series | {first[2]} | {first[0]} {first[1]}", xlabel="Time", ylabel="PM2.5 (µg/m³)")
    ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate()
    save_plot(fig, out_dir / "figures" / "movement_pm25_timeseries.png", label)
    plt.close(fig)

    # 3) Distribution per phase for that run.
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.boxplot([one.loc[one["phase"] == p, "pm25"].dropna() for p in ("static", "dynamic")],
               tick_labels=["static", "dynamic"])
    ax.set(title=f"PM2.5 distribution | {first[2]} | {first[0]} {first[1]}", ylabel="PM2.5 (µg/m³)")
    ax.grid(axis="y", alpha=.2)
    save_plot(fig, out_dir / "figures" / "movement_pm25_boxplot.png", label)
    plt.close(fig)


if __name__ == "__main__":
    run(Path("data/demo"), Path("outputs/demo"))
