"""Adaptive sampling: how much can we sample less, and what do we lose?

A full-rate stream is replayed through a simple rule-based policy:
  * SHORT mode: read every `short_s` seconds.
  * If the last `stable_window_s` of readings are calm (PM2.5 std below a
    threshold AND motion below a threshold), switch to LONG mode and read
    every `long_s` seconds for up to `hold_s` seconds.
  * Leave LONG mode early if a reading shows motion or a sudden PM2.5 jump.

We then reconstruct the full signal from the kept readings (holding the last
value) and measure the error against the full stream. This frames the
trade-off: fewer readings vs. information lost.

Note: fewer readings is NOT the same as measured energy savings. Claims about
battery life require current/voltage measurements on real hardware.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aq_analytics.io import figure_label, read_table, save_plot, save_table


@dataclass
class Policy:
    short_s: int = 10
    long_s: int = 60
    stable_window_s: int = 120
    hold_s: int = 600
    pm_std_threshold: float = 1.0
    motion_threshold: float = .3


def simulate(df: pd.DataFrame, policy: Policy, base_s: int) -> pd.DataFrame:
    """Return the stream with a boolean `sampled` column and the active `mode`."""
    short_step = max(1, policy.short_s // base_s)
    long_step = max(1, policy.long_s // base_s)
    window_n = max(2, policy.stable_window_s // policy.short_s)
    hold_steps = policy.hold_s // base_s
    pm, motion = df["pm25"].to_numpy(), df["motion_intensity"].to_numpy()
    sampled = np.zeros(len(df), dtype=bool)
    modes = np.empty(len(df), dtype=object)
    mode, recent, long_until, last_pm, i = "short", [], -1, None, 0
    while i < len(df):
        sampled[i] = True
        if mode == "long":
            jump = last_pm is not None and abs(pm[i] - last_pm) > 3 * policy.pm_std_threshold
            if motion[i] > policy.motion_threshold or jump or i >= long_until:
                mode, recent = "short", []
        if mode == "short":
            recent = (recent + [(pm[i], motion[i])])[-window_n:]
            calm = (len(recent) == window_n
                    and np.std([r[0] for r in recent]) < policy.pm_std_threshold
                    and max(r[1] for r in recent) < policy.motion_threshold)
            if calm:
                mode, long_until = "long", i + hold_steps
        last_pm = pm[i]
        step = long_step if mode == "long" else short_step
        modes[i:i + step] = mode
        i += step
    out = df.copy()
    out["sampled"], out["mode"] = sampled, modes
    return out


def evaluate(sim: pd.DataFrame, base_s: int, short_s: int) -> dict:
    truth = sim["pm25"].to_numpy()
    recon = sim["pm25"].where(sim["sampled"]).ffill().to_numpy()
    n_short_policy = int(np.ceil(len(sim) * base_s / short_s))
    kept = int(sim["sampled"].sum())
    return {
        "readings_kept": kept,
        "readings_fixed_short": n_short_policy,
        "reading_reduction_pct": 100 * (1 - kept / n_short_policy),
        "time_in_long_mode_pct": 100 * (sim["mode"] == "long").mean(),
        "reconstruction_mae": float(np.mean(np.abs(recon - truth))),
        "reconstruction_max_abs_error": float(np.max(np.abs(recon - truth))),
        "peak_true": float(truth.max()),
        "peak_captured": float(sim.loc[sim["sampled"], "pm25"].max()),
    }


def run(data_dir: Path, out_dir: Path):
    required = {"timestamp", "device_id", "motion_intensity", "pm25"}
    df = read_table(data_dir / "sampling_stream.csv", required)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=list(required)).sort_values("timestamp").reset_index(drop=True)
    label = figure_label(data_dir)
    base_s = int(round(df["timestamp"].diff().dt.total_seconds().median()))

    rows, sims = [], {}
    for device, stream in df.groupby("device_id"):
        stream = stream.reset_index(drop=True)
        for thr in (.25, .5, 1.0, 2.0, 4.0):
            policy = Policy(pm_std_threshold=thr)
            sim = simulate(stream, policy, base_s)
            sims[(device, thr)] = sim
            rows.append({"device_id": device, "pm_std_threshold": thr, **evaluate(sim, base_s, policy.short_s),
                         "energy_savings_measured": False})
    sweep = pd.DataFrame(rows)
    save_table(sweep, out_dir / "tables" / "sampling_policy_sweep.csv")

    device = sorted(df["device_id"].unique())[0]
    sim = sims[(device, Policy().pm_std_threshold)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[3, 1.3])
    ax1.plot(sim["timestamp"], sim["pm25"], lw=.8, color="grey", label="full-rate stream")
    kept = sim[sim["sampled"]]
    ax1.scatter(kept["timestamp"], kept["pm25"], s=5, label="readings kept by policy")
    ax1.set(title=f"Adaptive sampling replay | {device} | default policy", ylabel="PM2.5 (µg/m³)")
    ax1.legend(loc="upper left"); ax1.grid(alpha=.2)
    interval = np.where(sim["mode"] == "long", Policy().long_s, Policy().short_s)
    ax2.step(sim["timestamp"], interval, where="post")
    ax2.set(ylabel="interval (s)", xlabel="Time"); ax2.grid(alpha=.2)
    fig.autofmt_xdate()
    save_plot(fig, out_dir / "figures" / "sampling_policy_timeline.png", label)
    plt.close(fig)

    s = sweep[sweep["device_id"] == device]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(s["reading_reduction_pct"], s["reconstruction_mae"], marker="o")
    for _, r in s.iterrows():
        ax.annotate(f"thr={r['pm_std_threshold']:g}", (r["reading_reduction_pct"], r["reconstruction_mae"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set(title="Trade-off: fewer readings vs. reconstruction error",
           xlabel="Reading reduction vs. fixed short interval (%)", ylabel="Reconstruction MAE (µg/m³)")
    ax.grid(alpha=.2)
    save_plot(fig, out_dir / "figures" / "sampling_tradeoff.png", label)
    plt.close(fig)


if __name__ == "__main__":
    run(Path("data/demo"), Path("outputs/demo"))
