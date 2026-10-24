"""Generate reproducible, fully SYNTHETIC demo datasets.

Every number below is an arbitrary illustration value chosen to make the
analysis workflow interesting (non-linearity, device differences, drift,
subject variability). None of it comes from real measurements.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7


def _episode(t_min: np.ndarray, start: float, peak: float, rise: float, decay: float) -> np.ndarray:
    """A pollution episode: smooth rise, then exponential decay (t in minutes)."""
    dt = t_min - start
    up = np.clip(dt / rise, 0, 1) ** 2
    down = np.where(dt > rise, np.exp(-(dt - rise) / decay), 1.0)
    return peak * up * down


def make_calibration(rng: np.random.Generator) -> pd.DataFrame:
    """Co-location: wearable sensor next to a reference instrument, 3 sessions/day-like."""
    rows = []
    sessions = {"Session-1": 38, "Session-2": 65, "Session-3": 24}  # episode peak, ug/m3
    devices = {"Device-A": 1.00, "Device-B": 1.08}                  # PM gain differences
    n = 360                                                          # 1 h at 10 s
    t_min = np.arange(n) * 10 / 60
    for s_idx, (session, peak) in enumerate(sessions.items()):
        start = pd.Timestamp("2026-05-04 09:00") + pd.Timedelta(s_idx, unit="D")
        # Reference ("true") values first; the sensor then responds to them.
        pm25_ref = 7 + _episode(t_min, 8, peak, 6, 14) + rng.normal(0, .4, n)
        pm10_ref = 1.6 * pm25_ref + 3 + rng.normal(0, 1.0, n)
        tvoc_ref = 90 + _episode(t_min, 10, 5 * peak, 5, 18) + rng.normal(0, 6, n)
        eco2_ref = 470 + _episode(t_min, 6, 6 * peak, 10, 25) + rng.normal(0, 10, n)
        temp_ref = 25.5 + s_idx * .8 + .6 * np.sin(t_min / 9) + rng.normal(0, .05, n)
        rh_ref = 55 + s_idx * 4 + 3 * np.sin(t_min / 13) + rng.normal(0, .4, n)
        for device, gain in devices.items():
            pm25_ref_d = np.clip(pm25_ref, 0, None)
            pm10_ref_d = np.clip(pm10_ref, 0, None)
            df = pd.DataFrame({
                "timestamp": start + pd.to_timedelta(np.arange(n) * 10, unit="s"),
                "device_id": device,
                "session_id": session,
                # Low-cost sensor behaviour: gain error + mild non-linearity + noise.
                "pm25_raw": gain * (1.25 * pm25_ref_d + .004 * pm25_ref_d**2) + 2 + rng.normal(0, .8, n),
                "pm10_raw": gain * (1.10 * pm10_ref_d + .002 * pm10_ref_d**2) + 3 + rng.normal(0, 1.5, n),
                # Metal-oxide gas sensors saturate at higher concentrations.
                "tvoc_raw": 1.6 * tvoc_ref - .0009 * tvoc_ref**2 - 20 + rng.normal(0, 10, n),
                "eco2_raw": .85 * eco2_ref + 120 + rng.normal(0, 25, n),
                "temp_raw": temp_ref + 1.2 + rng.normal(0, .1, n),   # self-heating offset
                "rh_raw": .92 * rh_ref + 2 + rng.normal(0, .8, n),
                "pm25_ref": pm25_ref_d, "pm10_ref": pm10_ref_d, "tvoc_ref": tvoc_ref,
                "eco2_ref": eco2_ref, "temp_ref": temp_ref, "rh_ref": rh_ref,
            })
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_movement(rng: np.random.Generator) -> pd.DataFrame:
    """Static segment followed by a dynamic segment, per run and device.

    Background air slowly drifts through BOTH segments, so a naive
    static-vs-dynamic difference mixes movement effects with drift.
    """
    rows = []
    runs = [("indoor", "Run-1"), ("indoor", "Run-2"), ("indoor", "Run-3"),
            ("outdoor", "Run-1"), ("outdoor", "Run-2")]
    effect = {"Device-A": 1.2, "Device-B": .5}      # added PM2.5 during movement
    n = 60                                           # 10 min per phase at 10 s
    for r_idx, (site, run) in enumerate(runs):
        start = pd.Timestamp("2026-05-11 10:00") + pd.Timedelta(3 * r_idx, unit="h")
        drift_rate = rng.uniform(-.015, .02)         # ug/m3 per sample
        base = (9 + 2 * r_idx) + drift_rate * np.arange(2 * n) + rng.normal(0, .2, 2 * n).cumsum() * .1
        for device in effect:
            for p_idx, phase in enumerate(["static", "dynamic"]):
                idx = np.arange(p_idx * n, (p_idx + 1) * n)
                moving = phase == "dynamic"
                b = base[idx]
                motion = np.clip(rng.normal(1.5 if moving else .05, .4 if moving else .02, n), 0, None)
                artifact = effect[device] * moving
                noise = 1.6 if moving else 1.0
                rows.append(pd.DataFrame({
                    "timestamp": start + pd.to_timedelta(idx * 10, unit="s"),
                    "site": site, "run_id": run, "device_id": device, "phase": phase,
                    "motion_intensity": motion,
                    "pm25": b + artifact + rng.normal(0, noise, n),
                    "pm10": 1.5 * b + 2.2 * artifact + rng.normal(0, 1.5 * noise, n),
                    "tvoc": 110 + 3 * b + rng.normal(0, 6, n),
                    "eco2": 520 + 4 * b + rng.normal(0, 20, n),
                    "temp": 26 + .02 * idx / n + .15 * moving + rng.normal(0, .08, n),
                    "rh": 58 - .3 * moving + rng.normal(0, .5, n),
                }))
    return pd.concat(rows, ignore_index=True)


def make_har(rng: np.random.Generator, hz: int = 25, seconds: int = 45) -> pd.DataFrame:
    """Raw accelerometer (m/s^2) and gyroscope (rad/s) for 4 subjects x 4 activities.

    Each subject gets its own intensity and cadence so that a model trained on
    other people does not transfer perfectly (as in real life).
    """
    activities = {
        #            freq Hz, acc amp, gyro amp, noise
        "sitting": (0.0, .00, .00, .05),
        "walking": (1.8, 2.0, .60, .45),
        "running": (2.7, 4.5, 1.30, .80),
        "cycling": (1.3, 1.1, .90, .55),
    }
    n = hz * seconds
    t = np.arange(n) / hz
    rows = []
    for s in range(1, 5):
        gain = rng.uniform(.7, 1.3)          # body/placement differences
        cadence = rng.uniform(.85, 1.15)
        for a_idx, (activity, (f, acc, gyr, noise)) in enumerate(activities.items()):
            phase = rng.uniform(0, 2 * np.pi)
            # Cadence and intensity wander over time (terrain, fatigue, traffic).
            f_t = f * cadence * (1 + .12 * np.sin(2 * np.pi * t / rng.uniform(15, 30)))
            wave = np.sin(2 * np.pi * np.cumsum(f_t) / hz + phase)
            harm = .35 * np.sin(4 * np.pi * np.cumsum(f_t) / hz + phase)
            amp = acc * gain * (1 + .3 * np.sin(2 * np.pi * t / rng.uniform(10, 25)))
            start = pd.Timestamp("2026-05-18 08:00") + pd.Timedelta(30 * (4 * (s - 1) + a_idx), unit="min")
            rows.append(pd.DataFrame({
                "timestamp": start + pd.to_timedelta(t, unit="s"),
                "subject_id": f"Subject-{s}",
                "session_id": f"S{s}-{activity}",
                "activity": activity,
                "ax": .9 * amp * (wave + harm) + rng.normal(0, noise, n),
                "ay": .5 * amp * harm + rng.normal(0, noise, n),
                "az": 9.81 + amp * wave + rng.normal(0, noise, n),
                "gx": gyr * gain * wave + rng.normal(0, noise * .3, n),
                "gy": gyr * gain * .5 * np.cos(2 * np.pi * f * cadence * t) + rng.normal(0, noise * .3, n),
                "gz": rng.normal(0, .05 + noise * .2, n),
            }))
    return pd.concat(rows, ignore_index=True)


def make_sampling_stream(rng: np.random.Generator) -> pd.DataFrame:
    """A 4-hour stream at a fixed 10 s base interval with still/moving blocks."""
    n = 4 * 360
    t_min = np.arange(n) * 10 / 60
    moving = np.zeros(n, dtype=bool)
    for a, b in [(20, 45), (95, 110), (170, 200)]:     # minutes of walking
        moving[(t_min >= a) & (t_min < b)] = True
    motion = np.where(moving, rng.normal(1.6, .4, n), rng.normal(.06, .03, n)).clip(0)
    pm25 = (10 + .01 * t_min
            + _episode(t_min, 60, 35, 5, 12)          # e.g. cooking nearby
            + _episode(t_min, 150, 22, 3, 8)          # e.g. passing traffic
            + rng.normal(0, .5, n) + moving * rng.normal(.8, 1.2, n))
    return pd.DataFrame({
        "timestamp": pd.Timestamp("2026-05-25 08:00") + pd.to_timedelta(np.arange(n) * 10, unit="s"),
        "device_id": "Device-A",
        "motion_intensity": motion,
        "pm25": pm25.clip(0),
    })


def run(data_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "calibration.csv": make_calibration(rng),
        "movement.csv": make_movement(rng),
        "har_imu.csv": make_har(rng),
        "sampling_stream.csv": make_sampling_stream(rng),
    }
    for name, df in outputs.items():
        num = df.select_dtypes("number").columns
        df[num] = df[num].round(4)
        df.to_csv(data_dir / name, index=False)
        print(f"  > {data_dir / name} ({len(df)} rows)")
    print("Synthetic demo data ready (not real measurements).")


if __name__ == "__main__":
    run(Path("data/demo"))
