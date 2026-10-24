import numpy as np
import pandas as pd

from scripts.har_analysis import dominant_frequency, make_features
from scripts.sampling_analysis import Policy, evaluate, simulate


def _imu(n=500, hz=25):
    t = np.arange(n) / hz
    return pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="s"),
        "subject_id": "S1", "session_id": "s1", "activity": "walking",
        "ax": np.zeros(n), "ay": np.zeros(n), "az": 9.81 + np.sin(2 * np.pi * 2 * t),
        "gx": np.zeros(n), "gy": np.zeros(n), "gz": np.zeros(n),
    })


def test_window_count_with_overlap():
    # 500 samples at 25 Hz, window 3 s = 75 samples, stride 1 s = 25 samples
    # -> (500 - 75) // 25 + 1 = 18 windows
    assert len(make_features(_imu())) == 18


def test_dominant_frequency_finds_sine():
    t = np.arange(100) / 25
    assert abs(dominant_frequency(np.sin(2 * np.pi * 2 * t), 25) - 2.0) < .26


def _stream(motion, pm):
    n = len(pm)
    return pd.DataFrame({"timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(n) * 10, unit="s"),
                         "device_id": "D", "motion_intensity": motion, "pm25": pm})


def test_policy_samples_less_when_calm():
    calm = _stream(np.zeros(720), np.full(720, 10.0))
    result = evaluate(simulate(calm, Policy(), base_s=10), base_s=10, short_s=10)
    assert result["reading_reduction_pct"] > 50
    assert result["reconstruction_mae"] == 0


def test_policy_stays_short_when_moving():
    moving = _stream(np.ones(360), np.full(360, 10.0))
    result = evaluate(simulate(moving, Policy(), base_s=10), base_s=10, short_s=10)
    assert result["reading_reduction_pct"] == 0
