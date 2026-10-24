# Data dictionary

All inputs are CSV files with one header row, UTF-8 encoding, `.` as decimal separator and ISO timestamps (e.g. `2026-05-04 09:00:00`). Use one consistent timezone.

## `calibration.csv` — sensor next to a reference instrument

| Column | Meaning |
|---|---|
| `timestamp` | Time of the aligned reading |
| `device_id` | Wearable unit identifier |
| `session_id` | Co-location session (each is held out once during validation; need ≥ 2) |
| `<p>_raw` | Wearable sensor reading for parameter `<p>` |
| `<p>_ref` | Reference instrument reading for parameter `<p>` at the same time |

Parameters `<p>`: `pm25`, `pm10` (µg/m³), `tvoc` (ppb), `eco2` (ppm, estimated), `temp` (°C), `rh` (%). Align sensor and reference by timestamp (nearest within a stated tolerance), never by row number.

## `movement.csv` — static then dynamic segments

| Column | Meaning |
|---|---|
| `timestamp` | Reading time |
| `site` | e.g. `indoor`, `outdoor` |
| `run_id` | Run identifier within a site |
| `device_id` | Wearable unit |
| `phase` | `static` or `dynamic` |
| `motion_intensity` | Movement indicator from the IMU (may be blank) |
| `pm25`, `pm10`, `tvoc`, `eco2`, `temp`, `rh` | Sensor readings |

## `har_imu.csv` — raw IMU samples

| Column | Meaning |
|---|---|
| `timestamp` | Sample time (fixed rate; the default is 25 Hz, set in `har_analysis.py`) |
| `subject_id` | Person wearing the device |
| `session_id` | One continuous recording |
| `activity` | `sitting`, `walking`, `running`, `cycling` |
| `ax`, `ay`, `az` | Acceleration (m/s²) |
| `gx`, `gy`, `gz` | Angular rate (rad/s) |

## `sampling_stream.csv` — full-rate stream for the sampling replay

| Column | Meaning |
|---|---|
| `timestamp` | Fixed base interval (the demo uses 10 s) |
| `device_id` | Wearable unit |
| `motion_intensity` | Movement indicator |
| `pm25` | PM2.5 reading (µg/m³) |
