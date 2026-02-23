# Iris MWP

Minimal workflow:

1. Scrape Iris from UCI (cached locally).
2. Build a small config from real statistics.
3. Generate synthetic data and report utility-style metrics (`exp_var_diff`, `comp_angle_diff`, `qMSE`) plus basic moment/species deltas.

Run:

```bash
python demo/iris_synthesis.py
```

Outputs in `demo/output/`:

- `iris_real.csv`
- `iris_synthetic.csv`
- `iris_config.json`
- `summary.json`

Run logs are written to `demo/logs/`.

Cached source:

- `demo/data/iris.data`
