# MLOps Batch Signal Job

A minimal MLOps-style batch job that reads OHLCV market data, computes a rolling-mean signal, and writes structured metrics — fully reproducible and Dockerized.

---

## What it does

1. Loads and validates a YAML config (`seed`, `window`, `version`)
2. Reads an OHLCV CSV and validates the `close` column
3. Computes a rolling mean over `close` using the configured `window`
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes a structured `metrics.json` and a detailed `run.log`

---

## Local run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Command

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

No hard-coded paths — all paths are passed as CLI arguments.

---

## Docker build & run

```bash
# Build
docker build -t mlops-task .

# Run (uses data.csv + config.yaml baked into the image)
docker run --rm mlops-task
```

The container will:

- Print the final `metrics.json` to **stdout**
- Write `run.log` internally (copy out with `docker cp` if needed)
- Exit `0` on success, non-zero on failure

---

## Config (`config.yaml`)

```yaml
seed: 42
window: 5
version: "v1"
```

| Field     | Type   | Description                           |
| --------- | ------ | ------------------------------------- |
| `seed`    | int    | NumPy random seed for reproducibility |
| `window`  | int    | Rolling mean window size              |
| `version` | string | Pipeline version tag                  |

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 21.0,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: ['open', 'high']"
}
```

---

## Project structure

```
.
├── run.py            # Main batch job
├── config.yaml       # Config (seed, window, version)
├── data.csv          # Input OHLCV dataset (10 000 rows)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker build spec
├── metrics.json      # Sample output from successful run
├── run.log           # Sample log from successful run
└── README.md
```

---

## Evaluation checklist

- Deterministic via `seed` in config
- No hard-coded paths
- Metrics written in both success **and** error cases
- Docker exits `0` success / non-zero failure
- Structured JSON metrics + timestamped logs
