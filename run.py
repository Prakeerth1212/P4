import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch signal job")
    parser.add_argument("--input",    required=True, help="Path to input CSV (OHLCV data)")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping")

    required_fields = {"seed", "window", "version"}
    missing = required_fields - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str):
        raise ValueError(f"'version' must be a string, got: {type(cfg['version'])}")

    logger.info(f"Config loaded and validated — seed={cfg['seed']}, window={cfg['window']}, version={cfg['version']}")
    return cfg

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Dataset has no rows after parsing")
    df.columns = [c.strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"Required column 'close' not found. Available columns: {list(df.columns)}")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().all():
        raise ValueError("Column 'close' contains no valid numeric values")

    logger.info(f"Dataset loaded — {len(df)} rows, columns: {list(df.columns)}")
    return df

def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Computing rolling mean with window={window}")

    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    df["signal"] = np.where(
        df["rolling_mean"].notna() & (df["close"] > df["rolling_mean"]),
        1,
        0
    )
    nan_rows = df["rolling_mean"].isna().sum()
    logger.info(
        f"Signal generation complete — "
        f"{nan_rows} warm-up rows (NaN rolling mean) excluded from signal, "
        f"signal=1 count: {df['signal'].sum()}, signal=0 count: {(df['signal'] == 0).sum()}"
    )
    return df

def compute_metrics(df: pd.DataFrame, cfg: dict, latency_ms: float) -> dict:
    signal_rate = float(np.mean(df["signal"]))
    return {
        "version":        cfg["version"],
        "rows_processed": len(df),
        "metric":         "signal_rate",
        "value":          round(signal_rate, 4),
        "latency_ms":     round(latency_ms, 0),
        "seed":           cfg["seed"],
        "status":         "success",
    }

def write_metrics(metrics: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

def write_error_metrics(error_msg: str, output_path: str, version: str = "unknown"):
    error_payload = {
        "version":       version,
        "status":        "error",
        "error_message": error_msg,
    }
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(error_payload, f, indent=2)
        print(json.dumps(error_payload, indent=2))
    except Exception:
        # Last resort: always print something
        print(json.dumps(error_payload, indent=2))

def main():
    args = parse_args()
    logger = setup_logging(args.log_file)
    version = "unknown"
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Job started")
    logger.info(f"  input   : {args.input}")
    logger.info(f"  config  : {args.config}")
    logger.info(f"  output  : {args.output}")
    logger.info(f"  log-file: {args.log_file}")
    logger.info("=" * 60)
    try:
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        np.random.seed(cfg["seed"])
        logger.info(f"Random seed set to {cfg['seed']}")
        df = load_dataset(args.input, logger)
        df = compute_signals(df, cfg["window"], logger)
        latency_ms = (time.time() - start_time) * 1000
        metrics = compute_metrics(df, cfg, latency_ms)
        logger.info(f"Metrics — rows_processed={metrics['rows_processed']}, "
                    f"signal_rate={metrics['value']}, latency_ms={metrics['latency_ms']}")
        write_metrics(metrics, args.output)
        logger.info(f"Metrics written to {args.output}")
        logger.info("Job completed successfully")
        logger.info("=" * 60)
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Job failed: {exc}")
        write_error_metrics(str(exc), args.output, version=version)
        logger.info("=" * 60)
        sys.exit(1)
if __name__ == "__main__":
    main()
