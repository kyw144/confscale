#!/usr/bin/env python3
"""
Training Data Generator — produce 24h of synthetic request rate timeseries
for each workload pattern (A/B/C/D) to train the GRU predictor (task 04).

Computes RPS traces directly using the pattern functions from workload_gen.py
(no subprocess — much faster than trace-only mode which creates TickResult objects
for each of 86,400 iterations).

Usage:
  # Generate all 4 patterns (24h each)
  python generate_training_data.py --all

  # Generate a single pattern
  python generate_training_data.py --pattern A

  # Downsample to custom interval
  python generate_training_data.py --all --step 60

Output: training_data/pattern_<A|B|C|D>_24h.csv
  Columns: timestamp, rps
"""

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Import pattern functions directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workload_gen import (
    pattern_diurnal,
    pattern_bursty,
    pattern_batch_ramp,
    pattern_signaling,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "training_data"

PATTERN_LABELS = {
    "A": "Diurnal Sinusoidal",
    "B": "Bursty Spikes",
    "C": "Batch-Ramp",
    "D": "Signaling Pulse",
}


def generate_trace(pattern: str, duration_s: int = 86400,
                   output_dir: Path = OUTPUT_DIR,
                   step_s: int = 30,
                   seed: int = 42) -> Path:
    """
    Generate a synthetic RPS trace for one pattern.

    Args:
        pattern: Workload pattern (A/B/C/D)
        duration_s: Duration in seconds (default: 86400 = 24h)
        output_dir: Directory for output CSV
        step_s: Downsample interval in seconds
        seed: Random seed for reproducibility

    Returns:
        Path to the generated CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Determine label for filename suffix
    hours = duration_s // 3600
    suffix = f"{hours}h" if hours >= 24 else f"{int(duration_s / 60)}m"

    logger.info("Generating %s trace for Pattern %s (%s)...",
                 suffix, pattern, PATTERN_LABELS[pattern])

    # Build the RPS function
    if pattern == "A":
        rps_fn = lambda t: pattern_diurnal(t, rps_base=50)
    elif pattern == "B":
        # Pre-generate spike events (same as workload_gen.run)
        spike_events = []
        n_spikes = random.randint(5, 8)
        for _ in range(n_spikes):
            ti = random.uniform(60, duration_s - 60)
            peak = random.uniform(40, 120)
            spike_events.append((peak, ti))
        rps_fn = lambda t: max(1, pattern_bursty(t, rps_base=30, spike_events=spike_events))
    elif pattern == "C":
        rps_fn = lambda t: pattern_batch_ramp(t, rps_base=20, rps_peak=150)
    elif pattern == "D":
        rps_fn = lambda t: pattern_signaling(t, rps_pulse=120, rps_idle=5)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Generate per-second RPS values
    rps_values = []
    for t in range(duration_s):
        rps = max(0, int(round(rps_fn(float(t)))))
        rps_values.append(rps)

    # Downsample to step_s intervals
    output_csv = output_dir / f"pattern_{pattern}_{suffix}.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rps"])

        for bucket_start in range(0, duration_s, step_s):
            bucket_end = min(bucket_start + step_s, duration_s)
            bucket_vals = rps_values[bucket_start:bucket_end]
            mean_rps = round(sum(bucket_vals) / len(bucket_vals), 2)
            writer.writerow([bucket_start, mean_rps])

    n_rows = duration_s // step_s
    logger.info("Generated: %s (%d rows at %ds intervals)", output_csv, n_rows, step_s)
    return output_csv


def generate_all_patterns(output_dir: Path = OUTPUT_DIR, duration_s: int = 86400) -> dict[str, Path]:
    """Generate traces for all four patterns.

    For multi-day traces (duration_s > 86400), generates N independent
    24h realizations with different seeds, concatenated. This ensures
    burst events and signaling periods are distributed across the full
    trace rather than concentrated in one segment — critical for honest
    chronological train/val/test splits.
    """
    results = {}
    for pattern in ["A", "B", "C", "D"]:
        if duration_s <= 86400:
            # Single 24h trace
            path = generate_trace(pattern, duration_s=duration_s, output_dir=output_dir)
            results[pattern] = path
        else:
            # Multi-day: generate N independent 24h traces with different seeds
            days = duration_s // 86400
            all_rows = []
            for day in range(days):
                day_seed = 42 + day * 100  # Deterministic but different per day
                # Temporarily generate to a temp dir
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_dir = Path(tmpdir)
                    path = generate_trace(pattern, duration_s=86400,
                                          output_dir=tmp_dir, seed=day_seed)
                    # Read back the rows
                    import csv as _csv
                    with open(path) as f:
                        reader = _csv.DictReader(f)
                        rows = list(reader)
                    # Offset timestamps
                    day_offset = day * 86400
                    for row in rows:
                        row["timestamp"] = str(int(row["timestamp"]) + day_offset)
                        all_rows.append(row)

            # Write concatenated output
            hours = duration_s // 3600
            suffix = f"{hours}h"
            output_csv = output_dir / f"pattern_{pattern}_{suffix}.csv"
            step_s = 30
            with open(output_csv, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(["timestamp", "rps"])
                for row in all_rows:
                    writer.writerow([row["timestamp"], row["rps"]])
            n_rows = len(all_rows)
            logger.info("Generated: %s (%d rows, %d days concatenated)",
                         output_csv, n_rows, days)
            results[pattern] = output_csv

    return results


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 24h synthetic training data traces"
    )
    parser.add_argument("--all", action="store_true",
                        help="Generate all 4 patterns")
    parser.add_argument("--pattern", choices=["A", "B", "C", "D"],
                        help="Generate a single pattern")
    parser.add_argument("--duration", type=int, default=86400,
                        help="Duration in seconds (default: 86400 = 24h)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days of synthetic data to generate (default: 1). "
                             "Overrides --duration if > 1.")
    parser.add_argument("--step", type=int, default=30,
                        help="Downsample interval in seconds (default: 30)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    output_dir = Path(args.output_dir)
    t0 = time.time()

    # Resolve duration: --days overrides --duration if > 1
    if args.days > 1:
        duration_s = args.days * 86400
    else:
        duration_s = args.duration

    if args.all:
        results = generate_all_patterns(output_dir, duration_s=duration_s)
        success = sum(1 for p in results.values() if p.exists())
        logger.info("Generated %d/4 patterns in %.1fs", success, time.time() - t0)
        if success < 4:
            sys.exit(1)
    elif args.pattern:
        path = generate_trace(args.pattern, duration_s=duration_s,
                              output_dir=output_dir, step_s=args.step)
        if not path.exists():
            sys.exit(1)
        logger.info("Generated 1 pattern in %.1fs", time.time() - t0)
    else:
        parser.print_help()
        sys.exit(1)

    logger.info("Training data in: %s", output_dir)


if __name__ == "__main__":
    main()
