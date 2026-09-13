#!/usr/bin/env python3
"""
E-V6 focused test for the workload-trace path-doubling fix (Anomaly A4).

Asserts the two properties the fix must guarantee, WITHOUT touching a cluster:

  (a) With run_dir resolved to an absolute path (the fix at
      run_matrix.py:256), the workload-generator subprocess — launched with
      cwd=run_dir AND --output-dir str(run_dir) exactly as execute_single_run
      does — writes its trace FLAT under run_dir, where the flat
      run_dir.glob("workload_*_timeseries.csv") finds it. A regression guard
      shows the OLD relative run_dir doubles the path
      (<run_dir>/data/.../workload_*.csv) and the glob misses it.

  (b) Once that flat trace is found, compute_e2e_slo_metrics yields a
      NON-EMPTY e2e block, and collect.py's mapping turns it into non-empty
      metrics.json "e2e.*" keys — i.e. the knock-on that emptied e2e.* is gone.

Run:
  python test_ev6_e2e_path.py
or:
  pytest test_ev6_e2e_path.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent  # src/stage3_scale
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from collector.queries import compute_e2e_slo_metrics  # noqa: E402

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILURES.append(name)


# A tiny stand-in for workload_gen.py: same output contract — argparse
# --output-dir, os.makedirs(output_dir), write
# {output_dir}/workload_<pat>_<ts>_timeseries.csv with the real header.
_FAKE_GEN = '''\
import argparse, os
p = argparse.ArgumentParser()
p.add_argument("pattern")
p.add_argument("--output-dir", default="outputs")
a, _ = p.parse_known_args()
os.makedirs(a.output_dir, exist_ok=True)
prefix = f"{a.output_dir}/workload_{a.pattern}_20260531_000000"
with open(f"{prefix}_timeseries.csv", "w") as f:
    f.write("elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\\n")
    f.write("0.0,50,49,49,0,120.0,180.0,210.0\\n")   # below 200ms -> ok
    f.write("1.0,50,50,50,0,300.0,450.0,520.0\\n")   # above 200ms -> violation
    f.write("2.0,50,50,50,0,310.0,470.0,540.0\\n")   # above 200ms -> violation
'''


def _run_gen(gen_abs: Path, output_dir_arg: str, cwd: str) -> None:
    """Invoke the fake generator the way execute_single_run invokes the real
    one: cwd=run_dir, --output-dir str(run_dir). gen_abs is absolute so the
    script is always found regardless of the (deliberately varied) cwd."""
    subprocess.run(
        [sys.executable, str(gen_abs), "G", "--output-dir", output_dir_arg],
        cwd=cwd, check=True, capture_output=True, text=True,
    )


def test_resolved_run_dir_prevents_doubling() -> None:
    print("test_resolved_run_dir_prevents_doubling")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()  # stands in for the orchestrator launch cwd
        gen_abs = root / "_fake_gen.py"
        gen_abs.write_text(_FAKE_GEN)
        rel_output = Path("data/p3_runs/outputs/test_batch")  # relative --output-dir
        run_id = "confscale-pid_g_rep1_20260531_000000"
        prev = os.getcwd()
        os.chdir(root)
        try:
            # --- OLD behaviour (regression guard): run_dir left RELATIVE ---
            old_run_dir = rel_output / run_id
            old_run_dir.mkdir(parents=True, exist_ok=True)
            _run_gen(gen_abs, output_dir_arg=str(old_run_dir),
                     cwd=str(old_run_dir))
            old_flat = list(old_run_dir.glob("workload_*_timeseries.csv"))
            doubled = list(old_run_dir.glob(
                "data/p3_runs/outputs/**/workload_*_timeseries.csv"))
            check("OLD relative run_dir DOUBLES the trace (bug reproduced)",
                  len(old_flat) == 0 and len(doubled) == 1,
                  f"flat={len(old_flat)} doubled={len(doubled)}")

            # --- NEW behaviour (the fix): run_dir RESOLVED to absolute ---
            run_id2 = "confscale-pid_g_rep2_20260531_000000"
            new_run_dir = (rel_output / run_id2).resolve()
            new_run_dir.mkdir(parents=True, exist_ok=True)
            _run_gen(gen_abs, output_dir_arg=str(new_run_dir),
                     cwd=str(new_run_dir))
            new_flat = list(new_run_dir.glob("workload_*_timeseries.csv"))
            new_doubled = list(new_run_dir.glob(
                "data/p3_runs/outputs/**/workload_*_timeseries.csv"))
            check("(a) FIXED absolute run_dir lands trace FLAT",
                  len(new_flat) == 1 and len(new_doubled) == 0,
                  f"flat={len(new_flat)} doubled={len(new_doubled)}")
            check("(a) flat trace is where run_matrix.py:332 globs",
                  bool(new_flat) and new_flat[0].parent == new_run_dir)
        finally:
            os.chdir(prev)


def test_e2e_block_nonempty_from_flat_trace() -> None:
    print("test_e2e_block_nonempty_from_flat_trace")
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "workload_G_20260531_000000_timeseries.csv"
        csv_path.write_text(
            "elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\n"
            "0.0,50,49,49,0,120.0,180.0,210.0\n"   # p95 below 200 -> ok
            "1.0,50,50,50,0,300.0,450.0,520.0\n"   # p95 above 200 -> violation
            "2.0,50,50,50,0,310.0,470.0,540.0\n"   # p95 above 200 -> violation
        )
        e2e = compute_e2e_slo_metrics(csv_path)  # default threshold 200ms
        check("(b) compute_e2e_slo_metrics returns a dict", isinstance(e2e, dict))
        check("(b) e2e_p95_ms is non-null",
              e2e.get("e2e_p95_ms") is not None, f"p95={e2e.get('e2e_p95_ms')}")
        check("(b) e2e_total_intervals == 3", e2e.get("e2e_total_intervals") == 3)
        check("(b) e2e_slo_violation_rate == 2/3",
              abs(e2e.get("e2e_slo_violation_rate") - round(2 / 3, 4)) < 1e-9,
              f"viol={e2e.get('e2e_slo_violation_rate')}")

        # Replicate collect.py:268-273 mapping, then serialise like
        # _write_metrics_json -> assert the metrics.json "e2e.*" block is non-empty.
        scalar = {
            "e2e.p50_ms": e2e.get("e2e_p50_ms"),
            "e2e.p95_ms": e2e.get("e2e_p95_ms"),
            "e2e.p99_ms": e2e.get("e2e_p99_ms"),
            "e2e.slo_violation_rate": e2e.get("e2e_slo_violation_rate"),
            "e2e.slo_violation_intervals": e2e.get("e2e_slo_violation_intervals"),
            "e2e.total_intervals": e2e.get("e2e_total_intervals"),
        }
        nested: dict = {}
        for k, v in scalar.items():
            top, sub = k.split(".", 1)
            nested.setdefault(top, {})[sub] = v
        mj = Path(td) / "metrics.json"
        mj.write_text(json.dumps(nested, indent=2))
        reread = json.loads(mj.read_text())
        check("(b) metrics.json carries a non-empty e2e.* block",
              bool(reread.get("e2e")) and reread["e2e"].get("p95_ms") is not None,
              f"e2e={reread.get('e2e')}")


def main() -> int:
    test_resolved_run_dir_prevents_doubling()
    test_e2e_block_nonempty_from_flat_trace()
    print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
