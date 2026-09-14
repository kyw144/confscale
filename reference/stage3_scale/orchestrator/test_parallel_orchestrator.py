#!/usr/bin/env python3
"""
Smoke validation for the parallel orchestrator additions.

Does NOT touch any real kind cluster — only verifies:
  - Modules import cleanly
  - WorkerSlot allocation produces non-conflicting ports
  - Kind config rendering produces valid YAML
  - get_method() returns FRESH copies (parallel-safe)
  - Method instances pick up the per-slot kube_context
  - --dry-run works for both serial and parallel modes

Run:
  python test_parallel_orchestrator.py
or:
  pytest test_parallel_orchestrator.py -v
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from orchestrator.clusters import (  # noqa: E402
    DEFAULT_BASE_FRONTEND_PORT,
    DEFAULT_BASE_PROMETHEUS_PORT,
    DEFAULT_CLUSTER_PREFIX,
    make_slots,
    render_kind_config,
)
from orchestrator.methods import (  # noqa: E402
    KUBE_CONTEXT as DEFAULT_KUBE_CONTEXT,
    current_kube_context, set_thread_kube_context, clear_thread_kube_context, kubectl,
    HPAMethod,
    get_method,
)

# Use logger for status, but also a tiny PASS/FAIL sentinel for humans
logger = logging.getLogger("test_parallel")


PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    assert condition, f"{label}: {detail}"
    results.append((label, condition, detail))
    glyph = PASS if condition else FAIL
    print(f"  {glyph} {label}" + (f"  — {detail}" if detail else ""))


# ── Tests ───────────────────────────────────────────────────────────────

def test_slot_allocation():
    print("\n[1] WorkerSlot port allocation")
    slots = make_slots(4)
    check("4 slots allocated", len(slots) == 4)
    check("contexts unique",
          len({s.kube_context for s in slots}) == 4)
    check("frontend ports unique",
          len({s.frontend_port for s in slots}) == 4)
    check("prom ports unique",
          len({s.prometheus_port for s in slots}) == 4)
    check("frontend ports avoid legacy 30080",
          all(s.frontend_port != 30080 for s in slots),
          f"first slot: {slots[0].frontend_port}")
    check("prom ports avoid legacy 9090",
          all(s.prometheus_port != 9090 for s in slots),
          f"first slot: {slots[0].prometheus_port}")
    check("contexts match pattern",
          all(s.kube_context == f"kind-{DEFAULT_CLUSTER_PREFIX}-w{s.worker_id}" for s in slots))


def test_kind_config_render():
    print("\n[2] Kind config rendering")
    slots = make_slots(2)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for slot in slots:
            cfg_path = tmp_path / f"kind-config-w{slot.worker_id}.yaml"
            render_kind_config(slot, cfg_path)
            check(f"slot {slot.worker_id} file written", cfg_path.exists())
            parsed = yaml.safe_load(cfg_path.read_text())
            check(f"slot {slot.worker_id} valid YAML", isinstance(parsed, dict))
            check(f"slot {slot.worker_id} kind=Cluster",
                  parsed.get("kind") == "Cluster")
            mappings = parsed["nodes"][0]["extraPortMappings"]
            host_ports = {m["hostPort"] for m in mappings}
            check(f"slot {slot.worker_id} hostPort includes frontend port",
                  slot.frontend_port in host_ports,
                  f"got {host_ports}")


def test_get_method_returns_fresh_copies():
    a = get_method("hpa-reactive")
    b = get_method("hpa-reactive")
    a.max_replicas = 99
    check("mutation isolation", b.max_replicas != 99)


def test_method_uses_its_kube_context():
    from unittest.mock import patch
    try:
        set_thread_kube_context("kind-totally-fake-w7")
        with patch("orchestrator.methods.subprocess.run") as run:
            kubectl(["get", "pods"])
            check("explicit thread context", "--context=kind-totally-fake-w7" in run.call_args.args[0])
    finally:
        clear_thread_kube_context()
    check("default context restored", current_kube_context() == DEFAULT_KUBE_CONTEXT)


def test_imports_clean():
    print("\n[5] Module imports")
    try:
        # Import the MODULE (not the function re-exported via __init__).
        import importlib
        rm = importlib.import_module("orchestrator.run_matrix")
        check("orchestrator.run_matrix imports", True)
        check("run_matrix() callable", callable(rm.run_matrix))
        check("_run_matrix_parallel defined", hasattr(rm, "_run_matrix_parallel"))
        check("PortForward accepts kube_context",
              "kube_context" in rm.PortForward.__init__.__code__.co_varnames)
    except Exception as e:
        check("imports", False, str(e))


def test_dry_run_serial():
    print("\n[6] --dry-run --workers 1 (legacy serial path)")
    proc = subprocess.run(
        [sys.executable, str(_HERE / "run_matrix.py"),
         "--config", str(_HERE / "matrix.yaml"),
         "--dry-run", "--workers", "1"],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "CONFSCALE_ENABLE_REFERENCE_RUNTIME": "1"},
    )
    check("exit code 0", proc.returncode == 0,
          f"stderr: {proc.stderr[-200:] if proc.stderr else ''}")
    check("matrix breakdown in output",
          "DRY RUN" in (proc.stdout + proc.stderr))


def test_dry_run_parallel():
    print("\n[7] --dry-run --workers 4 (parallel path)")
    proc = subprocess.run(
        [sys.executable, str(_HERE / "run_matrix.py"),
         "--config", str(_HERE / "matrix.yaml"),
         "--dry-run", "--workers", "4"],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "CONFSCALE_ENABLE_REFERENCE_RUNTIME": "1"},
    )
    output = proc.stdout + proc.stderr
    check("exit code 0", proc.returncode == 0,
          f"stderr: {proc.stderr[-200:] if proc.stderr else ''}")
    check("parallel slots listed", "Parallel slots" in output)
    check("4 slot lines", output.count("kind-p3-experiments-w") >= 4)
    check("frontend port 31080 mentioned", "31080" in output)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING)  # quiet — keep our prints clean
    print("Parallel orchestrator validation")
    print("=" * 60)

    test_slot_allocation()
    test_kind_config_render()
    test_get_method_returns_fresh_copies()
    test_method_uses_its_kube_context()
    test_imports_clean()
    test_dry_run_serial()
    test_dry_run_parallel()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    if passed != total:
        print("\n  FAILED:")
        for label, ok, detail in results:
            if not ok:
                print(f"    {FAIL} {label}  — {detail}")
        sys.exit(1)
    print(f"  {PASS} ALL GREEN")


if __name__ == "__main__":
    main()
