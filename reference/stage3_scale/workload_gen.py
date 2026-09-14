#!/usr/bin/env python3
"""
InfoSys Benchmark — Workload Generator (4 base patterns + 3 drift modes).

Generates time-varying request rates matching the formal workload patterns
defined in the P3 experimental plan (paper_plan_P3_experiments.md).

Base patterns:
  A — Diurnal Sinusoidal (stable, predictable)
  B — Bursty Spikes (high uncertainty)
  C — Batch-Ramp (sustained load change)
  D — Signaling Pulse (intermittent, sharp transitions)
  E — Constant Flat (fixed RPS, for profiling/capacity testing)

Drift modes (mid-run distribution shifts, used by the drift-injection batch):
  F — Volatility Drift. Diurnal mean; residual std ramps from noise-start
      to noise-end over [drift-start, drift-start + drift-window]. Mean is
      ~stable, variance explodes. The divergence cell — point error stays
      low while interval coverage collapses.
  G — Level Drift. Diurnal shape; mean base ramps from rps-base to rps-peak
      over [drift-start, drift-start + drift-window]; noise stable. The
      control cell — rolling-MAE monitoring SHOULD detect this.
  H — Regime Change. Cosine crossfade from --from-pattern to --to-pattern
      over [transition-start, transition-start + transition-window].

Usage:
  python workload_gen.py A --duration 3600 --target http://localhost:30080
  python workload_gen.py B --duration 1800 --rps-base 30 --target http://localhost:30080
  python workload_gen.py C --duration 3600 --rps-peak 150
  python workload_gen.py D --duration 1200 --duty 60 --period 300
  python workload_gen.py E --duration 120 --constant-rps 50
  python workload_gen.py A --duration 86400 --trace-only  # generate trace without sending requests

  # Drift modes (trace-only — no cluster needed for smoke):
  python workload_gen.py F --duration 120 --trace-only \\
      --drift-start 30 --drift-window 60 --noise-start 3 --noise-end 25
  python workload_gen.py G --duration 120 --trace-only \\
      --drift-start 30 --drift-window 60 --rps-base 30 --rps-peak 90
  python workload_gen.py H --duration 120 --trace-only \\
      --from-pattern A --to-pattern D --transition-start 40 --transition-window 40

Outputs:
  - Real-time logs to stdout (per-second RPS, latency, errors)
  - JSON results file: outputs/workload_<pattern>_<timestamp>.json
  - CSV timeseries file: outputs/workload_<pattern>_<timestamp>.csv
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import argparse
import concurrent.futures
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Callable

import requests


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TickResult:
    """One second of load generation results."""
    elapsed_s: float
    target_rps: float
    actual_rps: float
    ok: int
    errors: int
    latencies_ms: List[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass
class RunResult:
    """Complete workload run results."""
    pattern: str
    duration_s: int
    total_requests: int
    total_errors: int
    error_rate: float
    overall_p50_ms: float
    overall_p95_ms: float
    overall_p99_ms: float
    ticks: List[TickResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Workload Pattern Functions  (rps = f(t_seconds))
# ═══════════════════════════════════════════════════════════════════

def pattern_diurnal(t: float, rps_base: float = 50, amplitude: float = 0.3,
                    period_s: float = 1800, noise_std: float = 5) -> float:
    """
    Pattern A: Diurnal Sinusoidal.
    r(t) = R_base * (1 + A * sin(2πt/P)) + ε_t
    """
    rps = rps_base * (1 + amplitude * math.sin(2 * math.pi * t / period_s))
    rps += random.gauss(0, noise_std)
    return max(0, rps)


def pattern_bursty(t: float, rps_base: float = 30,
                   spike_events: List[tuple] = None,
                   sigma_s: float = 45) -> float:
    """
    Pattern B: Bursty Spikes.
    r(t) = R_base + Σ S_i * exp(-(t - t_i)² / (2σ²))

    If spike_events is None, generates 5-8 random spike events.
    Each spike: (peak_rps, time_s)
    """
    rps = rps_base
    if spike_events is None:
        return rps  # Will be initialized on first call
    for peak, ti in spike_events:
        dt = t - ti
        rps += peak * math.exp(-(dt * dt) / (2 * sigma_s * sigma_s))
    return max(1, rps)


def pattern_batch_ramp(t: float, rps_base: float = 20, rps_peak: float = 150,
                       ramp_s: float = 300, sustain_s: float = 900) -> float:
    """
    Pattern C: Batch-Ramp.
    Linear ramp up (0→ramp_s), sustain (ramp_s→ramp_s+sustain_s), linear ramp down.

    Returns 0 if t is beyond the pattern window — caller should extend with
    repeated cycles or use a different base.
    """
    total = ramp_s + sustain_s + ramp_s  # ramp up + sustain + ramp down
    phase = t % total

    if phase < ramp_s:
        # Ramp up
        return rps_base + (rps_peak - rps_base) * (phase / ramp_s)
    elif phase < ramp_s + sustain_s:
        # Sustain
        return rps_peak
    else:
        # Ramp down
        down_phase = phase - ramp_s - sustain_s
        return rps_peak - (rps_peak - rps_base) * (down_phase / ramp_s)


def pattern_signaling(t: float, rps_pulse: float = 120, rps_idle: float = 5,
                      period_s: float = 300, duty_s: float = 60) -> float:
    """
    Pattern D: Signaling Pulse (square wave with duty cycle).
    Duty cycle: duty_s / period_s (default: 60/300 = 20%).
    """
    phase = t % period_s
    return rps_pulse if phase < duty_s else rps_idle


def pattern_constant(t: float, rps: float = 50) -> float:
    """
    Pattern E: Constant Flat Rate.
    Used for capacity profiling and saturation testing.
    """
    return rps


# ─── Drift modes ───────────────────────────────────────────────────

def _ramp_progress(t: float, start: float, window: float) -> float:
    """Linear 0→1 ramp over [start, start+window]; clamped outside."""
    if t <= start:
        return 0.0
    if t >= start + window:
        return 1.0
    return (t - start) / window


def pattern_volatility_drift(t: float, rps_base: float = 50,
                              amplitude: float = 0.3, period_s: float = 1800,
                              drift_start: float = 300, drift_window: float = 600,
                              noise_start: float = 3, noise_end: float = 25) -> float:
    """
    Pattern F: Volatility Drift.

    Diurnal mean kept stable; the residual std σ_t linearly interpolates
    from `noise_start` to `noise_end` over [drift_start, drift_start+drift_window].
    Point-error tracking on the mean stays in band; interval coverage of any
    fixed-width prediction collapses as σ_t grows — this is the canonical
    motivating cell for the C1 coverage-monitored controller (and for
    Adaptive Conformal Inference; Gibbs & Candès 2021, §2.2).
    """
    mean = rps_base * (1 + amplitude * math.sin(2 * math.pi * t / period_s))
    sigma = noise_start + (noise_end - noise_start) * _ramp_progress(
        t, drift_start, drift_window
    )
    return max(0, mean + random.gauss(0, sigma))


def pattern_level_drift(t: float, rps_base: float = 50, rps_peak: float = 150,
                         amplitude: float = 0.3, period_s: float = 1800,
                         drift_start: float = 300, drift_window: float = 600,
                         noise_std: float = 5) -> float:
    """
    Pattern G: Level Drift.

    Diurnal shape preserved; mean base linearly interpolates from `rps_base`
    to `rps_peak` over the drift window. Noise std stays constant. Mean
    shifts visibly, so a rolling-MAE error monitor SHOULD trigger — the
    control cell that separates "the error monitor catches it" (level
    drift) from "only the coverage monitor catches it" (volatility drift).
    """
    current_base = rps_base + (rps_peak - rps_base) * _ramp_progress(
        t, drift_start, drift_window
    )
    rps = current_base * (1 + amplitude * math.sin(2 * math.pi * t / period_s))
    rps += random.gauss(0, noise_std)
    return max(0, rps)


def pattern_regime_change(t: float, from_fn: Callable[[float], float],
                           to_fn: Callable[[float], float],
                           transition_start: float = 120,
                           transition_window: float = 120) -> float:
    """
    Pattern H: Regime Change.

    Cosine crossfade between two base patterns over the transition window:
        w_to(t) = 0.5 * (1 - cos(π · progress))
        r(t)    = (1 - w_to) · r_from(t)  +  w_to · r_to(t)
    Cosine (rather than linear) keeps the derivative continuous at both
    ends, avoiding artefacts at the transition boundary that would alias
    as scaling-event noise.
    """
    if t <= transition_start:
        return from_fn(t)
    if t >= transition_start + transition_window:
        return to_fn(t)
    progress = (t - transition_start) / transition_window
    w_to = 0.5 * (1 - math.cos(math.pi * progress))
    return (1 - w_to) * from_fn(t) + w_to * to_fn(t)


# ═══════════════════════════════════════════════════════════════════
# Rate-Controlled Request Dispatcher
# ═══════════════════════════════════════════════════════════════════

class WorkloadGenerator:
    """Generate time-varying load against a target endpoint."""

    def __init__(self, target_url: str, rps_fn: Callable[[float], float],
                 duration_s: int, pattern_name: str,
                 complexity: int = 50000, items: int = 2,
                 max_workers: int = 200, output_dir: str = "outputs",
                 trace_only: bool = False, seed: int = 0):
        self.target_url = f"{target_url.rstrip('/')}/api/process"
        self.health_url = f"{target_url.rstrip('/')}/health"
        self.rps_fn = rps_fn
        self.duration_s = duration_s
        self.pattern_name = pattern_name
        self.complexity = complexity
        self.items = items
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.trace_only = trace_only
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

        self.running = True
        self.ticks: List[TickResult] = []

    def _make_request(self) -> tuple:
        """Single request; returns (ok: bool, latency_ms: float)."""
        t0 = time.time()
        try:
            resp = requests.get(
                f"{self.target_url}?n={self.complexity}&items={self.items}",
                timeout=30
            )
            elapsed = (time.time() - t0) * 1000
            return resp.status_code == 200, elapsed
        except Exception:
            elapsed = (time.time() - t0) * 1000
            return False, elapsed

    def run(self):
        """Main loop: one tick per second for duration_s seconds."""
        random.seed(self.seed)
        mode_tag = "[TRACE-ONLY] " if self.trace_only else ""
        print(f"{mode_tag}Workload: {self.pattern_name} | {self.duration_s}s | target={self.target_url}")
        print(f"Complexity: n={self.complexity} items={self.items}")
        if not self.trace_only:
            print(f"{'time':>6s} {'target_rps':>10s} {'actual_rps':>10s} "
                  f"{'ok':>6s} {'err':>6s} {'p50_ms':>8s} {'p95_ms':>8s} {'p99_ms':>8s}")
        else:
            print(f"{'time':>6s} {'target_rps':>10s}")
        print("-" * 75)

        # Initialize bursty pattern spike events if needed
        if self.pattern_name == "B":
            # Use a fixed seed for reproducibility but vary across replicates
            random.seed(self.seed)
            n_spikes = random.randint(5, 8)
            spike_events = []
            for _ in range(n_spikes):
                ti = random.uniform(60, self.duration_s - 60)  # Avoid edges
                peak = random.uniform(40, 120)
                spike_events.append((peak, ti))
            # Wrap rps_fn to include spike events
            base_fn = self.rps_fn
            def rps_fn(t): return pattern_bursty(t, rps_base=30,
                                                  spike_events=spike_events)
            self.rps_fn = rps_fn

        # Trace-only mode uses simulated time (one tick per simulated second)
        # so a 24h trace generates in seconds, not 24h. The cluster-bound mode
        # below uses wall-clock pacing.
        if self.trace_only:
            print_every = max(1, self.duration_s // 60)  # ≤ 60 lines regardless of duration
            for tick_idx in range(self.duration_s):
                if not self.running:
                    break
                elapsed = float(tick_idx)
                target = max(0, int(round(self.rps_fn(elapsed))))
                self.ticks.append(TickResult(
                    elapsed_s=elapsed,
                    target_rps=target,
                    actual_rps=target,
                    ok=0, errors=0,
                    p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
                ))
                if tick_idx % print_every == 0 or tick_idx == self.duration_s - 1:
                    print(f"{elapsed:6.0f}s {target:10d}")
            result = self._summarize()
            self._save_outputs(result)
            return result

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            tick = 0
            while self.running and (time.time() - start) < self.duration_s:
                tick_start = time.time()
                elapsed = tick_start - start
                tick += 1

                # Compute target RPS for this second
                target = max(0, int(round(self.rps_fn(elapsed))))

                # Dispatch requests
                ok = 0
                errors = 0
                latencies = []
                futures = []
                for _ in range(target):
                    futures.append(pool.submit(self._make_request))

                # Collect results
                for f in concurrent.futures.as_completed(futures):
                    is_ok, lat_ms = f.result()
                    if is_ok:
                        ok += 1
                        latencies.append(lat_ms)
                    else:
                        errors += 1

                # Compute percentiles
                p50 = p95 = p99 = 0.0
                if latencies:
                    latencies.sort()
                    n = len(latencies)
                    p50 = latencies[n // 2]
                    p95 = latencies[int(n * 0.95)]
                    p99 = latencies[min(int(n * 0.99), n - 1)]

                actual_rps = ok / (time.time() - tick_start) if (time.time() - tick_start) > 0 else 0

                tick_result = TickResult(
                    elapsed_s=elapsed,
                    target_rps=target,
                    actual_rps=round(actual_rps, 1),
                    ok=ok,
                    errors=errors,
                    latencies_ms=latencies,
                    p50_ms=round(p50, 1),
                    p95_ms=round(p95, 1),
                    p99_ms=round(p99, 1),
                )
                self.ticks.append(tick_result)

                # Print per-second stats
                print(f"{elapsed:6.0f}s {target:10d} {actual_rps:10.1f} "
                      f"{ok:6d} {errors:6d} {p50:8.1f} {p95:8.1f} {p99:8.1f}")

                # Sleep until next second boundary
                work_time = time.time() - tick_start
                if work_time < 1.0:
                    time.sleep(1.0 - work_time)

        # Compute summary
        result = self._summarize()
        self._save_outputs(result)
        return result

    def _summarize(self) -> RunResult:
        all_latencies = []
        total_ok = 0
        total_err = 0
        for t in self.ticks:
            total_ok += t.ok
            total_err += t.errors
            all_latencies.extend(t.latencies_ms)

        if all_latencies:
            all_latencies.sort()
            n = len(all_latencies)
            p50 = all_latencies[n // 2]
            p95 = all_latencies[int(n * 0.95)]
            p99 = all_latencies[min(int(n * 0.99), n - 1)]
        else:
            p50 = p95 = p99 = 0.0

        return RunResult(
            pattern=self.pattern_name,
            duration_s=self.duration_s,
            total_requests=total_ok + total_err,
            total_errors=total_err,
            error_rate=total_err / max(1, total_ok + total_err),
            overall_p50_ms=round(p50, 1),
            overall_p95_ms=round(p95, 1),
            overall_p99_ms=round(p99, 1),
            ticks=self.ticks,
        )

    def _save_outputs(self, result: RunResult):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.output_dir}/workload_{self.pattern_name}_{ts}"

        # JSON summary
        summary = {
            "pattern": result.pattern,
            "duration_s": result.duration_s,
            "total_requests": result.total_requests,
            "total_errors": result.total_errors,
            "error_rate": result.error_rate,
            "p50_ms": result.overall_p50_ms,
            "p95_ms": result.overall_p95_ms,
            "p99_ms": result.overall_p99_ms,
            "config": {
                "target_url": self.target_url,
                "complexity": self.complexity,
                "items": self.items,
            }
        }
        with open(f"{prefix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # CSV timeseries
        with open(f"{prefix}_timeseries.csv", "w") as f:
            f.write("elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\n")
            for t in result.ticks:
                f.write(f"{t.elapsed_s:.1f},{t.target_rps},{t.actual_rps},"
                        f"{t.ok},{t.errors},{t.p50_ms},{t.p95_ms},{t.p99_ms}\n")

        print(f"\nSaved: {prefix}_summary.json")
        print(f"Saved: {prefix}_timeseries.csv")

        # Print summary
        print(f"\n{'='*50}")
        print(f"RUN COMPLETE — {result.pattern}")
        print(f"  Duration: {result.duration_s}s")
        print(f"  Requests: {result.total_requests} ({result.total_errors} errors, {result.error_rate:.2%})")
        print(f"  Latency: p50={result.overall_p50_ms}ms p95={result.overall_p95_ms}ms p99={result.overall_p99_ms}ms")
        print(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="InfoSys Benchmark — 4-Pattern Workload Generator"
    )
    parser.add_argument("pattern", choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                        help="Workload pattern to generate")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Duration in seconds (default: 3600 = 60 min)")
    parser.add_argument("--target", default="http://localhost:30080",
                        help="Frontend URL (default: http://localhost:30080)")
    parser.add_argument("--complexity", type=int, default=50000,
                        help="Compute complexity (sieve N, default: 50000)")
    parser.add_argument("--items", type=int, default=2,
                        help="Items per request for processor fanout (default: 2)")
    parser.add_argument("--max-workers", type=int, default=200,
                        help="Max thread pool workers (default: 200)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory for results (default: outputs)")
    parser.add_argument("--constant-rps", type=float, default=50,
                        help="Target RPS for Pattern E (constant flat rate)")
    parser.add_argument("--trace-only", action="store_true",
                        help="Generate RPS trace without sending requests (for synthetic training data)")

    # Pattern-specific params
    parser.add_argument("--rps-base", type=float, default=50,
                        help="Base RPS for Pattern A/B/C (default: 50)")
    parser.add_argument("--rps-peak", type=float, default=150,
                        help="Peak RPS for Pattern C (default: 150)")
    parser.add_argument("--rps-pulse", type=float, default=120,
                        help="Pulse RPS for Pattern D (default: 120)")
    parser.add_argument("--rps-idle", type=float, default=5,
                        help="Idle RPS for Pattern D (default: 5)")

    # Drift-mode params (F, G)
    parser.add_argument("--drift-start", type=float, default=300,
                        help="Seconds into the run when drift begins (F, G; default: 300)")
    parser.add_argument("--drift-window", type=float, default=600,
                        help="Duration of the drift ramp in seconds (F, G; default: 600)")
    parser.add_argument("--noise-start", type=float, default=3,
                        help="Pre-drift residual std for volatility drift F (default: 3)")
    parser.add_argument("--noise-end", type=float, default=25,
                        help="Post-drift residual std for volatility drift F (default: 25)")

    # Regime-change params (H)
    parser.add_argument("--from-pattern", choices=["A", "B", "C", "D", "E"], default="A",
                        help="Source pattern for regime change H (default: A)")
    parser.add_argument("--to-pattern", choices=["A", "B", "C", "D", "E"], default="D",
                        help="Target pattern for regime change H (default: D)")
    parser.add_argument("--transition-start", type=float, default=120,
                        help="Seconds into the run when the H crossfade begins (default: 120)")
    parser.add_argument("--transition-window", type=float, default=120,
                        help="Duration of the H crossfade in seconds (default: 120)")

    parser.add_argument("--seed", type=int, default=0, help="Recorded workload RNG seed")
    args = parser.parse_args()
    random.seed(args.seed)

    # B's spike train is sampled once upfront so it's stable across the
    # whole run (including when B appears as a sub-pattern inside H).
    # The legacy in-run() init covers the pattern == "B" case; we also
    # need spikes when H references B.
    spike_events_for_H = None
    if args.pattern == "H" and "B" in (args.from_pattern, args.to_pattern):
        random.seed(args.seed)
        n_spikes = random.randint(5, 8)
        spike_events_for_H = []
        for _ in range(n_spikes):
            ti = random.uniform(60, args.duration - 60)
            peak = random.uniform(40, 120)
            spike_events_for_H.append((peak, ti))

    # Build RPS function for each pattern
    pattern_fns = {
        "A": lambda t: pattern_diurnal(t, rps_base=args.rps_base),
        "B": lambda t: pattern_bursty(t, rps_base=args.rps_base),  # spikes init in run()
        "C": lambda t: pattern_batch_ramp(t, rps_base=args.rps_base,
                                           rps_peak=args.rps_peak),
        "D": lambda t: pattern_signaling(t, rps_pulse=args.rps_pulse,
                                          rps_idle=args.rps_idle),
        "E": lambda t: pattern_constant(t, rps=args.constant_rps),
        "F": lambda t: pattern_volatility_drift(
            t, rps_base=args.rps_base,
            drift_start=args.drift_start, drift_window=args.drift_window,
            noise_start=args.noise_start, noise_end=args.noise_end,
        ),
        "G": lambda t: pattern_level_drift(
            t, rps_base=args.rps_base, rps_peak=args.rps_peak,
            drift_start=args.drift_start, drift_window=args.drift_window,
        ),
    }

    if args.pattern == "H":
        # Build sub-pattern fns with the H-time-aligned B spikes if needed
        from_base = pattern_fns[args.from_pattern]
        to_base = pattern_fns[args.to_pattern]
        if args.from_pattern == "B":
            from_base = lambda t: pattern_bursty(  # noqa: E731
                t, rps_base=args.rps_base, spike_events=spike_events_for_H
            )
        if args.to_pattern == "B":
            to_base = lambda t: pattern_bursty(  # noqa: E731
                t, rps_base=args.rps_base, spike_events=spike_events_for_H
            )
        pattern_fns["H"] = lambda t: pattern_regime_change(
            t, from_base, to_base,
            transition_start=args.transition_start,
            transition_window=args.transition_window,
        )

    gen = WorkloadGenerator(
        target_url=args.target,
        rps_fn=pattern_fns[args.pattern],
        duration_s=args.duration,
        pattern_name=args.pattern,
        complexity=args.complexity,
        items=args.items,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        trace_only=args.trace_only,
        seed=args.seed,
    )

    # Graceful shutdown
    def handler(sig, frame):
        gen.running = False
        print("\nShutting down...")
    signal.signal(signal.SIGINT, handler)

    result = gen.run()
    return result


if __name__ == "__main__":
    main()
