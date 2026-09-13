"""Simple load generator to test HPA — runs concurrent requests to compute service."""

# Local artifact reference entrypoint; cluster behavior is unverified.
if True:
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import requests
import time
import concurrent.futures
import sys

URL = "http://localhost:8080/compute"
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 120
COMPLEXITY = int(sys.argv[3]) if len(sys.argv) > 3 else 200000

def hit():
    try:
        r = requests.get(f"{URL}?n={COMPLEXITY}", timeout=30)
        return r.status_code, r.elapsed.total_seconds()
    except Exception as e:
        return 0, str(e)

print(f"Starting load test: {CONCURRENCY} concurrent, {DURATION}s, n={COMPLEXITY}")
print(f"Target: {URL}")

start = time.time()
count = 0
errors = 0
latencies = []

with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    while time.time() - start < DURATION:
        futures = [executor.submit(hit) for _ in range(CONCURRENCY)]
        for f in concurrent.futures.as_completed(futures):
            code, lat = f.result()
            count += 1
            if code != 200:
                errors += 1
            elif isinstance(lat, float):
                latencies.append(lat)
        elapsed = time.time() - start
        if latencies:
            avg_lat = sum(latencies[-CONCURRENCY:]) / min(CONCURRENCY, len(latencies))
            print(f"  {elapsed:.0f}s | {count} req | avg {avg_lat*1000:.0f}ms | {errors} err", flush=True)

print(f"\nDone: {count} requests in {time.time()-start:.1f}s")
print(f"Errors: {errors}")
if latencies:
    latencies.sort()
    print(f"Latency: p50={latencies[len(latencies)//2]*1000:.0f}ms p95={latencies[int(len(latencies)*0.95)]*1000:.0f}ms")
