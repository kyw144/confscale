"""
Prometheus TSDB snapshot utilities.

Provides functions to trigger and copy Prometheus TSDB snapshots
at the end of experimental runs for archival and re-analysis.
"""

import logging
import subprocess
import time
import urllib.request
import urllib.error
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def trigger_snapshot(prometheus_url: str, max_retries: int = 3,
                     backoff_s: float = 3.0) -> Optional[str]:
    """
    Trigger a Prometheus TSDB snapshot via admin API.

    Args:
        prometheus_url: Prometheus HTTP API base URL (e.g., http://localhost:9090)
        max_retries: Number of retry attempts on connection failure
        backoff_s: Backoff multiplier in seconds

    Returns:
        Snapshot name string (e.g., '20260509T120000Z-abcdef'), or None on failure.

    Raises:
        Does NOT raise — failures are logged and return None.
        The snapshot is a backup; query-based extraction is the primary data source.
    """
    url = f"{prometheus_url.rstrip('/')}/api/v1/admin/tsdb/snapshot"
    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, method='POST')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            if data.get("status") == "success":
                snap_name = data["data"]["name"]
                logger.info("TSDB snapshot triggered: %s", snap_name)
                return snap_name
            else:
                error_msg = data.get("error", "unknown error")
                logger.error("Snapshot API returned error: %s", error_msg)
                return None
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300] if e.fp else ""
            last_error = f"HTTP {e.code}: {body}"
            logger.warning("Snapshot API request failed (attempt %d/%d): %s",
                           attempt + 1, max_retries, last_error)
            if e.code == 500 and "disabled" in body:
                logger.error("Admin API is disabled. Set enableAdminAPI: true in Prometheus CRD.")
                return None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_error = str(e)
            logger.warning("Snapshot connection failed (attempt %d/%d): %s",
                           attempt + 1, max_retries, e)
        if attempt < max_retries - 1:
            time.sleep(backoff_s * (attempt + 1))
    logger.error("Snapshot trigger failed after %d attempts: %s", max_retries, last_error)
    return None


def copy_snapshot(snapshot_name: str, output_dir: Path,
                  pod_name: str = "prometheus-prometheus-kube-prometheus-prometheus-0",
                  namespace: str = "monitoring",
                  container: str = "prometheus",
                  kube_context: str = "kind-p3-experiments") -> bool:
    """
    Copy a TSDB snapshot from the Prometheus pod to a local directory.

    Args:
        snapshot_name: Snapshot name from trigger_snapshot()
        output_dir: Local directory to copy the snapshot into
        pod_name: Prometheus pod name
        namespace: Kubernetes namespace
        container: Container name within the pod
        kube_context: kubectl context name

    Returns:
        True on success, False on failure.
    """
    snapshot_path = f"/prometheus/snapshots/{snapshot_name}"
    dest_dir = output_dir / "prometheus_snapshot"
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kubectl", f"--context={kube_context}",
        "cp", "-n", namespace, "-c", container,
        f"{pod_name}:{snapshot_path}",
        str(dest_dir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Verify the copy
            if any(dest_dir.iterdir()):
                logger.info("TSDB snapshot copied to %s", dest_dir)
                return True
            else:
                logger.error("Snapshot copy completed but directory is empty: %s", dest_dir)
                return False
        else:
            logger.error("kubectl cp failed (exit code %d): %s",
                         result.returncode, result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        logger.error("Snapshot copy timed out after 60s")
        return False
    except FileNotFoundError:
        logger.error("kubectl not found in PATH")
        return False
    except Exception as e:
        logger.error("Snapshot copy failed: %s", e)
        return False


def capture_snapshot(prometheus_url: str, output_dir: Path,
                     **kwargs) -> dict:
    """
    Trigger and copy a TSDB snapshot. Convenience wrapper.

    Returns:
        dict with 'success' (bool), 'snapshot_name' (str or None),
        and 'error' (str or None).
    """
    snap_name = trigger_snapshot(prometheus_url)
    if snap_name is None:
        return {"success": False, "snapshot_name": None, "error": "trigger_failed"}
    success = copy_snapshot(snap_name, output_dir, **kwargs)
    if not success:
        return {"success": False, "snapshot_name": snap_name, "error": "copy_failed"}
    return {"success": True, "snapshot_name": snap_name, "error": None}
