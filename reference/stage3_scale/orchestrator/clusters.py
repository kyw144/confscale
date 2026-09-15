#!/usr/bin/env python3
"""Cluster Slot Management — assign per-worker kind clusters and host ports."""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


DEFAULT_CLUSTER_PREFIX = "p3-experiments"
DEFAULT_BASE_FRONTEND_PORT = 31080      # NOT 30080 — leaves the existing single cluster usable
DEFAULT_BASE_PROMETHEUS_PORT = 9190     # NOT 9090 — same reason
DEFAULT_BASE_INGRESS_HTTPS_PORT = 31443


@dataclass
class WorkerSlot:
    """Resources owned by one parallel worker."""

    worker_id: int
    cluster_name: str
    kube_context: str
    frontend_port: int
    prometheus_port: int
    ingress_https_port: int = 0

    @property
    def frontend_url(self) -> str:
        return f"http://localhost:{self.frontend_port}"

    @property
    def prometheus_url(self) -> str:
        return f"http://localhost:{self.prometheus_port}"

    def label(self) -> str:
        return f"w{self.worker_id}[{self.cluster_name}]"


def make_slots(
    num_workers: int,
    cluster_prefix: str = DEFAULT_CLUSTER_PREFIX,
    base_frontend_port: int = DEFAULT_BASE_FRONTEND_PORT,
    base_prometheus_port: int = DEFAULT_BASE_PROMETHEUS_PORT,
    base_ingress_https_port: int = DEFAULT_BASE_INGRESS_HTTPS_PORT,
) -> list[WorkerSlot]:
    """Build N WorkerSlots with offset host ports."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    return [
        WorkerSlot(
            worker_id=i,
            cluster_name=f"{cluster_prefix}-w{i}",
            kube_context=f"kind-{cluster_prefix}-w{i}",
            frontend_port=base_frontend_port + i,
            prometheus_port=base_prometheus_port + i,
            ingress_https_port=base_ingress_https_port + i,
        )
        for i in range(num_workers)
    ]


def render_kind_config(slot: WorkerSlot, output_path: Path) -> Path:
    """Render kind-config-w{i}.yaml for one slot, mirroring the existing kind-config.yaml but with offset host port mappings."""
    config: dict[str, Any] = {
        "kind": "Cluster",
        "apiVersion": "kind.x-k8s.io/v1alpha4",
        "nodes": [
            {
                "role": "control-plane",
                "extraPortMappings": [
                    {
                        "containerPort": 30080,
                        "hostPort": slot.frontend_port,
                        "protocol": "TCP",
                    },
                    {
                        "containerPort": 30443,
                        "hostPort": slot.ingress_https_port,
                        "protocol": "TCP",
                    },
                ],
                "kubeadmConfigPatches": [
                    "kind: InitConfiguration\n"
                    "nodeRegistration:\n"
                    "  kubeletExtraArgs:\n"
                    '    node-labels: "ingress-ready=true"\n'
                ],
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))
    return output_path


def kind_cluster_exists(cluster_name: str) -> bool:
    """Return True if `kind get clusters` lists this cluster."""
    result = subprocess.run(
        ["kind", "get", "clusters"], capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        logger.warning("kind get clusters failed: %s", result.stderr.strip())
        return False
    return cluster_name in result.stdout.split()


def kind_create_cluster(slot: WorkerSlot, kind_config_path: Path, timeout: int = 600) -> bool:
    """Create a kind cluster from a rendered config."""
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise RuntimeError("Reference cluster mutation disabled; read README.md#cluster-runs")
    if kind_cluster_exists(slot.cluster_name):
        logger.info("Cluster %s already exists — skipping create", slot.cluster_name)
        return True
    cmd = [
        "kind", "create", "cluster",
        "--name", slot.cluster_name,
        "--config", str(kind_config_path),
    ]
    logger.info("Creating cluster %s ...", slot.cluster_name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("kind create failed for %s:\n%s", slot.cluster_name, result.stderr)
        return False
    logger.info("Cluster %s created", slot.cluster_name)
    return True


def kind_delete_cluster(slot: WorkerSlot, timeout: int = 120) -> bool:
    """Delete a kind cluster."""
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise RuntimeError("Reference cluster mutation disabled; read README.md#cluster-runs")
    if not kind_cluster_exists(slot.cluster_name):
        logger.info("Cluster %s does not exist — nothing to delete", slot.cluster_name)
        return True
    cmd = ["kind", "delete", "cluster", "--name", slot.cluster_name]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("kind delete failed for %s:\n%s", slot.cluster_name, result.stderr)
        return False
    logger.info("Cluster %s deleted", slot.cluster_name)
    return True
