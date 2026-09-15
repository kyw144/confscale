from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.methods import HPAMethod, get_method  # noqa: E402


def test_default_is_v1_path():
    m = HPAMethod()
    assert m.downscale_stabilization_s is None
    assert m.upscale_stabilization_s == 0
    st = m.get_state()
    assert st["hpa_api_version"] == "autoscaling/v1"
    assert st["cpu_target_pct"] == 50  # original default that produced 67,806


def test_v2_state_when_window_set():
    m = HPAMethod(downscale_stabilization_s=60, cpu_target=70)
    st = m.get_state()
    assert st["hpa_api_version"] == "autoscaling/v2"
    assert st["downscale_stabilization_s"] == 60
    assert st["upscale_stabilization_s"] == 0
    assert st["cpu_target_pct"] == 70


def test_v2_manifest_structure():
    m = HPAMethod(name="hpa-tuned-u70-s60", cpu_target=70,
                  downscale_stabilization_s=60, min_replicas=1, max_replicas=20)
    doc = yaml.safe_load(m._render_v2_hpa_manifest("infosys-benchmark"))

    assert doc["apiVersion"] == "autoscaling/v2"
    assert doc["kind"] == "HorizontalPodAutoscaler"
    assert doc["metadata"]["name"] == "compute-worker"
    assert doc["metadata"]["namespace"] == "infosys-benchmark"

    spec = doc["spec"]
    assert spec["scaleTargetRef"]["name"] == "compute-worker"
    assert spec["minReplicas"] == 1
    assert spec["maxReplicas"] == 20

    metric = spec["metrics"][0]
    assert metric["type"] == "Resource"
    assert metric["resource"]["name"] == "cpu"
    assert metric["resource"]["target"]["type"] == "Utilization"
    assert metric["resource"]["target"]["averageUtilization"] == 70

    beh = spec["behavior"]
    assert beh["scaleDown"]["stabilizationWindowSeconds"] == 60
    assert beh["scaleUp"]["stabilizationWindowSeconds"] == 0  # react immediately up
    # default-matching policy so v2@300 == v1 default behaviour
    assert beh["scaleDown"]["policies"][0] == {"type": "Percent", "value": 100, "periodSeconds": 15}


def test_setattr_override_path_matches_resolve_method_spec():
    m = get_method("hpa-reactive")
    for k, v in {"name": "hpa-tuned-u50-s120", "cpu_target": 50,
                 "downscale_stabilization_s": 120}.items():
        assert hasattr(m, k), f"field {k} missing — setattr path would silently skip it"
        setattr(m, k, v)
    doc = yaml.safe_load(m._render_v2_hpa_manifest("infosys-benchmark"))
    assert doc["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 50
    assert doc["spec"]["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == 120
    assert m.get_state()["hpa_api_version"] == "autoscaling/v2"


def test_reproduction_anchor_window_300():
    m = HPAMethod(name="hpa-anchor-u50-s300", cpu_target=50,
                  downscale_stabilization_s=300)
    doc = yaml.safe_load(m._render_v2_hpa_manifest("infosys-benchmark"))
    assert doc["spec"]["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == 300
    assert doc["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 50


def test_get_method_returns_fresh_hpa_copies():
    from orchestrator.methods import METHOD_REGISTRY
    a = get_method("hpa-reactive")
    b = get_method("hpa-reactive")
    assert a is not b, "get_method must return a fresh HPAMethod, not the singleton"
    a.cpu_target = 70
    a.downscale_stabilization_s = 60
    a.name = "hpa-tuned-u70-s60"
    assert b.cpu_target == 50 and b.downscale_stabilization_s is None
    assert b.name == "hpa-reactive"
    assert METHOD_REGISTRY["hpa-reactive"].cpu_target == 50
    assert METHOD_REGISTRY["hpa-reactive"].downscale_stabilization_s is None
    assert METHOD_REGISTRY["hpa-reactive"].name == "hpa-reactive"


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["python", "-m", "pytest", __file__, "-q"]))
