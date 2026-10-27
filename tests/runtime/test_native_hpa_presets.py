"""Native HPA spec and failure-path checks; all cluster calls are in-memory fakes."""
from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from confscale.native_hpa import PRESETS, hpa_manifest, make_native_hpa, stored_object_errors


@dataclass
class MethodConfig:
    name: str
    description: str = ""
    run_dir: Path | None = None


class Cluster:
    def __init__(self):
        self.items = []
        self.commands = []
        self.fail = None
        self.bad_inventory = False
        self.bad_readback = None
        self.keep_deleted = False
        self.ready = 1
        self.context = "kind-confscale-test"

    def kubectl(self, args, timeout=30):
        self.commands.append(args)
        if self.fail is not None and args[:len(self.fail)] == self.fail:
            return subprocess.CompletedProcess(args, 1, "", "injected failure")
        result = ""
        if args[:2] == ["apply", "-f"]:
            obj = json.loads(Path(args[2]).read_text())
            obj["metadata"]["uid"] = "api-server-uid"
            self.items.append(obj)
        elif args[:2] == ["get", "hpa"]:
            if args[2] == "-n":
                result = "invalid JSON" if self.bad_inventory else json.dumps({"items": self.items})
            else:
                obj = deepcopy(next(item for item in self.items if item["metadata"]["name"] == args[2]))
                if self.bad_readback:
                    self.bad_readback(obj)
                result = json.dumps(obj)
        elif args[:2] == ["delete", "hpa"] and not self.keep_deleted:
            self.items = [item for item in self.items if item["metadata"]["name"] != args[2]]
        elif args[:2] == ["get", "deployment"]:
            result = json.dumps({"spec": {"replicas": 1},
                                 "status": {"replicas": 1, "readyReplicas": self.ready}})
        return subprocess.CompletedProcess(args, 0, result, "")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "private-kubeconfig"))
    cluster = Cluster()
    methods = SimpleNamespace(MethodConfig=MethodConfig, kubectl=cluster.kubectl,
                              explicit_thread_kube_context=lambda: cluster.context)
    return cluster, methods


def method_for(runtime, tmp_path, preset="hpa-cpu-tuned"):
    method = make_native_hpa(preset, runtime[1])
    method.run_dir = tmp_path
    return method


@pytest.mark.parametrize("preset", PRESETS)
def test_exact_manifest_and_fresh_execution_identity(runtime, tmp_path, preset):
    first = method_for(runtime, tmp_path, preset)
    second = method_for(runtime, tmp_path, preset)
    assert first.hpa_name != second.hpa_name
    assert len(first.hpa_name) <= 63
    spec = first.manifest["spec"]
    assert spec["scaleTargetRef"] == {"apiVersion": "apps/v1", "kind": "Deployment", "name": "compute-worker"}
    assert (spec["minReplicas"], spec["maxReplicas"]) == (1, 20)
    assert spec["metrics"] == [{"type": "Resource", "resource": {"name": "cpu", "target": {
        "type": "Utilization", "averageUtilization": 70 if preset == "hpa-cpu-tuned" else 80}}}]
    if preset == "hpa-cpu-stock":
        assert "behavior" not in spec
    else:
        assert spec["behavior"] == {
            "scaleDown": {"stabilizationWindowSeconds": 60, "selectPolicy": "Max",
                          "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]},
            "scaleUp": {"stabilizationWindowSeconds": 0, "selectPolicy": "Max",
                        "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]},
        }


@pytest.mark.parametrize("preset", PRESETS)
def test_api_readback_and_checked_reset_preserve_identity(runtime, tmp_path, preset):
    cluster, _ = runtime
    method = method_for(runtime, tmp_path, preset)
    assert method.configure()
    assert method.hpa_uid == "api-server-uid"
    saved = json.loads((tmp_path / "hpa_stored.json").read_text())
    assert saved["metadata"]["uid"] == method.hpa_uid
    assert not stored_object_errors(saved, method.hpa_name, preset, uid=method.hpa_uid)
    cluster.commands.clear()
    assert method.reset()
    receipt = json.loads((tmp_path / "hpa_reset_check.json").read_text())
    assert receipt["ok"] and receipt["delete"]["ok"]
    assert receipt["hpa_uid"] == method.get_state()["hpa_uid"] == "api-server-uid"
    assert receipt["hpa_name"] == method.get_state()["hpa_name"] == method.hpa_name
    assert cluster.commands.index(receipt["delete"]["attempts"][-1]["readback"]["command"]) < \
        cluster.commands.index(receipt["scale"]["command"])
    assert not cluster.items
    with pytest.raises(ValueError, match="fresh method"):
        method.configure()


@pytest.mark.parametrize("preset", PRESETS)
def test_api_generated_metadata_and_status_do_not_change_preset(runtime, tmp_path, preset):
    method = method_for(runtime, tmp_path, preset)
    stored = method.manifest
    stored["metadata"].update(uid="api-server-uid", resourceVersion="42", generation=1,
                              creationTimestamp="2026-10-26T00:00:00Z", managedFields=[])
    stored["status"] = {"currentReplicas": 1, "desiredReplicas": 1,
                        "conditions": [{"type": "AbleToScale", "status": "True"}]}
    assert not stored_object_errors(stored, method.hpa_name, preset, uid="api-server-uid")
    if preset == "hpa-cpu-tuned":
        # Kubernetes defaults Max in present behavior branches; we submit it explicitly.
        assert all(rule["selectPolicy"] == "Max" for rule in stored["spec"]["behavior"].values())
    else:
        assert "behavior" not in stored["spec"]


@pytest.mark.parametrize("behavior", [None, {}])
def test_stock_rejects_present_behavior_even_null_or_empty(runtime, tmp_path, behavior):
    runtime[0].bad_readback = lambda obj: obj["spec"].update(behavior=behavior)
    method = method_for(runtime, tmp_path, "hpa-cpu-stock")
    assert not method.configure()
    reads = json.loads((tmp_path / "hpa_stored_readback.json").read_text())
    assert len(reads) == 1 and reads[0]["returncode"] == 0 and reads[0]["errors"]
    assert method.reset()


@pytest.mark.parametrize("mutation", [
    lambda obj: obj["metadata"].pop("uid"),
    lambda obj: obj["metadata"].update(namespace="other"),
    lambda obj: obj["spec"]["metrics"][0]["resource"]["target"].update(averageUtilization=50),
    lambda obj: obj["spec"]["behavior"]["scaleUp"]["policies"].append({
        "type": "Pods", "value": 4, "periodSeconds": 15}),
])
def test_missing_uid_or_changed_spec_or_identity_fails(runtime, tmp_path, mutation):
    runtime[0].bad_readback = mutation
    assert not method_for(runtime, tmp_path).configure()


@pytest.mark.parametrize("command", [["scale"], ["rollout"], ["apply"], ["get", "hpa"]])
def test_configuration_command_failures_propagate(runtime, tmp_path, command):
    runtime[0].fail = command
    assert not method_for(runtime, tmp_path).configure()
    journal = json.loads((tmp_path / "hpa_configure_check.json").read_text())
    last = journal["commands"][-1]
    assert not journal["ok"] and last["state"] == "failed"
    assert last["returncode"] == 1 and last["stderr"] == "injected failure"
    assert last["time"] <= last["end_time"]


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
def test_interrupted_apply_is_journaled_before_interruption_propagates(runtime, tmp_path, failure):
    cluster, methods = runtime
    def interrupt(args, timeout=30):
        if args[0] == "apply":
            pending = json.loads((tmp_path / "hpa_configure_check.json").read_text())
            assert pending["commands"][-1]["command"] == args
            assert pending["commands"][-1]["state"] == "running"
            raise failure("interrupted apply")
        return cluster.kubectl(args, timeout)
    methods.kubectl = interrupt
    with pytest.raises(failure):
        method_for(runtime, tmp_path).configure()
    record = json.loads((tmp_path / "hpa_configure_check.json").read_text())
    assert record["ok"] is False and record["interrupted"] is True
    assert record["commands"][-1]["state"] == "interrupted"
    assert record["commands"][-1]["interrupted"] is True
    assert failure.__name__ in record["commands"][-1]["error"]


def test_second_readback_interruption_preserves_first_attempt(runtime, tmp_path):
    cluster, methods = runtime
    calls = 0
    def readback(args, timeout=30):
        nonlocal calls
        if args[:2] == ["get", "hpa"] and args[2] != "-n":
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess(args, 1, "", "temporary readback failure")
            raise KeyboardInterrupt
        return cluster.kubectl(args, timeout)
    methods.kubectl = readback
    with pytest.raises(KeyboardInterrupt):
        method_for(runtime, tmp_path).configure()
    reads = json.loads((tmp_path / "hpa_stored_readback.json").read_text())
    assert len(reads) == 1 and reads[0]["stderr"] == "temporary readback failure"
    journal = json.loads((tmp_path / "hpa_configure_check.json").read_text())
    assert journal["commands"][-1]["state"] == "interrupted"


def test_interrupted_reset_preserves_current_command(runtime, tmp_path):
    cluster, methods = runtime
    method = method_for(runtime, tmp_path)
    assert method.configure()
    def interrupted(args, timeout=30):
        if args[0] == "delete":
            raise KeyboardInterrupt
        return cluster.kubectl(args, timeout)
    methods.kubectl = interrupted
    with pytest.raises(KeyboardInterrupt):
        method.reset()
    record = json.loads((tmp_path / "hpa_reset_check.json").read_text())
    assert record["ok"] is False and record["interrupted"] is True
    assert record["commands"][-1]["command"][0] == "delete"
    assert record["commands"][-1]["state"] == "interrupted"


@pytest.mark.parametrize("operation", ["configure", "reset"])
@pytest.mark.parametrize("binary", [False, True])
def test_timeout_preserves_available_partial_output(runtime, tmp_path, operation, binary):
    cluster, methods = runtime
    method = method_for(runtime, tmp_path)
    if operation == "reset":
        assert method.configure()
    stdout = b"partial response\xff" if binary else "partial response"
    stderr = b"partial diagnostics\n" if binary else "partial diagnostics\n"
    failed_command = "apply" if operation == "configure" else "delete"
    def timeout(args, timeout=30):
        if args[0] == failed_command:
            raise subprocess.TimeoutExpired(args, timeout, output=stdout, stderr=stderr)
        return cluster.kubectl(args, timeout)
    methods.kubectl = timeout
    assert getattr(method, operation)() is False
    journal = json.loads((tmp_path / f"hpa_{operation}_check.json").read_text())
    failed = [row for row in journal["commands"] if row["command"][0] == failed_command]
    assert failed
    for row in failed:
        assert row["state"] == "failed" and "TimeoutExpired" in row["error"]
        assert row["stdout"] == (stdout.decode("utf-8", errors="replace") if binary else stdout)
        assert row["stderr"] == "partial diagnostics\n"
        assert "returncode" not in row and "interrupted" not in row


def test_contradictory_successful_readback_stops_before_retry(runtime, tmp_path):
    cluster, methods = runtime
    calls = 0
    def inconsistent(args, timeout=30):
        nonlocal calls
        result = cluster.kubectl(args, timeout)
        if args[:2] == ["get", "hpa"] and args[2] != "-n":
            calls += 1
            if calls == 1:
                obj = json.loads(result.stdout)
                obj["spec"]["maxReplicas"] = 21
                result.stdout = json.dumps(obj)
        return result
    methods.kubectl = inconsistent
    method = method_for(runtime, tmp_path)
    assert not method.configure()
    assert calls == 1
    assert method.hpa_uid == "api-server-uid"
    assert json.loads((tmp_path / "hpa_stored.json").read_text())["spec"]["maxReplicas"] == 21
    assert not json.loads((tmp_path / "hpa_configure_check.json").read_text())["ok"]
    assert method.reset()


def test_command_routing_is_bound_without_recording_private_path(runtime, tmp_path):
    method = method_for(runtime, tmp_path)
    assert method.configure() and method.reset()
    state = method.get_state()
    assert state["context"] == runtime[0].context
    assert len(state["kubeconfig_path_sha256"]) == 64
    record = json.loads((tmp_path / "hpa_configure_check.json").read_text())
    for row in [record, *record["commands"], method.reset_evidence]:
        assert row["context"] == state["context"]
        assert row["kubeconfig_path_sha256"] == state["kubeconfig_path_sha256"]
        assert "kubeconfig" not in row
    assert "private-kubeconfig" not in (tmp_path / "hpa_configure_check.json").read_text()


def test_default_context_is_not_an_explicit_thread_context(runtime, tmp_path):
    runtime[0].context = None
    runtime[1].current_kube_context = lambda: "kind-default"
    with pytest.raises(ValueError, match="explicit context"):
        method_for(runtime, tmp_path).configure()
    assert not runtime[0].commands


@pytest.mark.parametrize("failure", ["delete", "inventory_error", "invalid_json", "survivor", "scale", "rollout", "deployment", "unready"])
def test_reset_failure_cannot_report_success(runtime, tmp_path, failure):
    cluster, _ = runtime
    method = method_for(runtime, tmp_path)
    assert method.configure()
    cluster.commands.clear()
    if failure in ("delete", "scale", "rollout"):
        cluster.fail = [failure]
    elif failure == "inventory_error":
        cluster.fail = ["get", "hpa"]
    elif failure == "invalid_json":
        cluster.bad_inventory = True
    elif failure == "survivor":
        cluster.keep_deleted = True
    elif failure == "deployment":
        cluster.fail = ["get", "deployment"]
    else:
        cluster.ready = 0
    assert method.reset() is False
    assert not json.loads((tmp_path / "hpa_reset_check.json").read_text())["ok"]
    if failure in ("delete", "inventory_error", "invalid_json", "survivor"):
        assert not any(command[0] == "scale" for command in cluster.commands)


def test_reset_refuses_replaced_uid(runtime, tmp_path):
    cluster, _ = runtime
    method = method_for(runtime, tmp_path)
    assert method.configure()
    cluster.items[0]["metadata"]["uid"] = "replacement-uid"
    cluster.commands.clear()
    assert not method.reset()
    assert not any(command[0] in ("delete", "scale") for command in cluster.commands)


def test_existing_hpa_is_preserved_and_prevents_configuration(runtime, tmp_path):
    cluster, _ = runtime
    existing = hpa_manifest("confscale-cpu-stock-" + "0" * 32, "hpa-cpu-stock")
    cluster.items = [existing]
    assert not method_for(runtime, tmp_path).configure()
    assert cluster.items == [existing]
    assert all(command[0] == "get" for command in cluster.commands)


def test_conflicting_hpa_after_delete_prevents_scale(runtime, tmp_path):
    cluster, _ = runtime
    method = method_for(runtime, tmp_path)
    assert method.configure()
    cluster.items.append(hpa_manifest("confscale-cpu-stock-" + "0" * 32, "hpa-cpu-stock"))
    cluster.commands.clear()
    assert not method.reset()
    assert not any(command[0] == "scale" for command in cluster.commands)


@pytest.mark.parametrize("change", ["context", "kubeconfig", "thread"])
def test_reset_cannot_cross_runtime_identity(runtime, tmp_path, monkeypatch, change):
    cluster, _ = runtime
    method = method_for(runtime, tmp_path)
    assert method.configure()
    cluster.commands.clear()
    if change == "context":
        cluster.context = "kind-other"
    elif change == "kubeconfig":
        monkeypatch.setenv("KUBECONFIG", "different-private-kubeconfig")
    else:
        monkeypatch.setattr("confscale.native_hpa.threading.get_ident", lambda: -1)
    with pytest.raises(RuntimeError, match="context, or private kubeconfig changed"):
        method.reset()
    assert not cluster.commands


def test_private_kubeconfig_is_required_before_commands(runtime, tmp_path, monkeypatch):
    monkeypatch.delenv("KUBECONFIG")
    method = method_for(runtime, tmp_path)
    with pytest.raises(ValueError, match="private KUBECONFIG"):
        method.configure()
    assert not method.reset()
    assert not runtime[0].commands


def test_pure_import_does_not_load_reference_runtime(monkeypatch):
    # Block all imports from the opt-in runtime while re-importing this module.
    real_import = __import__
    def guarded_import(name, *args, **kwargs):
        if name.startswith(("orchestrator", "collector", "predictor", "uq")):
            raise AssertionError("Imported opt-in runtime")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded_import)
    module = importlib.reload(sys.modules["confscale.native_hpa"])
    with pytest.raises(ValueError, match="Unknown native HPA preset"):
        module.make_native_hpa("unknown", Mock())
