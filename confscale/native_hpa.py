"""Execution-scoped native HPA presets, with checked API readback and cleanup."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from uuid import uuid4


NAMESPACE = "infosys-benchmark"
PRESETS = ("hpa-cpu-tuned", "hpa-cpu-stock")


def hpa_manifest(name, preset, namespace=NAMESPACE):
    """Return the exact requested API spec; stock deliberately omits behavior."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown native HPA preset: {preset}")
    prefix = "confscale-" + preset.removeprefix("hpa-") + "-"
    if not isinstance(name, str) or re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", name) is None:
        raise ValueError("Native HPA requires a preset-specific execution UUID name")
    spec = {
        "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": "compute-worker"},
        "minReplicas": 1,
        "maxReplicas": 20,
        "metrics": [{"type": "Resource", "resource": {
            "name": "cpu", "target": {"type": "Utilization",
                                     "averageUtilization": 70 if preset == "hpa-cpu-tuned" else 80}}}],
    }
    if preset == "hpa-cpu-tuned":
        # Spell out server-defaulted policy selection for exact stored-spec checks.
        spec["behavior"] = {
            "scaleDown": {"stabilizationWindowSeconds": 60,
                          "selectPolicy": "Max",
                          "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]},
            "scaleUp": {"stabilizationWindowSeconds": 0,
                        "selectPolicy": "Max",
                        "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]},
        }
    return {"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": name, "namespace": namespace}, "spec": spec}


def stored_object_errors(obj, name, preset, namespace=NAMESPACE, uid=None):
    """Validate API identity and exact preset without filling omitted defaults."""
    expected = hpa_manifest(name, preset, namespace)
    if not isinstance(obj, dict):
        return ["Stored HPA object missing"]
    errors = []
    if any(obj.get(key) != expected[key] for key in ("apiVersion", "kind", "spec")):
        errors.append("Stored HPA API kind or preset spec differs")
    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        return errors + ["Stored HPA metadata missing"]
    if any(meta.get(key) != expected["metadata"][key] for key in ("name", "namespace")):
        errors.append("Stored HPA identity differs")
    if not isinstance(meta.get("uid"), str) or not meta["uid"] or (uid is not None and meta["uid"] != uid):
        errors.append("Stored HPA UID missing or differs")
    return errors


def _object(record):
    if record.get("returncode") != 0:
        return None
    try:
        obj = json.loads(record["stdout"])
        return obj if isinstance(obj, dict) else None
    except (KeyError, TypeError, ValueError):
        return None


def _hpas(record):
    obj = _object(record)
    if obj is None or not isinstance(obj.get("items"), list):
        return None
    items = obj["items"]
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("metadata"), dict)
                or not isinstance(item["metadata"].get("name"), str)
                or not item["metadata"]["name"] or not isinstance(item.get("spec"), dict)
                or not isinstance(item["spec"].get("scaleTargetRef"), dict)):
            return None
    return items


def _targets_worker(item):
    target = item["spec"]["scaleTargetRef"]
    return target.get("kind") == "Deployment" and target.get("name") == "compute-worker"


def make_native_hpa(preset, methods_module):
    """Inject the opt-in runtime so importing this module never loads it."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown native HPA preset: {preset}")

    class NativeHPAMethod(methods_module.MethodConfig):
        def __init__(self):
            super().__init__(name=preset, description="Native Kubernetes CPU HPA; " + preset)
            self.hpa_name = "confscale-" + preset.removeprefix("hpa-") + "-" + uuid4().hex
            self.hpa_uid = None
            self.reset_evidence = {}
            self.configure_evidence = {}
            self._configure_record = None
            self._reset_record = None
            self._namespace = NAMESPACE
            self._identity = None
            self._configure_started = False

        @property
        def manifest(self):
            return hpa_manifest(self.hpa_name, preset, self._namespace)

        def _current_identity(self):
            return (threading.get_ident(), methods_module.explicit_thread_kube_context(), os.environ.get("KUBECONFIG"))

        def _routing(self):
            return {"context": self._identity[1] if self._identity else None,
                    "kubeconfig_path_sha256": hashlib.sha256(self._identity[2].encode()).hexdigest()
                    if self._identity else None}

        def _command(self, command, timeout=30):
            if self._identity is None or self._current_identity() != self._identity:
                raise RuntimeError("Native HPA thread, context, or private kubeconfig changed")
            row = {"command": command, "time": datetime.now(timezone.utc).isoformat(),
                   "state": "running", **self._routing()}
            active = self._configure_record if self._configure_record is not None else self._reset_record
            filename = "hpa_configure_check.json" if self._configure_record is not None else "hpa_reset_check.json"
            if active is not None:
                active["commands"].append(row)
                self._save(filename, active)
            try:
                result = methods_module.kubectl(command, timeout=timeout)
                row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                           state="completed" if result.returncode == 0 else "failed")
            except BaseException as exc:
                interrupted = not isinstance(exc, Exception)
                row.update(error=f"{type(exc).__name__}: {exc}",
                           state="interrupted" if interrupted else "failed")
                for key in ("stdout", "stderr"):
                    value = getattr(exc, key, None)
                    if key == "stdout" and value is None:
                        value = getattr(exc, "output", None)
                    row[key] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value) if value is not None else ""
                if interrupted:
                    row["interrupted"] = True
                    raise
            finally:
                row["end_time"] = datetime.now(timezone.utc).isoformat()
                if active is not None:
                    self._save(filename, active)
            return row

        def _save(self, name, value, exclusive=False):
            path = Path(self.run_dir) / name
            pending = path if exclusive else path.with_name("." + name + ".tmp")
            with pending.open("x" if exclusive else "w") as output:
                json.dump(value, output, indent=2)
                output.write("\n")
            if not exclusive:
                pending.replace(path)

        def _inventory(self):
            return self._command(["get", "hpa", "-n", self._namespace, "-o", "json"])

        def configure(self, namespace=NAMESPACE):
            if self.run_dir is None:
                raise ValueError("Native HPA configure requires a per-cell run_dir")
            if self._configure_started:
                raise ValueError("Native HPA requires a fresh method instance for every execution")
            if self.name != preset:
                raise ValueError("Native HPA preset name cannot be overridden")
            self._configure_started = True
            self._namespace = namespace
            record = self.configure_evidence = {"hpa_name": self.hpa_name, "hpa_uid": None,
                "namespace": namespace, **self._routing(), "ok": False, "commands": []}
            self._configure_record = record
            try:
                identity = self._current_identity()
                if not identity[1] or not identity[2]:
                    raise ValueError("Native HPA requires an explicit context and private KUBECONFIG")
                self._identity = identity
                record.update(self._routing())
                record["ok"] = self._configure(namespace)
                return record["ok"]
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    record["interrupted"] = True
                raise
            finally:
                record["hpa_uid"] = self.hpa_uid
                try:
                    self._save("hpa_configure_check.json", record)
                finally:
                    self._configure_record = None

        def _configure(self, namespace):
            inventory = self._inventory()
            self._save("hpa_initial_inventory.json", inventory, exclusive=True)
            items = _hpas(inventory)
            if items is None or any(_targets_worker(item) for item in items):
                return False
            scale = self._command(["scale", "deployment", "compute-worker", "--replicas=1", "-n", namespace])
            if scale.get("returncode") != 0:
                return False
            rollout = self._command(["rollout", "status", "deployment", "compute-worker",
                                     "-n", namespace, "--timeout=120s"], timeout=130)
            if rollout.get("returncode") != 0:
                return False
            # JSON is valid YAML, avoiding a runtime dependency for pure manifests.
            self._save("hpa_v2.yaml", self.manifest, exclusive=True)
            manifest_path = Path(self.run_dir) / "hpa_v2.yaml"
            self._configure_record.update(manifest_path=str(manifest_path),
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            applied = self._command(["apply", "-f", str(manifest_path)])
            if applied.get("returncode") != 0:
                return False
            reads = []
            obj = None
            for _ in range(2):
                row = self._command(["get", "hpa", self.hpa_name, "-n", namespace, "-o", "json"])
                obj = _object(row)
                row["errors"] = stored_object_errors(obj, self.hpa_name, preset, namespace)
                reads.append(row)
                self._save("hpa_stored_readback.json", reads, exclusive=len(reads) == 1)
                if row.get("returncode") == 0 or not row["errors"]:
                    break
            if obj is not None:
                self._save("hpa_stored.json", obj, exclusive=True)
                meta = obj.get("metadata")
                self.hpa_uid = meta.get("uid") if isinstance(meta, dict) else None
            return not reads[-1]["errors"]

        def reset(self, namespace=NAMESPACE):
            record = self.reset_evidence = {"hpa_name": self.hpa_name, "hpa_uid": self.hpa_uid,
                                            "namespace": namespace, **self._routing(), "ok": False, "commands": []}
            self._reset_record = record
            try:
                if namespace != self._namespace or self._identity is None:
                    return False
                record["inventory"] = self._inventory()
                items = _hpas(record["inventory"])
                if items is None:
                    return False
                own = [item for item in items if item["metadata"]["name"] == self.hpa_name]
                if own and self.hpa_uid is not None and own[0]["metadata"].get("uid") != self.hpa_uid:
                    return False
                deletion = record["delete"] = {"ok": False, "attempts": []}
                for _ in range(2):
                    deleted = self._command(["delete", "hpa", self.hpa_name, "-n", namespace,
                                             "--ignore-not-found=true"])
                    readback = self._inventory()
                    deletion["attempts"].append({"delete": deleted, "readback": readback})
                    remaining = _hpas(readback)
                    deletion["ok"] = bool(deleted.get("returncode") == 0 and remaining is not None
                                          and all(item["metadata"]["name"] != self.hpa_name
                                                  and not _targets_worker(item) for item in remaining))
                    if deletion["ok"]:
                        break
                if not deletion["ok"]:
                    return False
                record["scale"] = self._command(["scale", "deployment", "compute-worker",
                                                  "--replicas=1", "-n", namespace])
                if record["scale"].get("returncode") != 0:
                    return False
                record["rollout"] = self._command(["rollout", "status", "deployment", "compute-worker",
                                                    "-n", namespace, "--timeout=120s"], timeout=130)
                if record["rollout"].get("returncode") != 0:
                    return False
                record["deployment"] = self._command(["get", "deployment", "compute-worker",
                                                       "-n", namespace, "-o", "json"])
                obj = _object(record["deployment"])
                if obj is not None:
                    spec, status = obj.get("spec"), obj.get("status")
                    record["ok"] = bool(isinstance(spec, dict) and isinstance(status, dict)
                                        and spec.get("replicas") == status.get("readyReplicas")
                                        == status.get("replicas") == 1)
                return record["ok"]
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    record["interrupted"] = True
                raise
            finally:
                try:
                    if self.run_dir is not None:
                        self._save("hpa_reset_check.json", record)
                finally:
                    self._reset_record = None

        def get_state(self):
            return {"method": preset, "min_replicas": 1, "max_replicas": 20,
                    **self._routing(),
                    "cpu_target_pct": 70 if preset == "hpa-cpu-tuned" else 80,
                    "hpa_api_version": "autoscaling/v2", "hpa_name": self.hpa_name,
                    "hpa_uid": self.hpa_uid, "behavior_present": preset == "hpa-cpu-tuned",
                    "downscale_stabilization_s": 60 if preset == "hpa-cpu-tuned" else None,
                    "upscale_stabilization_s": 0 if preset == "hpa-cpu-tuned" else None}

    return NativeHPAMethod()
