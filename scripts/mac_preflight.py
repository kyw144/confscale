"""Offline inventory only: no subprocess, sockets, downloads or cluster calls."""
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys

root = Path(__file__).resolve().parents[1]
report = {
    'kind':'LOCAL_INVENTORY_ONLY_NOT_RUNTIME_VALIDATION',
    'platform':platform.platform(), 'machine':platform.machine(), 'python':platform.python_version(),
    'commands_available':{name:bool(shutil.which(name)) for name in ['docker','kind','kubectl']},
    'optional_modules_available':{name:importlib.util.find_spec(name) is not None for name in ['numpy','torch','yaml','pandas','prometheus_client']},
    'runtime_input_directories':{name:(root/'reference/stage3_scale'/name).is_dir() for name in ['models/gru','models/uq','outputs/training_data']},
    'cluster_contacted':False, 'runtime_validated':False,
    'next':'Read docs/MAC_VERIFICATION.md before enabling historical runtime.'}
print(json.dumps(report,indent=2,sort_keys=True))
