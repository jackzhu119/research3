"""Visible, action-local startup checks for the original parallel campaign.

The original validators execute before any result is reused. Successful returns
are memoized only for this one action; no scientific source or checkpoint is
modified, and all original functions are restored on leaving the context.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import faulthandler
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Non-string validator key")
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    raise TypeError("Validator argument cannot be represented exactly")


def _key(function, args, kwargs):
    binding = inspect.signature(function).bind(*args, **kwargs)
    binding.apply_defaults()
    return _encoded(_plain(dict(binding.arguments)))


def _progress_path(root, action, action_file=None):
    digest = hashlib.sha256(str(action.get("key", "")).encode()).hexdigest()
    base = Path(root) / "runs/training_campaign_parallel_v1"
    if action_file is None:
        try:
            index = sys.argv.index("--action-file")
            action_file = sys.argv[index + 1]
        except (ValueError, IndexError):
            pass
    if action_file:
        candidate = Path(action_file).resolve()
        try:
            relative = candidate.relative_to((base / "actions" / digest / "attempts").resolve())
            if len(relative.parts) == 2 and relative.name == "action.json":
                return candidate.with_name("startup-progress.json")
        except ValueError:
            pass
    return base / "action-startup" / (digest + ".json")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".startup-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _gpu_profile(torch_module):
    if not torch_module.cuda.is_available():
        raise ValueError("The requested campaign allocation has no available CUDA GPU")
    torch_module.cuda.set_device(0)
    properties = torch_module.cuda.get_device_properties(0)
    return {
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unavailable")),
        "device_total_memory_bytes": properties.total_memory,
        "device_compute_capability": [properties.major, properties.minor],
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "cuda": str(torch_module.version.cuda),
    }


@contextmanager
def active(root, action, *, modules=None, torch_module=None, emit=None,
           action_file=None, traceback_interval=300):
    """Probe the allocated GPU early, then reuse same-action verified inputs.

    Injection arguments support CPU-only software tests. Production callers need
    only ``with active(root, action):`` after installing their native overlays.
    """
    root = Path(root).resolve()
    progress = _progress_path(root, action, action_file)
    started = time.monotonic()
    counters = {"checks_completed": 0, "cache_hits": 0}
    saved = []
    trace_enabled = False

    def notice(event, **fields):
        value = {"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "action_key": action.get("key"), "action_kind": action.get("kind"),
                 "elapsed_seconds": round(time.monotonic() - started, 3),
                 **counters, **fields}
        try:
            if emit is None:
                print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
            else:
                emit(event, **{key: child for key, child in value.items() if key != "event"})
        except Exception:
            pass
        try:
            _write(progress, value)
        except OSError:
            # Observability writes cannot invalidate a successful native check.
            pass

    try:
        kind = action.get("kind", "")
        needs_gpu = kind in {"meta_engineering", "meta_screening", "meta_confirmation",
                             "calibration_profile", "development", "internal", "extension_development"}
        if needs_gpu:
            notice("R1_STARTUP_GPU_PROBE_STARTED", phase="gpu_probe")
            if torch_module is None:
                torch_module = importlib.import_module("torch")
            profile = _gpu_profile(torch_module)
            from audited_runtime import hardware_type
            hardware = hardware_type(profile)
            notice("R1_STARTUP_GPU_PROBE_FINISHED", phase="gpu_probe", hardware_type=hardware,
                   observed_gpu=profile)
        if kind in {"meta_engineering", "meta_screening", "meta_confirmation"}:
            if modules is None:
                modules = (importlib.import_module("p3work.meta_training"),
                           importlib.import_module("p3work.meta_engineering"))
            targets = ((modules[0], "verify_meta_inputs"), (modules[1], "verify_meta_engineering"))
            for module, name in targets:
                original = getattr(module, name)
                cache = {}

                def build(function, label, successes):
                    @functools.wraps(function)
                    def verified(*args, **kwargs):
                        try:
                            identity = _key(function, args, kwargs)
                        except (TypeError, ValueError):
                            identity = None
                        if identity is not None and identity in successes:
                            counters["cache_hits"] += 1
                            notice("R1_INPUT_VERIFY_CACHE_HIT", phase=label)
                            return successes[identity]
                        notice("R1_INPUT_VERIFY_STARTED", phase=label)
                        check_started = time.monotonic()
                        try:
                            result = function(*args, **kwargs)
                        except BaseException as error:
                            notice("R1_INPUT_VERIFY_FAILED", phase=label, error_type=type(error).__name__)
                            raise
                        if identity is not None:
                            successes[identity] = result
                        counters["checks_completed"] += 1
                        notice("R1_INPUT_VERIFY_FINISHED", phase=label,
                               check_elapsed_seconds=round(time.monotonic() - check_started, 3))
                        return result
                    return verified

                setattr(module, name, build(original, name, cache))
                saved.append((module, name, original))
        if traceback_interval:
            try:
                faulthandler.dump_traceback_later(traceback_interval, repeat=True)
                trace_enabled = True
            except (RuntimeError, OSError, ValueError):
                pass
        notice("R1_STARTUP_SCOPE_ACTIVE", phase="native_action")
        yield
    except BaseException as error:
        notice("R1_STARTUP_SCOPE_RETURNED", phase="native_action", status="ERROR", error_type=type(error).__name__)
        raise
    else:
        notice("R1_STARTUP_SCOPE_RETURNED", phase="native_action", status="RETURNED")
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)
        if trace_enabled:
            faulthandler.cancel_dump_traceback_later()
