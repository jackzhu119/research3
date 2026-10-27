"""Action-local snapshots of fully verified R1 classical source capabilities.

The original verifier always runs on the first call. Only an already validated,
frozen source snapshot is reused inside one action; it is never persisted across
actions or trusted from declared hashes alone. A small set of identity anchors
is checked on hits, and every captured dependency is checked before completion.
No adaptation fit, support/query result, mutable learned model or RNG is cached.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import time

VERSION = 'paper3-r1-action-verified-source-snapshot-v1'
ENABLED_KINDS = frozenset(('calibration_profile', 'development', 'internal',
                           'extension_development', 'confirmation'))


class FrozenDependencyChanged(ValueError):
    pass


def _signature(path):
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise FrozenDependencyChanged('Validated dependency is no longer a regular file')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _plain(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    raise TypeError('Source-loader arguments do not have a stable JSON identity')


def _call_key(function, args, kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return json.dumps(_plain(bound.arguments), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def _clone_context(result):
    """Clone metadata, retaining only the original read-only source mmap views."""
    cache, contract = result
    duplicate = copy.copy(cache)
    for name in ('rows', 'locations', 'artifact_sha256', 'upstream_acceptance_sha256'):
        if hasattr(cache, name):
            setattr(duplicate, name, copy.deepcopy(getattr(cache, name)))
    if hasattr(cache, 'arrays'):
        duplicate.arrays = list(cache.arrays)
    if hasattr(cache, 'labels'):
        duplicate.labels = cache.labels.copy()
        duplicate.labels.flags.writeable = False
    return duplicate, copy.deepcopy(contract)


def _clone_bank(model):
    # Both freshly loaded and privately cached models retain their exact native
    # state identity. A caller cannot mutate the cache through its receipt.
    import numpy as np
    model.assert_unchanged()
    arrays = {name: np.array(value, copy=True, order='K') for name, value in model.arrays.items()}
    for value in arrays.values():
        value.flags.writeable = False
    duplicate = type(model)(model.method, arrays, copy.deepcopy(model.receipt), model.state_sha256)
    duplicate.assert_unchanged()
    return duplicate


class ActionSnapshot:
    def __init__(self, root, action, *, emit=None):
        self.root = Path(root).resolve(strict=True)
        self.action = action
        self.emit = emit
        self.dependencies = {}
        self.anchors = {}
        self.status_identity = None
        self.status_observation = None
        self.contexts = {}
        self.banks = {}
        self.mapping = {}
        self.stats = dict(context_verifications=0, context_hits=0, bank_verifications=0,
                          bank_hits=0, boundary_checks=0, anchor_checks=0)
        self.last_notice = time.monotonic()

    def notice(self, event, **fields):
        record = dict(version=VERSION, action_key=self.action.get('key'),
                      action_kind=self.action.get('kind'),
                      utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **self.stats, **fields)
        try:
            if self.emit is not None:
                self.emit(event, **record)
            else:
                print(json.dumps(dict(event=event, **record), ensure_ascii=False), flush=True)
        except Exception:
            pass

    def path(self, value):
        path = Path(value)
        path = path if path.is_absolute() else self.root/path
        result = path.resolve(strict=True)
        if not result.is_relative_to(self.root):
            raise FrozenDependencyChanged('Source dependency escaped its verified root')
        return result

    def add(self, value, *, anchor=False):
        path = self.path(value); signature = _signature(path)
        if path in self.dependencies and self.dependencies[path] != signature:
            raise FrozenDependencyChanged('Source changed while establishing an action snapshot: '+str(path))
        self.dependencies[path] = signature
        if anchor:
            self.anchors[path] = signature
        return path

    def _check(self, values):
        for path, signature in values.items():
            try:
                matches = _signature(path) == signature and path.resolve(strict=True) == path
            except OSError:
                matches = False
            if not matches:
                raise FrozenDependencyChanged('Verified source dependency changed within this action: '+str(path))

    def checkpoint(self):
        """Full dependency guard before the original final receipt is published."""
        self._check(self.dependencies)
        self._check_source_status()
        self.stats['boundary_checks'] += 1

    def _hit(self):
        self._check(self.anchors)
        self._check_source_status()
        self.stats['anchor_checks'] += 1
        if time.monotonic()-self.last_notice >= 30:
            self.last_notice = time.monotonic()
            self.notice('R1_VERIFIED_SOURCE_SNAPSHOT_PROGRESS', dependencies=len(self.dependencies),
                        anchors=len(self.anchors), source_payloads_revalidated_on_hit=False)

    def _source_status(self):
        path = self.path('status/tasks.json')
        signature = _signature(path)
        if self.status_observation is not None and self.status_observation[:2] == (path, signature):
            return self.status_observation[2]
        try:
            value = json.loads(path.read_text())
            identity = json.dumps({name: value.get(name) for name in ('W1', 'W2')},
                                  sort_keys=True, separators=(',', ':'), allow_nan=False)
            if _signature(path) != signature:
                raise FrozenDependencyChanged('Source W1/W2 status changed while reading it')
            self.status_observation = (path, signature, identity)
            return identity
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise FrozenDependencyChanged('Source W1/W2 task identity became unreadable') from error

    def _check_source_status(self):
        if self.status_identity is not None and self._source_status() != self.status_identity:
            raise FrozenDependencyChanged('Verified source W1/W2 status or cache binding changed within this action')

    def _capture_context(self, result):
        cache, _ = result
        # All of these digests were produced/checked by the original successful
        # verifier. They are dependency names, never a shortcut around it.
        for relative in cache.artifact_sha256:
            if self.path(relative) != self.root/'status/tasks.json':
                self.add(relative)
        current_status = self._source_status()
        if self.status_identity is not None and current_status != self.status_identity:
            raise FrozenDependencyChanged('Source W1/W2 identity changed while establishing a snapshot')
        self.status_identity = current_status
        self.add('contracts/input_contract.json', anchor=True)
        manifest_path = self.add(cache.manifest_path, anchor=True)
        manifest = json.loads(manifest_path.read_text())
        inventory_descriptor = manifest.get('raw_inventory')
        if isinstance(inventory_descriptor, dict) and inventory_descriptor.get('path'):
            inventory = self.add(inventory_descriptor['path'], anchor=True)
            evidence = json.loads(inventory.read_text()).get('provider_terms_evidence')
            if isinstance(evidence, dict) and evidence.get('path'):
                self.add(evidence['path'])

    def _capture_bank(self, args, kwargs, function):
        bound = inspect.signature(function).bind(*args, **kwargs); bound.apply_defaults()
        descriptor = bound.arguments['receipt_path']
        receipt_path = descriptor['path'] if isinstance(descriptor, dict) else descriptor
        receipt_file = self.add(receipt_path, anchor=True)
        receipt = json.loads(receipt_file.read_text())
        job = receipt.get('job', {}); job_id = job.get('job_id')
        if not isinstance(job_id, str) or not job_id:
            # This branch is only for injected software verifiers. A production
            # successful original loader necessarily supplied the registered job.
            return
        output = receipt_file.parent.parents[len(Path(job_id).parts)-1]
        identity_path = self.add(output/'identity.json', anchor=True)
        identity = json.loads(identity_path.read_text())
        budget = identity.get('budget')
        if isinstance(budget, dict) and budget.get('path'):
            self.add(budget['path'], anchor=True)
        plan_path = self.add(output/'plan.json', anchor=True)
        plan = json.loads(plan_path.read_text())
        for group in plan.get('code_policy', {}).values():
            if isinstance(group, dict):
                for relative in group:
                    self.add(relative)
        jobs = [item for item in plan['jobs'] if item.get('group') == 'core' or item == job]
        for item in jobs:
            folder = output/item['job_id']
            state_path = self.add(folder/'status.json')
            state = json.loads(state_path.read_text())
            manifest_path = self.add(state['artifact_manifest']['path'])
            members = json.loads(manifest_path.read_text())['files']
            for member in members:
                self.add(member['path'])

    def _replace(self, original, wrapper):
        self.mapping[original] = wrapper
        for name, module in list(sys.modules.items()):
            if not (name.startswith('p3work.') or name.startswith('tools.')) or module is None:
                continue
            for key, value in list(vars(module).items()):
                if value is original:
                    setattr(module, key, wrapper)

    def install(self, training, development=None):
        original_context = training.verified_context
        original_bank = training.load_verified_source_bank

        @functools.wraps(original_context)
        def context(*args, **kwargs):
            identity = _call_key(original_context, args, kwargs)
            if identity in self.contexts:
                self._hit(); self.stats['context_hits'] += 1
                return _clone_context(self.contexts[identity])
            result = original_context(*args, **kwargs)
            self._capture_context(result)
            self.contexts[identity] = _clone_context(result)
            self.stats['context_verifications'] += 1
            self.notice('R1_SOURCE_CONTEXT_VERIFIED_ONCE', dependencies=len(self.dependencies))
            return result

        @functools.wraps(original_bank)
        def bank(*args, **kwargs):
            identity = _call_key(original_bank, args, kwargs)
            if identity in self.banks:
                self._hit(); self.stats['bank_hits'] += 1
                return _clone_bank(self.banks[identity])
            result = original_bank(*args, **kwargs)
            self._capture_bank(args, kwargs, original_bank)
            self.banks[identity] = _clone_bank(result)
            self.stats['bank_verifications'] += 1
            self.notice('R1_SOURCE_BANK_VERIFIED_ONCE', method=getattr(result, 'method', None))
            return result

        self._replace(original_context, context)
        self._replace(original_bank, bank)
        # Injected modules need not be registered under their production name.
        training.verified_context = context
        training.load_verified_source_bank = bank
        self.training = training
        self.original_context = original_context
        self.original_bank = original_bank
        if development is not None and hasattr(development, '_publish_completion'):
            original_publish = development._publish_completion
            @functools.wraps(original_publish)
            def publish(*args, **kwargs):
                self.checkpoint()
                return original_publish(*args, **kwargs)
            self._replace(original_publish, publish)
            development._publish_completion = publish
            self.development = development
            self.original_publish = original_publish
        return self

    def close(self, *, successful=True):
        try:
            if successful:
                self.checkpoint()
        finally:
            reverse = {id(wrapper): original for original, wrapper in self.mapping.items()}
            for name, module in list(sys.modules.items()):
                if not (name.startswith('p3work.') or name.startswith('tools.')) or module is None:
                    continue
                for key, value in list(vars(module).items()):
                    original = reverse.get(id(value))
                    if original is not None:
                        setattr(module, key, original)
            self.training.verified_context = self.original_context
            self.training.load_verified_source_bank = self.original_bank
            if hasattr(self, 'development'):
                self.development._publish_completion = self.original_publish
            self.contexts.clear(); self.banks.clear()
            self.notice('R1_VERIFIED_SOURCE_SNAPSHOT_FINISHED', dependencies=len(self.dependencies),
                        anchors=len(self.anchors), source_snapshot_persisted=False)


@contextmanager
def active(root, action, *, emit=None, training=None, development=None):
    """Install only for one calibration/development action; restore on all exits."""
    if action.get('kind') not in ENABLED_KINDS:
        yield None
        return
    training = training or importlib.import_module('p3work.classical_training')
    development = development or importlib.import_module('p3work.development_evaluation')
    scope = ActionSnapshot(root, action, emit=emit).install(training, development)
    successful = False
    try:
        yield scope
        successful = True
    finally:
        scope.close(successful=successful)
