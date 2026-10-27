"""Process-local source verification reuse for future R1 selection actions.

Only ``development_selection`` is covered. Native receipt verification, metrics,
selection rules and final immutable writer are unchanged. The existing verified
source snapshot checks every captured dependency immediately before the native
selection receipt writer. Nothing is installed into an already running child.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import importlib
import json
from pathlib import Path
import time


VERSION = 'paper3-r1-selection-source-snapshot-v1'
SUPPORTED_KINDS = frozenset(('development_selection',))
_STATE = '_paper3_r1_selection_source_snapshot_v1'
SELECTION_STATUS = 'DEVELOPMENT_CONFIGURATION_FROZEN'


class SelectionPublicationGuardError(ValueError):
    pass


def _notice(emit, event, action, **fields):
    value = dict(version=VERSION, action_key=action.get('key'),
                 action_kind=action.get('kind'),
                 utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    try:
        if emit is None:
            print(json.dumps(dict(event=event, **value), ensure_ascii=False), flush=True)
        else:
            emit(event, **value)
    except Exception:
        # Observability has no authority over the scientific action.
        pass


def _expected_output(root, action):
    relative = action.get('expected_receipt')
    if not isinstance(relative, str) or not relative:
        raise SelectionPublicationGuardError('Selection expected receipt is required')
    target = (root / relative).resolve(strict=False)
    if not target.is_relative_to(root / 'runs'):
        raise SelectionPublicationGuardError('Selection receipt escaped owned runs directory')
    if not isinstance(action.get('method'), str) or not action['method']:
        raise SelectionPublicationGuardError('Selection method identity is required')
    return target


@contextmanager
def active(root, action, old_verified_cache, *, training=None, development=None,
           emit=None):
    """Reuse proven source capabilities inside exactly one future action.

    This deliberately constructs ``ActionSnapshot`` without changing the old
    module's ENABLED_KINDS. The old startup context therefore remains unchanged
    for every other action, including meta-selection and scientific fitting.
    """
    if action.get('kind') not in SUPPORTED_KINDS:
        yield None
        return
    root = Path(root).resolve(strict=True)
    target = _expected_output(root, action)
    training = training or importlib.import_module('p3work.classical_training')
    development = development or importlib.import_module('p3work.development_evaluation')
    native_put = development._put
    scope = old_verified_cache.ActionSnapshot(root, action, emit=emit).install(
        training, development)
    succeeded = False
    guarded_publications = 0

    @functools.wraps(native_put)
    def guarded_put(path, value):
        nonlocal guarded_publications
        actual = Path(path).resolve(strict=False)
        selection_value = isinstance(value, dict) and value.get('status') == SELECTION_STATUS
        if actual == target or selection_value:
            if (actual != target or not selection_value
                    or value.get('method') != action['method']):
                raise SelectionPublicationGuardError('Selection output identity differs from current action')
            # All ordinary native receipt checks have run. Reusing a source
            # model/context is allowed only while every captured byte source
            # retains the exact inode/size/mtime/ctime identity verified earlier.
            scope.checkpoint()
            _notice(emit, 'R1_SELECTION_SOURCE_DEPENDENCIES_CHECKED_BEFORE_PUBLICATION',
                    action, dependencies=len(scope.dependencies),
                    anchors=len(scope.anchors), native_writer_retained=True)
            result = native_put(path, value)
            guarded_publications += 1
            return result
        return native_put(path, value)

    development._put = guarded_put
    _notice(emit, 'R1_SELECTION_VERIFICATION_CACHE_ACTIVE', action,
            process_local=True, original_receipt_verifiers_retained=True,
            source_cache_persists_across_actions=False,
            native_selection_writer_guarded=True,
            original_scientific_files_changed=False)
    try:
        yield scope
        succeeded = True
    finally:
        # Restore only our still-current writer. Never overwrite a later owner.
        if development._put is guarded_put:
            development._put = native_put
        try:
            scope.close(successful=succeeded)
        finally:
            _notice(emit, 'R1_SELECTION_VERIFICATION_CACHE_FINISHED', action,
                    successful=succeeded, guarded_publications=guarded_publications,
                    source_cache_persists_across_actions=False)


def _contains(function, target):
    seen = set()
    while function is not None and id(function) not in seen:
        if function is target:
            return True
        seen.add(id(function))
        function = getattr(function, '__wrapped__', None)
    return False


def install(startup_module, old_verified_cache, root, *, training=None,
            development=None, emit=None):
    """Wrap a fresh child's startup context; no files/processes/jobs touched."""
    expected_root = Path(root).resolve(strict=True)
    prior = getattr(startup_module, _STATE, None)
    if prior is not None and _contains(startup_module.active, prior['wrapper']):
        if prior['expected_root'] != expected_root:
            raise SelectionPublicationGuardError('Existing selection cache has a different root')
        return dict(status='ALREADY_INSTALLED', repair_id=VERSION,
                    supported_kinds=sorted(SUPPORTED_KINDS),
                    scientific_files_changed=False, signals_sent=0)
    if 'development_selection' in getattr(old_verified_cache, 'ENABLED_KINDS', ()):
        raise SelectionPublicationGuardError('Refusing overlapping old selection cache scope')
    previous = startup_module.active

    @contextmanager
    @functools.wraps(previous)
    def wrapped(root, action, *args, **kwargs):
        if action.get('kind') in SUPPORTED_KINDS and Path(root).resolve(strict=True) != expected_root:
            raise SelectionPublicationGuardError('Startup root differs from installed selection scope')
        selected_emit = emit if emit is not None else kwargs.get('emit')
        with active(root, action, old_verified_cache, training=training,
                    development=development, emit=selected_emit):
            with previous(root, action, *args, **kwargs) as result:
                yield result

    startup_module.active = wrapped
    setattr(startup_module, _STATE, dict(wrapper=wrapped, original=previous,
                                      expected_root=expected_root))
    return dict(status='R1_SELECTION_SOURCE_CACHE_INSTALLED', repair_id=VERSION,
                supported_kinds=sorted(SUPPORTED_KINDS),
                original_scientific_files_changed=False, signals_sent=0,
                existing_children_modified=False)


def restore(startup_module):
    prior = getattr(startup_module, _STATE, None)
    if prior is None or startup_module.active is not prior['wrapper']:
        return False
    startup_module.active = prior['original']
    delattr(startup_module, _STATE)
    return True
