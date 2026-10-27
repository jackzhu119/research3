"""R1 diagnostics without a periodic native stack-unwinding thread.

The observed school exits are SIGSEGV-like (-11) and the failing log ends in
the middle of a periodic faulthandler dump. This correlation is not a proven
native root cause. Fresh owned children remove that optional diagnostic risk
while retaining fatal fault reports and the existing parent's progress beats.
No training, validation, timing budget, data or checkpoint semantics change.
"""
from __future__ import annotations

from contextlib import contextmanager
import faulthandler
import functools
import sys

REPAIR_ID = 'paper3-r1-no-periodic-native-stack-v1'
_STATE = '_paper3_r1_safe_diagnostics'


def _contains(function, target):
    seen = set()
    while function is not None and id(function) not in seen:
        if function is target:
            return True
        seen.add(id(function))
        function = getattr(function, '__wrapped__', None)
    return False


def install(startup_module, *, emit=None, faults=None):
    """Wrap ``action_startup.active`` in this fresh R1 process only.

    The original context manager still performs every check and restores its
    own action-local patches. Only its optional ``traceback_interval`` is set
    to zero, including when the caller explicitly asks for 300 seconds.

    Existing ``R1_KERNEL_RUNNING`` and startup/cache events remain the bounded
    heartbeat mechanism; no new stack-sampling thread or signal is introduced.
    A usable stderr allows standard fatal-fault reports. Failure to enable a
    diagnostic never becomes a new scientific startup gate.
    """
    prior = getattr(startup_module, _STATE, None)
    if prior is not None and _contains(startup_module.active, prior['wrapper']):
        return dict(status='ALREADY_INSTALLED', repair_id=REPAIR_ID,
                    periodic_native_stack_dump=False,
                    fatal_fault_reporting=prior['fatal_fault_reporting'],
                    source_files_changed=False)

    previous = startup_module.active
    faults = faults or faulthandler
    fatal = False
    diagnostic_error = None
    try:
        fatal = bool(faults.is_enabled())
        if not fatal:
            faults.enable(file=sys.stderr, all_threads=True)
            fatal = bool(faults.is_enabled())
    except (OSError, RuntimeError, ValueError, AttributeError) as error:
        diagnostic_error = type(error).__name__

    @contextmanager
    @functools.wraps(previous)
    def active(*args, **kwargs):
        options = dict(kwargs)
        options['traceback_interval'] = 0
        with previous(*args, **options) as value:
            yield value

    active._paper3_r1_diagnostics_repair_id = REPAIR_ID
    startup_module.active = active
    setattr(startup_module, _STATE, dict(original=previous, wrapper=active,
        fatal_fault_reporting=fatal))
    record = dict(status='R1_SAFE_DIAGNOSTICS_INSTALLED', repair_id=REPAIR_ID,
        periodic_native_stack_dump=False, fatal_fault_reporting=fatal,
        existing_parent_heartbeat_retained=True, new_background_threads=0,
        checkpoint_signal_handlers_changed=False, source_files_changed=False,
        scientific_parameters_changed=False, diagnostic_error=diagnostic_error,
        native_crash_root_cause_proven=False)
    if emit is not None:
        try:
            emit('R1_SAFE_DIAGNOSTICS_INSTALLED', **{k: v for k, v in record.items() if k != 'status'})
        except Exception:
            pass
    return record


def restore(startup_module):
    """Restore only our context-manager wrapper, never another later overlay.

    Fatal fault handlers remain enabled for the lifetime of the owned child.
    They are not periodic timers and do not touch its checkpoint signals.
    """
    prior = getattr(startup_module, _STATE, None)
    if prior is None:
        return False
    if startup_module.active is not prior['wrapper']:
        return False
    startup_module.active = prior['original']
    delattr(startup_module, _STATE)
    return True
