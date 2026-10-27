"""One-time, metadata-only M12 preparation and fixed scientific DAG.

No Slurm, GPU acquisition, EEG fitting, network request or old-round mutation
occurs here. Full numerical work starts only in explicitly submitted workers.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

from .common import atomic_json, digest, load_plan, lock, read_json, sha_file
from .native_data import freeze_inputs, checked, read as native_read

SCHEMA = 'paper3-m12-plan-v1'
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (4, 8, 12, 16, 20)


def _task(key, kind, dependencies=(), **fields):
    return dict(key=key, kind=kind, dependencies=list(dependencies), **fields)


def _external_people(root, inputs):
    found, unavailable = {}, {}
    for cohort in ('pan2023', 'weibo2014'):
        descriptor = inputs['registry'].get('external_manifests', {}).get(cohort)
        if not descriptor:
            unavailable[cohort] = 'Existing prepared external cache not declared; core study proceeds'
            continue
        try:
            manifest = native_read(checked(root, descriptor))
            people = sorted({str(item['participant_id']) for item in manifest.get('sessions', [])})
            if not people:
                raise ValueError('External metadata declares no prepared participants')
            found[cohort] = people
        except (OSError, ValueError, KeyError) as error:
            unavailable[cohort] = dict(error_type=type(error).__name__,
                                       reason='Optional external metadata unavailable; no automatic download')
    return found, unavailable


def code_inventory(package):
    package = Path(package).resolve()
    names = [p for p in (package/'paper3m12').rglob('*.py') if '__pycache__' not in p.parts]
    names += [package/name for name in ('EXPERIMENT_CONFIG.json', 'PRIMARY_COMPARISONS.csv')]
    return {p.relative_to(package).as_posix(): sha_file(p) for p in sorted(names) if p.is_file()}


def build_plan(root, output, package, config, inputs, *, external_people=None,
               unavailable_external=None, code_manifest=None, python=None):
    root, output, package = (Path(value).resolve() for value in (root, output, package))
    seeds = tuple(int(seed) for seed in config['model']['source_seeds'])
    if seeds != SEEDS or config['data']['acquired_budgets'] != list(BUDGETS):
        raise ValueError('The fixed three source seeds and five acquired budgets must be retained')
    people = inputs['people']
    if len(people['pilot']) != 8 or len(people['reference']) != 32:
        raise ValueError('The registered32-reference/8-pilot participant split is required')
    if set(people['pilot']) & set(people['reference']):
        raise ValueError('Reference and pilot participants overlap')
    tasks = []
    calibration = []
    for seed in seeds:
        key = 'calibrate-seed-' + str(seed)
        calibration.append(key)
        tasks.append(_task(key, 'calibration', seed=seed, role='reference', cohort='meta_train',
                           participant_ids=list(people['pilot']), budgets=list(BUDGETS),
                           target_query_access=False))
    tau_key = 'freeze-tau'
    tasks.append(_task(tau_key, 'freeze_tau', calibration,
                       calibration_tasks=calibration, query_labels_used=False))
    pilot = []
    for person in people['pilot']:
        for seed in seeds:
            key = 'paired-pilot-' + str(person) + '-s' + str(seed)
            pilot.append(key)
            tasks.append(_task(key, 'paired_block', [tau_key], role='pilot', cohort='pilot',
                               participant=str(person), seed=seed, budgets=list(BUDGETS)))
    decision = 'pilot-decision'
    tasks.append(_task(decision, 'pilot_decision', pilot, pilot_tasks=pilot,
                       requires_full_three_seed_coverage=True))
    # These are predeclared competing-explanation diagnostics. They may run
    # after a spatial-route STOP, without changing any primary comparison.
    secondary = []
    for kind, prefix in (('head_block', 'head'), ('shift_block', 'shift')):
        for person in people['pilot']:
            for seed in seeds:
                key = prefix + '-pilot-' + str(person) + '-s' + str(seed)
                secondary.append(key)
                tasks.append(_task(key, kind, [decision, tau_key], role='pilot', cohort='pilot',
                                   participant=str(person), seed=seed, budgets=list(BUDGETS),
                                   inference_scope='secondary_development_mechanism_only'))
    pilot_gate = dict(task=decision, field='decision', equals='GO')
    validation = []
    for person in people['meta_validation']:
        for seed in seeds:
            key = 'paired-meta_validation-' + str(person) + '-s' + str(seed)
            validation.append(key)
            tasks.append(_task(key, 'paired_block', [decision, tau_key], gate=pilot_gate,
                               role='meta_validation', cohort='meta_validation',
                               participant=str(person), seed=seed, budgets=list(BUDGETS)))
    freeze = 'development-freeze'
    tasks.append(_task(freeze, 'development_freeze', validation + [decision, tau_key],
                       gate=pilot_gate, validation_tasks=validation,
                       no_hyperparameter_or_method_search=True))
    validation_gate = dict(task=freeze, field='decision', equals='GO')
    exploratory = []
    cohort_people = {'holdout': list(people['internal_holdout'])}
    cohort_people.update(external_people or {})
    for role, participants in cohort_people.items():
        for person in participants:
            for seed in seeds:
                key = 'paired-' + role + '-' + str(person) + '-s' + str(seed)
                exploratory.append(key)
                tasks.append(_task(key, 'paired_block', [freeze, tau_key], gate=validation_gate,
                                   role=role, cohort=role, participant=str(person), seed=seed,
                                   budgets=list(BUDGETS),
                                   inference_scope='exploratory_previously_used_cohort'))
    tasks.append(_task('final-report', 'final_report', [item['key'] for item in tasks],
                       allow_skipped_dependencies=True, allow_blocked_dependencies=True,
                       independent_confirmation_available=False))
    resolved_config = copy.deepcopy(config)
    resolved_config.update(status='REGISTERED_RULES_INPUTS_FROZEN_TAU_PENDING', execution_enabled=True,
                           historical_design_status=config.get('status'))
    resolved_config['model'].update(
        source_checkpoint_sha256_by_seed={seed: item['checkpoint']['sha256']
                                         for seed, item in inputs['source_by_seed'].items()},
        dictionary_sha256={seed: item['checkpoint']['sha256']
                           for seed, item in inputs['dictionary_by_seed'].items()},
        reference_event_uids_sha256='DEFERRED_TO_SUPPORT_ONLY_CALIBRATION_RECEIPTS')
    resolved_config['resolved_input_identity'] = inputs['input_identity']
    resolved_config['matching']['tau_freeze_artifact'] = 'frozen_tau.json'
    value = dict(schema_version=SCHEMA, round=12, experiment='controlled_spatial_failure_mechanism',
                 root=str(root), output=str(output), package=str(package),
                 worker_entry=str(package/'paper3m12/entry.py'), python=python or sys.executable,
                 stage='controlled_mechanism', config=resolved_config,
                 native_inputs=copy.deepcopy(inputs), input_identity=inputs['input_identity'],
                 resources=dict(jobs=4, gpus_per_job=1, cpus_per_job=16, memory_per_job='64G',
                                partition='hpc_gpu'), tasks=tasks,
                 expected_pilot_tasks=pilot, expected_calibration_tasks=calibration,
                 expected_validation_tasks=validation, expected_secondary_tasks=secondary,
                 expected_exploratory_tasks=exploratory,
                 participants=dict(pilot=list(people['pilot']),
                                   meta_validation=list(people['meta_validation']), **cohort_people),
                 phase_task_counts=dict(Counter(t['kind'] for t in tasks)),
                 max_task_failures=3, max_worker_attempts=12,
                 start_sidecars=True, automatic_child_refresh=False,
                 original_rounds_modified=False, unavailable_external=unavailable_external or {},
                 independent_confirmation=dict(status='NOT_AVAILABLE',
                     reason='No new independent cohort was supplied; old cohorts remain exploratory',
                     planned_evaluable_participants=90, planned_initial_participants=110),
                 scientific_scope='Core A/B, predeclared secondary C, development and old-cohort exploration; no independent confirmation claim',
                 code_manifest=code_manifest if code_manifest is not None else code_inventory(package))
    value['recipe_sha256'] = digest(value)
    return value


def prepare(root, output, package):
    root, package = Path(root).expanduser().resolve(), Path(package).expanduser().resolve()
    output = Path(output).expanduser()
    output = (root/output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(root/'runs') or output.name in {'training_campaign_parallel_v1', 'r11_campaign'}:
        raise ValueError('M12 output must be its own directory under root/runs')
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output/'plan.json'
    with lock(output/'prepare.lock'):
        if plan_path.is_file():
            _, previous = load_plan(plan_path)
            if (Path(previous['root']) != root or Path(previous['output']) != output
                    or previous.get('schema_version') != SCHEMA):
                raise ValueError('Existing M12 output belongs to a different frozen plan')
            return dict(status='EXISTING_M12_PLAN_REUSED', plan=str(plan_path),
                        recipe_sha256=previous['recipe_sha256'], total_tasks=len(previous['tasks']),
                        existing_completed_results_retained=True, jobs_submitted=0, GPU_used=False)
        config = read_json(package/'EXPERIMENT_CONFIG.json')
        if not isinstance(config, dict):
            raise ValueError('Packaged controlled study configuration is required')
        inputs = freeze_inputs(root, output, config)
        external_people, unavailable = _external_people(root, inputs)
        plan = build_plan(root, output, package, config, inputs,
                          external_people=external_people, unavailable_external=unavailable)
        atomic_json(plan_path, plan)
        load_plan(plan_path)  # One metadata-only DAG check before any Slurm RPC.
        receipt = dict(status='M12_PLAN_READY_NO_GPU_ALLOCATED', plan=str(plan_path),
                       recipe_sha256=plan['recipe_sha256'], input_identity=inputs['input_identity'],
                       total_tasks=len(plan['tasks']), phase_task_counts=plan['phase_task_counts'],
                       pilot_participants=8, independent_confirmation='NOT_AVAILABLE',
                       resources=plan['resources'], jobs_submitted=0, GPU_used=False,
                       old_round_files_modified=0, automatic_child_refresh=False)
        atomic_json(output/'PREPARED.json', receipt)
        return receipt
