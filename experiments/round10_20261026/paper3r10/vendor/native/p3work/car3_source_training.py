"""Independent CAR3 source f0; default public CLI is read-only and GPU gated.

No old module globals, protocol source allocation, preparation contract, external
payload permission, 11-channel checkpoint, Ma pool, meta learner or method grid
is changed. This module reuses only generic integrity/math/clock helpers.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import copy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
import numpy as np
import torch
from torch import nn
from .contracts import SOURCE_IDS, digest, file_digest
from .spatial_geometry import joint_trial_rms
from . import car3_cache as preparation
from .source_training import (SourceTrainingError, atomic_json, immutable_json,
    atomic_checkpoint, verified_checkpoint, checkpoint_sidecar, capture_rng, restore_rng,
    hierarchical_weights, participant_validation_loss, final_epoch_from_lodo,
    operational_budget, _source_wall_clock, _observe_source_wall)

ARCHITECTURE = "compact_eegnet_v1_car3_independent"
CONTRACT = "contracts/car3_source_training_v1.json"
CACHE = "cache/source_c3czc4_v1/source_cache3.json"
CACHE_SHA = "001d1e9c3810140c8b71b367c223bf61204c83e569971eb88de1cdffac7d9492"
PREPARATION_SHA = "b9c1b07f24915cb9359c5443a6d483859d3764a5e723e7010779be614c278dae"
PARENT_SHA = "bfc204577dfe51c1fc798c1241391ab63509fc2ad9ebfb267ba09254e01ae644"
SEED_PROFILES = {"development": (20261018,20261019,20261020),
                 "confirmation": (20261018,20261019,20261020,20261021,20261022)}
SOURCE_CONFIG = {'epochs': 80, 'steps_per_epoch': 64, 'batch_size': 64, 'optimizer': 'Adam', 'learning_rate': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0, 'amsgrad': False, 'foreach': False, 'dropout': 0.25, 'patience': 15, 'min_delta': 1e-06, 'threads': 4, 'checkpoint_every_updates': 16, 'retain_epoch_checkpoints': True, 'validation': 'dataset_then_participant_equal_mean_clipped_cross_entropy', 'final_epochs': 'max(1,int(median(four_completed_same_seed_LODO_selected_epochs)))', 'sampling': 'dataset_then_subject_then_observed_class_then_trial_with_replacement', 'precision': 'float64_joint_RMS_then_float32_no_AMP_no_TF32'}
EXPECTED_CONTRACT = {'schema_version': 'paper3-car3-source-f0-execution-contract-v1', 'status': 'SEPARATE_SOURCE_F0_EXECUTION_POLICY_ONLY_NOT_META_OR_EXTERNAL_READY', 'channels': ['C3', 'Cz', 'C4'], 'reference': 'CAR_exact_C3_Cz_C4', 'native_reference_requirement': 'monopolar_common_reference_only', 'bipolar_derivations_accepted': False, 'BCIIV2b_accepted': False, 'C': 3, 'centered_dimension_d': 2, 'Rref_shape': [2, 2], 'future_spatial_rank_choices': [1, 2], 'future_LoRA_rank_choices': [1, 2], 'future_CSP_pairs': 1, 'original_r4_r8_accepted': False, 'sfreq_hz': 128, 'context_s': [0, 1.5], 'crop_s': [0.25, 1.25], 'n_times': 128, 'input_stage': 'pre_rms_car3_temporal_demeaned_from_verified_float32_car11', 'cache_manifest': {'path': 'cache/source_c3czc4_v1/source_cache3.json', 'sha256': '001d1e9c3810140c8b71b367c223bf61204c83e569971eb88de1cdffac7d9492'}, 'parent_cache_sha256': 'bfc204577dfe51c1fc798c1241391ab63509fc2ad9ebfb267ba09254e01ae644', 'preparation_contract_semantic_sha256': 'b9c1b07f24915cb9359c5443a6d483859d3764a5e723e7010779be614c278dae', 'source_datasets': ['BNCI2014_001', 'Cho2017', 'Lee2019_MI', 'PhysionetMI'], 'source_provider_ids': 224, 'source_trial_uids': 28830, 'same_source_people_and_UIDs': True, 'new_independent_people': 0, 'Ma2022_in_source': False, 'primary11_checkpoint_accepted': False, 'architecture': 'compact_eegnet_v1_car3_independent', 'normalization': 'float64_joint_trial_RMS_then_float32_no_floor', 'source_selection': 'four_source_LODO_dataset_then_participant_equal_mean_clipped_cross_entropy', 'epochs': 80, 'steps_per_epoch': 64, 'batch_size': 64, 'final_epochs': 'max(1,int(median(four_completed_same_seed_LODO_selected_epochs)))', 'seed_profiles': {'development': [20261018, 20261019, 20261020], 'confirmation': [20261018, 20261019, 20261020, 20261021, 20261022]}, 'execution_requires': ['--execute', '--user-started-gpu', 'own_CAR3_GPU_exact_resume_engineering_pilot_receipt'], 'engineering_pilot': {'scope': 'CAR3_SOURCE_ENGINEERING_ONLY', 'continuous_updates': 8, 'split_updates': [4, 4], 'expected_total_optimizer_calls': 16, 'max_observed_plus_uncertain_optimizer_calls': 32, 'max_invocations': 8, 'default_wall_seconds': 1800, 'independent_from_main11_200update_pilot': True, 'fixed_leg_interruption_policy': 'Each 0to8,0to4,4to8 leg must complete in one kernel invocation with immutable start/end/call-count/checkpoint-SHA evidence. An intermediate committed cursor cannot be resealed as the fixed uninterrupted leg; preserve evidence and require a separately explicit version.'}, 'f0_receipt_status': 'CAR3_SOURCE_F0_COMPLETE', 'source_Rref_training_prerequisite_only': True, 'Rref_export_implemented': False, 'independent_3G_meta_training_implemented': False, 'all14_methods_implemented': False, 'W7_external_two_seals_required': True, 'external_payload_access_allowed': False, 'RunPod_or_server_boot_allowed': False, 'initial_actual_fits': 0, 'initial_GPU_resume_pilot_verified': False, 'numerical_limitation': 'Derived from accepted float32 CAR11 cache; native float64 pre-cast preprocessing and independent native CAR3 QC are not reconstructed.', 'old_preparation_contract_training_ready': False, 'old_preparation_contract_modified': False, 'completion_resource_policy': 'No final completion after observed wall overrun. Real f0 and CAR3 engineering receipts bind immutable closed-wall and optimizer-ledger snapshots; prior seed snapshots survive explicit confirmation extension.'}
CODE_FILES = ("p3work/car3_source_training.py", "tools/train_car3_source.py",
    "p3work/source_training.py", "p3work/spatial_geometry.py", "p3work/car3_cache.py",
    "p3work/contracts.py", "p3work/recovery_budget.py", "tools/run_controlled_pilot.py")
POLICY_FILES = ("protocol/EXPERIMENT_PROTOCOL.md", "protocol/WORK_TASKS.md",
    "protocol/legacy_baseline_v1/configs/default.json", "protocol/legacy_baseline_v1/p3/neural.py",
    "contracts/sensitivity_car3_v1.json", CONTRACT)


class CompactEEGNet3(nn.Module):
    """Same compact f0 layers; only spatial height is three measured CAR3 leads."""
    def __init__(self, channels=3, times=128, dropout=.25):
        super().__init__()
        if channels != 3 or times != 128:
            raise SourceTrainingError("Independent_CAR3_3_by_128_input_required")
        self.features=nn.Sequential(
            nn.Conv2d(1,8,(1,63),padding=(0,31),bias=False),nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(3,1),groups=8,bias=False),nn.BatchNorm2d(16),
            nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(dropout),
            nn.Conv2d(16,16,(1,15),padding=(0,7),groups=16,bias=False),
            nn.Conv2d(16,16,(1,1),bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(dropout))
        self.classifier=nn.Linear(16*(times//32),2)
    def forward(self,x):
        if x.ndim != 3 or x.shape[1:] != (3,128):
            raise SourceTrainingError("CAR3_tensor_shape_required")
        return self.classifier(self.features(x[:,None]).flatten(1))


def local(root, value, *, exists=True):
    return preparation.local(Path(root),value,exists=exists)


def artifact(root,path):
    return preparation.descriptor(Path(root),Path(path))


def verify_artifact(root,item):
    return preparation.checked(Path(root),item,{})


def _code_policy(root):
    return {"policy":{p:file_digest(root/p) for p in POLICY_FILES},
            "code":{p:file_digest(root/p) for p in CODE_FILES}}


def _uids(cache,indices):
    return [cache.rows[int(i)]["event_uid"] for i in indices]


def immutable_checkpoint(path,value):
    if Path(path).exists() or checkpoint_sidecar(path).exists():
        from tools.run_controlled_pilot import _same_tree
        if not _same_tree(verified_checkpoint(path),value):
            raise SourceTrainingError("Immutable_CAR3_epoch_checkpoint_differs")
    else:
        atomic_checkpoint(path,value)


def immutable_npy(path,value,**kwargs):
    path=Path(path); payload=io.BytesIO();np.save(payload,value,allow_pickle=False)
    raw=payload.getvalue();path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=raw:raise SourceTrainingError("Immutable_CAR3_array_differs")
    finally:Path(tmp).unlink(missing_ok=True)


@dataclass
class VerifiedCar3Cache:
    root: Path
    rows: list
    labels: np.ndarray
    locations: list
    artifact_sha256: dict
    input_contract_sha256: str
    manifest_path: str = CACHE
    manifest_sha256: str = CACHE_SHA
    n_times: int = 128
    def batch(self,indices):
        indices=np.asarray(indices)
        if indices.ndim!=1 or indices.dtype.kind not in "iu" or np.any(indices<0) or np.any(indices>=len(self.rows)):
            raise SourceTrainingError("CAR3_batch_indices_invalid")
        return np.stack([self.locations[int(i)][0][self.locations[int(i)][1]] for i in indices])


def verify_inputs(root):
    """Full source parent+CAR3 SHA/schema/UID/projection validation; no writes/GPU."""
    root=Path(root).resolve(strict=True)
    contract_path=local(root,CONTRACT);contract=json.loads(contract_path.read_text())
    if contract != EXPECTED_CONTRACT:
        raise SourceTrainingError("Exact_separate_CAR3_source_f0_execution_contract_required")
    parent=preparation.validate_inputs(root)
    files=dict(parent["artifacts"])
    path=preparation.checked(root,{"path":CACHE,"sha256":CACHE_SHA},files)
    manifest=json.loads(path.read_text())
    checks={"schema_version":"paper3-source-car3-cache-preparation-v1",
        "status":"SOURCE_CAR3_SENSITIVITY_PREPARED_NOT_TRAINING_READY",
        "role":"source_sensitivity_preparation_only","training_ready":False,"actual_fits":0,
        "channels":["C3","Cz","C4"],"stage":preparation.STAGE3,
        "normalization_applied":False,"synthetic":False,"old_npz_reused":False,
        "source_provider_participant_ids":224,"source_valid_trials":28830,
        "sensitivity_contract_semantic_sha256":PREPARATION_SHA,"new_independent_participants":0,
        "no_source_UID_or_label_changed":True,"primary_11channel_compatible":False}
    if any(manifest.get(k)!=v for k,v in checks.items()) or manifest.get("baseline_manifest",{}).get("sha256")!=PARENT_SHA:
        raise SourceTrainingError("Exact_real_preparation_only_CAR3_cache_required")
    if len(manifest.get("shards",[]))!=224:
        raise SourceTrainingError("Exact_224_CAR3_shards_required")
    parents={(x["shard"]["dataset"],str(x["shard"]["subject_id"])):x for x in parent["records"]}
    rows=[];labels=[];locations=[];seen=set();counts=Counter()
    for shard in manifest["shards"]:
        key=shard.get("dataset"),str(shard.get("subject_id"))
        if key not in parents or key in counts:
            raise SourceTrainingError("CAR3_source_namespace_duplicate_or_non_source")
        counts[key]+=1
        if shard.get("role")!="source_sensitivity_preparation_only" or shard.get("stage")!=preparation.STAGE3:
            raise SourceTrainingError("CAR3_pre_RMS_source_role_required")
        paths={k:preparation.checked(root,shard[k],files,required_prefix="cache/source_c3czc4_v1")
               for k in ("X","y","events","uids","exclusions","header")}
        x=np.load(paths["X"],mmap_mode="r",allow_pickle=False);y=np.load(paths["y"],allow_pickle=False)
        events=json.loads(paths["events"].read_text());uids=json.loads(paths["uids"].read_text())
        header=json.loads(paths["header"].read_text());original=parents[key]
        px=np.load(original["paths"]["X"],mmap_mode="r",allow_pickle=False)
        py=np.load(original["paths"]["y"],allow_pickle=False)
        pe=json.loads(original["paths"]["events"].read_text())
        n=original["n"]
        if (x.shape!=(n,3,128) or x.dtype!=np.dtype("float32") or y.shape!=(n,) or y.dtype!=np.dtype("int64")
                or not np.array_equal(y,py) or len(events)!=n or len(uids)!=n
                or header.get("channels")!=["C3","Cz","C4"] or header.get("reference")!="CAR_exact_C3_Cz_C4"
                or header.get("normalization_applied") is not False or header.get("training_ready") is not False):
            raise SourceTrainingError("CAR3_shard_signal_labels_header_mismatch")
        for start in range(0,n,64):
            projected=preparation.project_car3(px[start:start+64]).astype(np.float32)
            if not np.array_equal(projected,x[start:start+64]):
                raise SourceTrainingError("CAR3_signal_not_exact_parent_projection")
        for i,row in enumerate(events):
            uid=row.get("event_uid")
            if (uid in seen or uid!=uids[i] or uid!=pe[i]["event_uid"] or row.get("trial_uid")!=uid
                    or row.get("dataset")!=key[0] or str(row.get("subject_id"))!=key[1]
                    or row.get("role")!="source_sensitivity_preparation_only"
                    or row.get("input_contract_sha256")!=PREPARATION_SHA
                    or row.get("label")!=int(y[i]) or row.get("canonical_label")!=pe[i]["canonical_label"]
                    or row.get("processed_signal_sha256")!=preparation.signal_sha(x[i])):
                raise SourceTrainingError("CAR3_UID_label_signal_identity_mismatch")
            seen.add(uid);rows.append(row);labels.append(int(y[i]));locations.append((x,i))
    if len(rows)!=28830 or set(k[0] for k in counts)!=SOURCE_IDS:
        raise SourceTrainingError("CAR3_complete_four_source_coverage_required")
    files[CONTRACT]=preparation.descriptor(root,contract_path)
    return VerifiedCar3Cache(root,rows,np.asarray(labels,dtype=np.int64),locations,
        {k:v["sha256"] for k,v in files.items()},digest(contract))


def build_training_plan(cache,*,profile="development"):
    if profile not in SEED_PROFILES or {r["dataset"] for r in cache.rows}!=SOURCE_IDS:
        raise SourceTrainingError("Registered_CAR3_four_source_seed_profile_required")
    if len({r["event_uid"] for r in cache.rows})!=len(cache.rows):
        raise SourceTrainingError("CAR3_duplicate_UID")
    signals=defaultdict(set)
    for row in cache.rows:signals[row["processed_signal_sha256"]].add(row["dataset"])
    if any(len(v)>1 for v in signals.values()):raise SourceTrainingError("CAR3_cross_source_duplicate_signal")
    jobs=[]
    for phase,held in [("lodo",d) for d in sorted(SOURCE_IDS)]+[("final",None)]:
        train=[i for i,r in enumerate(cache.rows) if held is None or r["dataset"]!=held]
        val=[i for i,r in enumerate(cache.rows) if r["dataset"]==held] if held else []
        if set(cache.labels[train])!={0,1} or (val and set(cache.labels[val])!={0,1}):
            raise SourceTrainingError("CAR3_explicit_LR_classes_required")
        for seed in SEED_PROFILES[profile]:
            jobs.append({"job_id":f"{phase}/{held or 'ALL_SOURCE'}/eegnet3/seed-{seed}","phase":phase,
                "heldout_dataset":held,"seed":seed,"train_indices":train,"validation_indices":val,
                "train_uids_sha256":digest(_uids(cache,train)),"validation_uids_sha256":digest(_uids(cache,val)),
                "train_dataset_ids":sorted({cache.rows[i]["dataset"] for i in train}),
                "max_epochs":80 if held else None})
    plan={"schema_version":"paper3-CAR3-source-f0-plan-v1","profile":profile,"seeds":list(SEED_PROFILES[profile]),
        "architecture":ARCHITECTURE,"cache_manifest":{"path":cache.manifest_path,"sha256":cache.manifest_sha256},
        "input_contract_sha256":cache.input_contract_sha256,"preparation_contract_sha256":PREPARATION_SHA,
        "configuration":copy.deepcopy(SOURCE_CONFIG),"policy_and_code":_code_policy(cache.root),"jobs":jobs,
        "planned_neural_fits":len(jobs),"max_committed_updates":len(jobs)*80*64,
        "max_observed_plus_uncertain_optimizer_calls":len(jobs)*80*64*2,
        "status":"CAR3_SOURCE_F0_PLANNED_NOT_TRAINED","training_started":False,
        "independent_3G_and_all14_methods_ready":False,"Ma2022_in_source":False}
    return {**plan,"plan_sha256":digest(plan)}


def validate_training_plan(cache,plan):
    if not isinstance(plan,dict) or plan!=build_training_plan(cache,profile=plan.get("profile")):
        raise SourceTrainingError("Exact_complete_CAR3_training_plan_required")


def fit_identity(cache,job,config,runtime,*,synthetic_fixture=False):
    return {"schema_version":"paper3-CAR3-source-fit-v1","job":job,"configuration":config,
        "architecture":ARCHITECTURE,"input_contract_sha256":cache.input_contract_sha256,
        "preparation_contract_sha256":PREPARATION_SHA,"cache_manifest_sha256":cache.manifest_sha256,
        "cache_artifacts":dict(cache.artifact_sha256),"upstream_acceptance_sha256":{},
        "policy_and_code":_code_policy(cache.root),"runtime":runtime,"device":runtime["device"],
        "synthetic":bool(synthetic_fixture)}


def _copy_model(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def predict_indices(model, cache, indices, *, device, batch_size=256):
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = np.asarray(indices[start:start + batch_size], dtype=np.int64)
            batch = joint_trial_rms(torch.from_numpy(cache.batch(selected))).float().to(device)
            values.append(model(batch).detach().cpu())
    logits = torch.cat(values) if values else torch.empty((0, 2))
    return logits.numpy(), torch.softmax(logits, dim=1).numpy()


def run_fit(cache, job, folder, *, runtime, epochs, config=None,
            max_new_updates=None, synthetic_fixture=False, deadline=None, heartbeat=None):
    """Resumable kernel. CPU is restricted to explicit mathematical test fixtures.

    The public CLI never exposes synthetic_fixture/config overrides. A fixture
    status cannot be sealed as a scientific source fit or exported as f0.
    """
    config = copy.deepcopy(SOURCE_CONFIG if config is None else config)
    if not synthetic_fixture and (config != SOURCE_CONFIG or runtime.get("device") != "cuda:0"):
        raise SourceTrainingError("Real source fits require the fixed configuration and CUDA runtime")
    if not synthetic_fixture and job["phase"] == "lodo" and epochs != 80:
        raise SourceTrainingError("Source LODO uses the registered 80-epoch ceiling and source early stop")
    if not 1 <= epochs <= config["epochs"] or (max_new_updates is not None and max_new_updates < 1):
        raise SourceTrainingError("Invalid bounded source fit length")
    device = runtime["device"]
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    effective = {**config, "effective_epochs": epochs}
    identity = fit_identity(cache, job, effective, runtime, synthetic_fixture=synthetic_fixture)
    identity_sha = digest(identity)
    with (folder / ".fit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceTrainingError("The same source fit is already active") from None
        identity_file = folder / "identity.json"
        if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
            raise SourceTrainingError("Source checkpoint data/code/config/runtime identity changed")
        immutable_json(identity_file, identity)
        # A kill after immutable artifact publication but before status is a
        # finalization retry, not another optimizer/prediction/checkpoint write.
        if (folder / "artifact_manifest.json").exists():
            if not synthetic_fixture:
                existing = verify_complete_fit(cache,job,folder,recover_finalization=True)
                if heartbeat is not None:heartbeat()
                if deadline is not None and time.monotonic()>deadline:
                    raise SourceTrainingError("CAR3_finalization_crossed_wall_deadline")
                atomic_json(folder / "status.json", {"status":"CAR3_SOURCE_FIT_COMPLETE",
                    "synthetic":False,"identity_sha256":identity_sha,"training_started":True,
                    "artifact_manifest":artifact(cache.root,folder / "artifact_manifest.json")})
                return {**existing,"new_updates_this_invocation":0,"wall_seconds_this_invocation":0.}
            status=json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
            if status.get("status")=="SYNTHETIC_FIXTURE_COMPLETE":
                manifest=json.loads((folder / "artifact_manifest.json").read_text())
                for item in manifest["files"]:verify_artifact(cache.root,item)
                existing=json.loads((folder / "fit_summary.json").read_text())
                return {**existing,"new_updates_this_invocation":0,"wall_seconds_this_invocation":0.}
        seed = job["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(config["threads"])
        model = CompactEEGNet3(3, cache.n_times, config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
            eps=config["eps"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"], foreach=False)
        sampler = torch.Generator(device="cpu").manual_seed(seed + 1000003)
        last = folder / "last_state.pt"
        progress_file = folder / "execution_progress.json"
        state = {"identity": identity, "identity_sha256": identity_sha, "completed_updates": 0,
                 "completed_epochs": 0, "update_in_epoch": 0, "epoch_loss_sum": 0.0,
                 "history": [], "best_loss": None, "best_epoch": 0, "bad_epochs": 0,
                 "best_model": None, "observed_optimizer_calls_total": 0}
        if last.exists() or checkpoint_sidecar(last).exists():
            saved = verified_checkpoint(last)
            if saved.get("identity_sha256") != identity_sha or saved.get("identity") != identity:
                raise SourceTrainingError("Checkpoint model/optimizer/RNG/cursor identity changed")
            # If SIGKILL occurs between the epoch commit and its archive, the
            # committed boundary checkpoint itself is the exact missing file.
            if config["retain_epoch_checkpoints"] and saved["completed_epochs"] > 0 and saved["update_in_epoch"] == 0:
                epoch_file = folder / f"epoch_{saved['completed_epochs']:04d}.pt"
                if not epoch_file.exists() and not checkpoint_sidecar(epoch_file).exists():
                    immutable_checkpoint(epoch_file, saved)
                else:
                    archived = verified_checkpoint(epoch_file)
                    from tools.run_controlled_pilot import _same_tree
                    if (archived.get("identity_sha256") != identity_sha or
                            archived.get("completed_updates") != saved["completed_updates"] or
                            archived.get("update_in_epoch") != 0 or
                            any(not _same_tree(archived.get(key), saved.get(key)) for key in ("model", "optimizer", "rng"))):
                        raise SourceTrainingError("Retained epoch checkpoint identity/cursor differs")
            model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"], sampler, device=device)
            state = {key: value for key, value in saved.items() if key not in {"model", "optimizer", "rng"}}
        else:
            if progress_file.exists() or (folder / "status.json").exists():
                raise SourceTrainingError("Initial source checkpoint is missing after observed work; it was not reconstructed")
            state.update(model=_copy_model(model), optimizer=optimizer.state_dict(), rng=capture_rng(sampler, device=device))
            atomic_checkpoint(last, state)
            state = {key: value for key, value in state.items() if key not in {"model", "optimizer", "rng"}}
        prior = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        if prior and prior.get("identity_sha256") != identity_sha:
            raise SourceTrainingError("Source optimizer-call journal belongs to a different fit identity")
        observed_total = max(state["observed_optimizer_calls_total"], prior.get("observed_optimizer_calls_total", 0))
        replayed_observed = max(0, prior.get("last_observed_trajectory_update", 0) - state["completed_updates"])
        state["observed_optimizer_calls_total"] = observed_total
        train = np.asarray(job["train_indices"], dtype=np.int64)
        val = np.asarray(job["validation_indices"], dtype=np.int64)
        weights = hierarchical_weights([cache.rows[i] for i in train], np.asarray(cache.labels)[train])
        target_updates = epochs * config["steps_per_epoch"]
        initial_update = state["completed_updates"]
        has_val = bool(len(val)); elapsed = time.monotonic(); new_updates = 0
        pause_reason = "explicit_update_pause"
        atomic_json(folder / "status.json", {"status": "RUNNING", "synthetic": synthetic_fixture,
            "identity_sha256": identity_sha, "training_started": state["completed_updates"] > 0})

        def commit():
            atomic_checkpoint(last, {**state, "model": _copy_model(model), "optimizer": optimizer.state_dict(),
                                      "rng": capture_rng(sampler, device=device)})

        while state["completed_updates"] < target_updates and not (has_val and state["bad_epochs"] >= config["patience"]):
            if deadline is not None and time.monotonic() >= deadline:
                pause_reason = "operational_wall_budget_pause"
                break
            if max_new_updates is not None and new_updates >= max_new_updates:
                break
            model.train()
            sampled = torch.multinomial(weights, config["batch_size"], replacement=True, generator=sampler).numpy()
            indices = train[sampled]
            batch = joint_trial_rms(torch.from_numpy(cache.batch(indices))).float().to(device)
            labels = torch.from_numpy(np.asarray(cache.labels)[indices].astype(np.int64, copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch), labels)
            if not bool(torch.isfinite(loss)):
                raise SourceTrainingError("Nonfinite source optimization loss")
            loss.backward(); optimizer.step()
            state["completed_updates"] += 1; state["update_in_epoch"] += 1
            state["epoch_loss_sum"] += float(loss.detach().cpu()); new_updates += 1
            state["observed_optimizer_calls_total"] += 1
            atomic_json(progress_file, {"identity_sha256": identity_sha,
                "last_observed_trajectory_update": state["completed_updates"],
                "observed_optimizer_calls_total": state["observed_optimizer_calls_total"],
                "replayed_observed_updates_this_invocation": replayed_observed,
                "unobserved_optimizer_call_before_hard_kill_remains_possible": True})
            if heartbeat is not None:
                heartbeat()
            if state["update_in_epoch"] == config["steps_per_epoch"]:
                epoch = state["completed_epochs"] + 1
                current = None
                if has_val:
                    _, probabilities = predict_indices(model, cache, val, device=device)
                    current = participant_validation_loss([cache.rows[i] for i in val], np.asarray(cache.labels)[val], probabilities)
                if not has_val or state["best_loss"] is None or current < state["best_loss"] - config["min_delta"]:
                    state.update(best_loss=current, best_epoch=epoch, bad_epochs=0, best_model=_copy_model(model))
                else:
                    state["bad_epochs"] += 1
                state["history"].append({"epoch": epoch, "optimizer_updates": state["completed_updates"],
                    "train_cross_entropy": state["epoch_loss_sum"] / config["steps_per_epoch"],
                    "source_validation_cross_entropy": current})
                state.update(completed_epochs=epoch, update_in_epoch=0, epoch_loss_sum=0.0)
                commit()
                if config["retain_epoch_checkpoints"]:
                    immutable_checkpoint(folder / f"epoch_{epoch:04d}.pt", verified_checkpoint(last))
            elif state["completed_updates"] % config["checkpoint_every_updates"] == 0:
                commit()
        commit()
        complete = state["completed_updates"] >= target_updates or (has_val and state["bad_epochs"] >= config["patience"])
        summary = {"complete": complete, "synthetic": synthetic_fixture, "identity_sha256": identity_sha,
            "epochs_completed": state["completed_epochs"], "selected_epoch": state["best_epoch"],
            "committed_optimizer_updates": state["completed_updates"], "new_updates_this_invocation": new_updates,
            "observed_optimizer_calls_total": state["observed_optimizer_calls_total"],
            "replayed_observed_updates_this_invocation": replayed_observed,
            "best_source_validation_cross_entropy": state["best_loss"], "history": state["history"],
            "wall_seconds_this_invocation": time.monotonic() - elapsed,
            "stop_reason": "source_early_stopping" if has_val and state["bad_epochs"] >= config["patience"] else "fixed_epoch_budget" if complete else pause_reason,
            "checkpoint_cursor": {"epoch": state["completed_epochs"], "update_in_epoch": state["update_in_epoch"]},
            "training_started": state["completed_updates"] > 0,
            "exact_continuation_scope": "same verified code/data/config/runtime identity only",
            "gpu_exact_resume_empirically_verified": False}
        atomic_json(folder / "fit_summary.json", summary)
        if not complete:
            atomic_json(folder / "status.json", {"status": "PAUSED_AT_COMMITTED_UPDATE", "synthetic": synthetic_fixture,
                                                 "identity_sha256": identity_sha, "training_started": summary["training_started"]})
            return summary
        if state["best_model"] is None:
            raise SourceTrainingError("No completed epoch exists for source checkpoint selection")
        selected = {"identity": identity, "identity_sha256": identity_sha, "model": state["best_model"],
                    "channels": 3, "n_times": cache.n_times, "config": config,
                    "selected_epoch": state["best_epoch"], "synthetic": synthetic_fixture}
        selected_path = folder / "selected.pt"
        if selected_path.exists() or checkpoint_sidecar(selected_path).exists():
            previous_selected = verified_checkpoint(selected_path)
            from tools.run_controlled_pilot import _same_tree
            if not _same_tree(previous_selected, selected):
                raise SourceTrainingError("Existing selected source checkpoint differs from the verified committed selection")
        else:
            atomic_checkpoint(selected_path, selected)
        model.load_state_dict(state["best_model"], strict=True)
        for role, indices in (("training", train), ("source_validation", val)):
            immutable_json(folder / f"{role}_uids.json", {
                "schema_version": "paper3-source-training-uids-v2" if role == "training" else "paper3-source-validation-uids-v2",
                "role": "source_pretrain_train" if role == "training" else "source_validation",
                "signal_contract_sha256": cache.input_contract_sha256, "event_uids": _uids(cache, indices),
                "synthetic": synthetic_fixture, "pipeline_seed": seed})
            if len(indices):
                logits, probabilities = predict_indices(model, cache, indices, device=device)
                immutable_npy(folder / f"{role}_logits.npy", logits, allow_pickle=False)
                immutable_npy(folder / f"{role}_probabilities.npy", probabilities, allow_pickle=False)
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix in {".json", ".pt", ".npy"}
                 and p.name not in {"status.json", "artifact_manifest.json"}]
        immutable_json(folder / "artifact_manifest.json", {"identity_sha256": identity_sha,
            "files": [artifact(cache.root, path) for path in sorted(files)]})
        if heartbeat is not None:heartbeat()
        if deadline is not None and time.monotonic()>deadline:
            raise SourceTrainingError("CAR3_prediction_or_serialization_crossed_wall_deadline")
        atomic_json(folder / "status.json", {"status": "SYNTHETIC_FIXTURE_COMPLETE" if synthetic_fixture else "CAR3_SOURCE_FIT_COMPLETE",
            "synthetic": synthetic_fixture, "identity_sha256": identity_sha, "training_started": True,
            "artifact_manifest": artifact(cache.root, folder / "artifact_manifest.json")})
        return summary


def verify_complete_fit(cache, job, folder, *, recover_finalization=False):
    folder = Path(folder)
    if recover_finalization:
        # Validate every durable artifact before publishing any completion flag.
        identity_record=json.loads((folder / "identity.json").read_text())
        status={"status":"CAR3_SOURCE_FIT_COMPLETE","synthetic":False,
            "identity_sha256":digest(identity_record),
            "artifact_manifest":artifact(cache.root,folder / "artifact_manifest.json")}
    else:
        status = json.loads((folder / "status.json").read_text())
    if status.get("status") != "CAR3_SOURCE_FIT_COMPLETE" or status.get("synthetic") is not False:
        raise SourceTrainingError("A real completed source fit is required; fixtures/paused plans are pending")
    identity = json.loads((folder / "identity.json").read_text())
    if (identity.get("architecture") != ARCHITECTURE or identity.get("schema_version") != "paper3-CAR3-source-fit-v1" or identity.get("job") != job or identity.get("synthetic") is not False or
            identity.get("runtime", {}).get("device") != "cuda:0" or
            identity.get("input_contract_sha256") != cache.input_contract_sha256 or
            identity.get("cache_manifest_sha256") != cache.manifest_sha256 or
            identity.get("cache_artifacts") != {k: v for k, v in cache.artifact_sha256.items() if k != "status/tasks.json"} or
            identity.get("upstream_acceptance_sha256") != dict(getattr(cache, "upstream_acceptance_sha256", {})) or
            identity.get("policy_and_code") != _code_policy(cache.root) or
            status.get("identity_sha256") != digest(identity)):
        raise SourceTrainingError("Completed source fit identity is stale")
    manifest = json.loads(verify_artifact(cache.root, status["artifact_manifest"]).read_text())
    if manifest.get("identity_sha256") != digest(identity) or not isinstance(manifest.get("files"), list):
        raise SourceTrainingError("Source artifact manifest identity is incomplete")
    files = [verify_artifact(cache.root, item) for item in manifest["files"]]
    if len(set(files)) != len(files) or any(path.parent != folder.resolve() for path in files):
        raise SourceTrainingError("Source artifact manifest duplicates or crosses fit directories")
    summary = json.loads((folder / "fit_summary.json").read_text())
    if not summary.get("complete") or summary.get("synthetic") is not False or summary.get("selected_epoch", 0) < 1:
        raise SourceTrainingError("Actual source fit completion/selection evidence missing")
    config = identity["configuration"]
    if (set(config) != set(SOURCE_CONFIG) | {"effective_epochs"} or
            {key: config.get(key) for key in SOURCE_CONFIG} != SOURCE_CONFIG):
        raise SourceTrainingError("Completed source fit configuration differs from the frozen source policy")
    if job["phase"] == "lodo" and config.get("effective_epochs") != 80:
        raise SourceTrainingError("Completed source LODO did not use its registered epoch ceiling")
    last = verified_checkpoint(folder / "last_state.pt")
    selected = verified_checkpoint(folder / "selected.pt")
    from tools.run_controlled_pilot import _same_tree
    if (last.get("identity_sha256") != digest(identity) or selected.get("identity_sha256") != digest(identity) or
            last.get("identity") != identity or selected.get("identity") != identity or
            summary.get("identity_sha256") != digest(identity) or
            summary["committed_optimizer_updates"] != last.get("completed_updates") or
            last.get("completed_updates") != last.get("completed_epochs", 0) * config["steps_per_epoch"] or
            last.get("completed_epochs") != summary["epochs_completed"] or
            last.get("history") != summary.get("history") or last.get("update_in_epoch") != 0 or
            selected.get("selected_epoch") != summary["selected_epoch"] or
            last.get("best_epoch") != summary["selected_epoch"] or
            not _same_tree(selected.get("model"), last.get("best_model")) or
            selected.get("channels") != 3 or selected.get("n_times") != cache.n_times or
            selected.get("config") != SOURCE_CONFIG or selected.get("synthetic") is not False or
            not 1 <= summary["selected_epoch"] <= summary["epochs_completed"]):
        raise SourceTrainingError("Committed optimizer/cursor/selected-weight evidence is incomplete")
    terminal = (summary["epochs_completed"] == config["effective_epochs"] or
                (job["phase"] == "lodo" and last.get("bad_epochs", 0) >= config["patience"]))
    if not terminal or config.get("effective_epochs", 0) > 80:
        raise SourceTrainingError("Source fit did not reach its fixed budget or registered early stop")
    expected_files = {"identity.json", "last_state.pt", "selected.pt", "execution_progress.json",
                      "fit_summary.json", "training_uids.json", "source_validation_uids.json",
                      "training_logits.npy", "training_probabilities.npy"}
    expected_files.update(f"epoch_{epoch:04d}.pt" for epoch in range(1, summary["epochs_completed"] + 1))
    expected_files.update(name + ".sha256.json" for name in tuple(expected_files) if name.endswith(".pt"))
    if job["validation_indices"]:
        expected_files.update({"source_validation_logits.npy", "source_validation_probabilities.npy"})
    if {path.name for path in files} != expected_files:
        raise SourceTrainingError("Complete source fit artifact coverage is required")
    for path in files:
        if path.suffix == ".pt":
            verified_checkpoint(path, load=False)
    progress = json.loads((folder / "execution_progress.json").read_text())
    if (progress.get("identity_sha256") != digest(identity) or
            progress.get("last_observed_trajectory_update") != summary["committed_optimizer_updates"] or
            progress.get("observed_optimizer_calls_total") != summary.get("observed_optimizer_calls_total") or
            summary.get("observed_optimizer_calls_total", -1) < summary["committed_optimizer_updates"] or
            last.get("observed_optimizer_calls_total") != summary.get("observed_optimizer_calls_total")):
        raise SourceTrainingError("Actual source optimizer-call journal differs from its committed trajectory")
    uid = json.loads((folder / "training_uids.json").read_text())
    if (uid.get("schema_version") != "paper3-source-training-uids-v2" or
            uid.get("role") != "source_pretrain_train" or uid.get("synthetic") is not False or
            uid.get("signal_contract_sha256") != cache.input_contract_sha256 or
            uid.get("pipeline_seed") != job["seed"] or uid.get("event_uids") != _uids(cache, job["train_indices"]) or
            len(set(uid["event_uids"])) != len(uid["event_uids"])):
        raise SourceTrainingError("Actual f0 training UID evidence differs from its source partition")
    validation_uid = json.loads((folder / "source_validation_uids.json").read_text())
    if (validation_uid.get("schema_version") != "paper3-source-validation-uids-v2" or
            validation_uid.get("role") != "source_validation" or validation_uid.get("synthetic") is not False or
            validation_uid.get("signal_contract_sha256") != cache.input_contract_sha256 or
            validation_uid.get("pipeline_seed") != job["seed"] or
            validation_uid.get("event_uids") != _uids(cache, job["validation_indices"])):
        raise SourceTrainingError("Source validation UID evidence differs from the held-out dataset")
    for role, indices in (("training", job["train_indices"]), ("source_validation", job["validation_indices"])):
        if not indices:
            continue
        logits = np.load(folder / f"{role}_logits.npy", allow_pickle=False)
        probabilities = np.load(folder / f"{role}_probabilities.npy", allow_pickle=False)
        if (logits.shape != (len(indices), 2) or probabilities.shape != logits.shape or
                not np.isfinite(logits).all() or not np.isfinite(probabilities).all() or
                not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-6)):
            raise SourceTrainingError("Source fit prediction evidence has invalid coverage or values")
    return summary


def _safe_output(cache,output):
    output=Path(output)
    if output.is_absolute():
        try:relative=output.relative_to(cache.root).as_posix()
        except ValueError:raise SourceTrainingError("CAR3_output_outside_project") from None
    else:relative=output.as_posix()
    if not relative.startswith("runs/"):
        raise SourceTrainingError("CAR3_output_requires_independent_runs_namespace")
    if any(part in {"source_training_v2","controlled_source_pilot_seed20261018"} for part in Path(relative).parts):
        raise SourceTrainingError("Main11_output_namespace_forbidden")
    return local(cache.root,relative,exists=False)


def _claim(cache,output,runtime,scope,*,seeds=()):
    output=_safe_output(cache,output)
    common={"schema_version":"paper3-CAR3-exclusive-output-claim-v1","scope":scope,
        "output":output.relative_to(cache.root).as_posix(),"architecture":ARCHITECTURE,
        "cache_manifest_sha256":cache.manifest_sha256,"input_contract_sha256":cache.input_contract_sha256,
        "runtime":runtime,"main11_pilot_or_output_reused":False}
    path=output/"output_claim.json"
    if not path.exists() and output.exists() and any(p.name!=".car3.lock" for p in output.iterdir()):
        raise SourceTrainingError("Existing_unclaimed_CAR3_output_cannot_be_adopted")
    claims=cache.root/"runs/car3_source_execution_claims"
    identities=[("engineering-v1",None)] if scope=="CAR3_SOURCE_ENGINEERING_ONLY" else [(f"source-f0-seed-{s}",s) for s in seeds]
    for name,seed in identities:
        immutable_json(claims/(name+".json"),{**common,"seed":seed,"max_outputs_for_identity":1})
    immutable_json(path,common)
    return output,common


def _ledger(output,folders,*,limit,max_invocations=100000):
    path=output/"optimizer_call_ledger.json"
    if path.exists():state=json.loads(path.read_text())
    else:
        if any((Path(f)/"execution_progress.json").exists() or (Path(f)/"last_state.pt").exists() for f in folders):
            raise SourceTrainingError("Missing_CAR3_optimizer_ledger_not_reconstructed")
        state={"schema_version":"paper3-CAR3-observed-optimizer-calls-v1","observed_by_fit":{},
        "uncertain_optimizer_calls":0,"active_fit":None,"invocations":0,
        "max_observed_plus_uncertain_calls":limit,"max_invocations":max_invocations,
        "interruption_uncertainty_policy":"one possible call after last per-step journal before interruption"}
    if (state.get("schema_version")!="paper3-CAR3-observed-optimizer-calls-v1"
            or state.get("max_observed_plus_uncertain_calls")!=limit
            or state.get("max_invocations")!=max_invocations
            or not isinstance(state.get("observed_by_fit"),dict)
            or type(state.get("uncertain_optimizer_calls")) is not int
            or state["uncertain_optimizer_calls"]<0):
        raise SourceTrainingError("CAR3_optimizer_ledger_identity_or_schema_invalid")
    known={str(Path(folder).relative_to(output)):Path(folder) for folder in folders}
    if set(state["observed_by_fit"])-set(known):
        raise SourceTrainingError("CAR3_optimizer_ledger_unknown_fit")
    if state.get("active_fit") is not None:
        if state["active_fit"] not in known:raise SourceTrainingError("CAR3_active_fit_unknown")
        state["uncertain_optimizer_calls"]+=1
        state.setdefault("unclean_recoveries",[]).append({"fit":state["active_fit"],"charged_possible_unjournaled_calls":1})
        state["active_fit"]=None
        atomic_json(path,state)
    def observe():
        for name,folder in known.items():
            progress=folder/"execution_progress.json"
            value=json.loads(progress.read_text()).get("observed_optimizer_calls_total") if progress.exists() else 0
            if type(value) is not int or value<state["observed_by_fit"].get(name,0):
                raise SourceTrainingError("CAR3_observed_optimizer_journal_missing_or_rolled_back")
            state["observed_by_fit"][name]=value
        total=sum(state["observed_by_fit"].values())+state["uncertain_optimizer_calls"]
        state["observed_plus_uncertain_calls"]=total
        atomic_json(path,state)
        if total>limit:raise SourceTrainingError("CAR3_observed_optimizer_budget_exceeded")
        return max(0,limit-total)
    observe()
    return path,state,observe


def _run_with_ledger(cache,job,folder,output,runtime,epochs,state,path,observe,heartbeat,*,limit=None,deadline=None):
    remaining=observe()
    if state["invocations"]>=state["max_invocations"] or remaining<=0:
        return None
    cap=remaining if limit is None else min(remaining,limit)
    name=str(Path(folder).relative_to(output))
    state["active_fit"]=name;state["invocations"]+=1;atomic_json(path,state)
    def pulse():
        observe();heartbeat()
    try:
        result=run_fit(cache,job,folder,runtime=runtime,epochs=epochs,max_new_updates=cap,
                       deadline=deadline,heartbeat=pulse)
    except BaseException:
        # The active marker survives hard kill; a handled exception is charged
        # conservatively before clearing, including optimizer/journal ambiguity.
        state["uncertain_optimizer_calls"]+=1;state["active_fit"]=None
        atomic_json(path,state)
        observe()
        raise
    state["active_fit"]=None;atomic_json(path,state);observe()
    return result


def _start_wall(output,seconds,*,amend=None,note=None):
    if not (Path(output)/"operational_budget.json").exists() and (Path(output)/"optimizer_call_ledger.json").exists():
        prior=json.loads((Path(output)/"optimizer_call_ledger.json").read_text())
        if prior.get("observed_plus_uncertain_calls",0)>0:
            raise SourceTrainingError("Missing_CAR3_wall_ledger_not_reconstructed")
    path,state=operational_budget(output,initial_seconds=seconds,amend_seconds=amend,amendment_note=note)
    clock=_source_wall_clock()
    remaining=max(0.,state["authorized_total_seconds"]-state["consumed_active_seconds"])
    if remaining<=0:raise SourceTrainingError("CAR3_operational_wall_budget_exhausted")
    state["clock_state"]["interval"]={"base_wall_seconds":state["consumed_active_seconds"],
        "start_utc_seconds":clock["utc_seconds"],"start_monotonic_seconds":clock["monotonic_seconds"],
        "last_observed_utc_seconds":clock["utc_seconds"],"last_observed_monotonic_seconds":clock["monotonic_seconds"],
        "boot_id":clock["boot_id"],"intent":"persisted_before_CAR3_work"}
    state["active_invocation"]=True;atomic_json(path,state)
    def heartbeat():_observe_source_wall(path,state)
    def close():
        heartbeat()
        state["last_closed_interval"]=copy.deepcopy(state["clock_state"]["interval"])
        state["active_invocation"]=False;state["clock_state"]["interval"]=None;atomic_json(path,state)
        consumed=state.get("consumed_active_seconds")
        if (type(consumed) not in (int,float) or not math.isfinite(consumed) or consumed<0
                or state["clock_state"].get("clock_block") is not None
                or consumed>state["authorized_total_seconds"]):
            raise SourceTrainingError("CAR3_operational_wall_overrun_no_completion")
    return clock["monotonic_seconds"]+remaining,heartbeat,close


def _global_lock(cache):
    path=local(cache.root,"runs/car3_source_execution_claims/.exclusive.lock",exists=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open("a")
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close();raise SourceTrainingError("CAR3_source_or_engineering_job_already_active") from None
    return stream


def engineering_pilot(cache,output,*,runtime,max_wall_seconds=1800,amend_budget_seconds=None,amendment_note=None):
    """Future user-started GPU only: 8 continuous vs 4+4 exact continuation."""
    if runtime.get("device")!="cuda:0":raise SourceTrainingError("Real_CAR3_engineering_pilot_requires_CUDA")
    plan=build_training_plan(cache)
    lock=_global_lock(cache)
    try:
        output,claim=_claim(cache,output,runtime,"CAR3_SOURCE_ENGINEERING_ONLY")
        receipt=output/"engineering_receipt.json"
        if receipt.exists():return verify_engineering_pilot(cache,receipt,runtime=runtime)
        job=plan["jobs"][0];folders=[output/"continuous",output/"resumed"]
        ledger,state,observe=_ledger(output,folders,limit=32,max_invocations=8)
        deadline,heartbeat,close=_start_wall(output,max_wall_seconds,amend=amend_budget_seconds,note=amendment_note)
        try:
            specs=[("continuous",folders[0],0,8),("resumed_prefix",folders[1],0,4),("resumed_suffix",folders[1],4,8)]
            for name,folder,begin,target in specs:
                leg_path=output/(name+"_single_call.json")
                current=verified_checkpoint(folder/"last_state.pt")["completed_updates"] if (folder/"last_state.pt").exists() else 0
                if current>=target:
                    if not leg_path.exists():
                        raise SourceTrainingError("Missing_CAR3_single_call_leg_proof_not_reconstructed")
                    if current>target and not (name=="resumed_prefix" and current==8):
                        raise SourceTrainingError("CAR3_engineering_cursor_exceeds_registered_leg")
                    continue
                if current!=begin:
                    raise SourceTrainingError("Partial_CAR3_engineering_leg_cannot_be_called_uninterrupted_8_vs_4plus4")
                before=state["observed_by_fit"].get(str(folder.relative_to(output)),0)
                summary=_run_with_ledger(cache,job,folder,output,runtime,80,state,ledger,observe,
                    heartbeat,limit=target-begin,deadline=deadline)
                if summary is None or summary["committed_optimizer_updates"]<target:
                    return {"status":"CAR3_ENGINEERING_PAUSED_NOT_VERIFIED","training_fits_completed":0,
                        "observed_plus_uncertain_calls":state.get("observed_plus_uncertain_calls"),
                        "fixed_leg_interruption_requires_new_explicit_version_if_partial_cursor_committed":True}
                after=state["observed_by_fit"][str(folder.relative_to(output))]
                if summary["new_updates_this_invocation"]!=target-begin or after-before!=target-begin:
                    raise SourceTrainingError("CAR3_engineering_leg_was_not_one_complete_registered_call")
                checkpoint=folder/"last_state.pt"
                if name=="resumed_prefix":
                    checkpoint=output/"resumed_at_4.pt"
                    immutable_checkpoint(checkpoint,verified_checkpoint(folder/"last_state.pt"))
                immutable_json(leg_path,{"schema_version":"paper3-CAR3-engineering-single-call-v1",
                    "scope":"CAR3_SOURCE_ENGINEERING_ONLY","name":name,"start_completed_updates":begin,
                    "end_completed_updates":target,"new_updates_this_invocation":target-begin,
                    "observed_calls_before":before,"observed_calls_after":after,
                    "checkpoint":artifact(cache.root,checkpoint),"single_uninterrupted_kernel_call":True,
                    "formal_fit_complete":False})
            left=verified_checkpoint(folders[0]/"last_state.pt");right=verified_checkpoint(folders[1]/"last_state.pt")
            from tools.run_controlled_pilot import _same_tree
            if (left["completed_updates"]!=8 or right["completed_updates"]!=8
                    or any(not _same_tree(left[k],right[k]) for k in ("model","optimizer","rng"))):
                raise SourceTrainingError("CAR3_GPU_resume_model_optimizer_rng_not_bitwise_equal")
            selected=job["train_indices"][:16]
            outputs=[]
            for folder,saved in zip(folders,[left,right]):
                model=CompactEEGNet3().to(runtime["device"]);model.load_state_dict(saved["model"],strict=True)
                logits,_=predict_indices(model,cache,selected,device=runtime["device"])
                immutable_npy(folder/"engineering_logits.npy",logits);outputs.append(logits)
            if not np.array_equal(*outputs):raise SourceTrainingError("CAR3_GPU_resume_predictions_differ")
            proof={"schema_version":"paper3-CAR3-engineering-resume-v1","scope":"CAR3_SOURCE_ENGINEERING_ONLY",
                "status":"CAR3_ENGINEERING_RESUME_VERIFIED","architecture":ARCHITECTURE,
                "cache_manifest_sha256":cache.manifest_sha256,"input_contract_sha256":cache.input_contract_sha256,
                "runtime":runtime,"policy_and_code":_code_policy(cache.root),"claim":artifact(cache.root,output/"output_claim.json"),
                "continuous_updates":8,"split_updates":[4,4],"expected_optimizer_calls":16,
                "observed_plus_uncertain_optimizer_calls":state["observed_plus_uncertain_calls"],
                "ledger":artifact(cache.root,ledger),"model_optimizer_rng_bitwise_equal":True,
                "single_call_leg_proofs":[artifact(cache.root,output/(name+"_single_call.json")) for name in ("continuous","resumed_prefix","resumed_suffix")],
                "split_checkpoint":artifact(cache.root,output/"resumed_at_4.pt"),
                "split_checkpoint_integrity":artifact(cache.root,checkpoint_sidecar(output/"resumed_at_4.pt")),
                "prediction_indices":selected,"prediction_uids_sha256":digest(_uids(cache,selected)),
                "branches":[{"checkpoint":artifact(cache.root,f/"last_state.pt"),
                    "checkpoint_integrity":artifact(cache.root,checkpoint_sidecar(f/"last_state.pt")),
                    "logits":artifact(cache.root,f/"engineering_logits.npy")} for f in folders],
                "exact_resume_verified":True,"formal_source_fit_complete":False,"actual_scientific_fits":0,
                "main11_200updates_evidence_reused":False,"external_payloads_accessed":False}
        finally:close()
        proof["operational_resources"]=_freeze_resource_snapshot(cache,output,20261018,runtime,
            scope="CAR3_SOURCE_ENGINEERING_ONLY",statuses=[])
        immutable_json(receipt,proof)
        return verify_engineering_pilot(cache,receipt,runtime=runtime)
    finally:lock.close()


def verify_engineering_pilot(cache,path,*,runtime=None):
    path=Path(path)
    if not path.is_absolute():path=local(cache.root,str(path))
    else:path=local(cache.root,path.relative_to(cache.root).as_posix())
    proof=json.loads(path.read_text())
    if (proof.get("schema_version")!="paper3-CAR3-engineering-resume-v1"
            or proof.get("status")!="CAR3_ENGINEERING_RESUME_VERIFIED"
            or proof.get("scope")!="CAR3_SOURCE_ENGINEERING_ONLY"
            or proof.get("architecture")!=ARCHITECTURE
            or proof.get("cache_manifest_sha256")!=cache.manifest_sha256
            or proof.get("input_contract_sha256")!=cache.input_contract_sha256
            or proof.get("policy_and_code")!=_code_policy(cache.root)
            or proof.get("runtime",{}).get("device")!="cuda:0"
            or runtime is not None and proof["runtime"]!=runtime
            or proof.get("main11_200updates_evidence_reused") is not False
            or proof.get("actual_scientific_fits")!=0
            or proof.get("continuous_updates")!=8 or proof.get("split_updates")!=[4,4]
            or proof.get("expected_optimizer_calls")!=16):
        raise SourceTrainingError("Own_real_CAR3_GPU_resume_pilot_required_not_main11_proof")
    _verify_resource_snapshot(cache,proof.get("operational_resources"),path.parent,20261018,proof["runtime"],
        scope="CAR3_SOURCE_ENGINEERING_ONLY")
    claim=json.loads(verify_artifact(cache.root,proof["claim"]).read_text())
    if claim.get("scope")!="CAR3_SOURCE_ENGINEERING_ONLY" or claim.get("output")!=path.parent.relative_to(cache.root).as_posix():
        raise SourceTrainingError("CAR3_engineering_claim_output_mismatch")
    ledger=json.loads(verify_artifact(cache.root,proof["ledger"]).read_text())
    total=ledger.get("observed_plus_uncertain_calls")
    if (type(total) is not int or not 16<=total<=32 or total!=proof.get("observed_plus_uncertain_optimizer_calls")
            or ledger.get("active_fit") is not None or ledger.get("max_observed_plus_uncertain_calls")!=32):
        raise SourceTrainingError("CAR3_engineering_optimizer_budget_proof_invalid")
    plan=build_training_plan(cache);job=plan["jobs"][0];branches=proof.get("branches")
    if not isinstance(branches,list) or len(branches)!=2:raise SourceTrainingError("Two_CAR3_resume_branches_required")
    prefix=verify_artifact(cache.root,proof["split_checkpoint"])
    if prefix!=path.parent/"resumed_at_4.pt":raise SourceTrainingError("CAR3_split_checkpoint_path_mismatch")
    verify_artifact(cache.root,proof["split_checkpoint_integrity"])
    prefix_state=verified_checkpoint(prefix)
    expected=fit_identity(cache,job,{**SOURCE_CONFIG,"effective_epochs":80},proof["runtime"])
    if prefix_state.get("completed_updates")!=4 or prefix_state.get("identity")!=expected:
        raise SourceTrainingError("Real_CAR3_4update_resume_boundary_required")
    if ledger.get("invocations",0)<3:raise SourceTrainingError("CAR3_continuous_and_split_kernel_invocations_required")
    leg_items=proof.get("single_call_leg_proofs")
    if not isinstance(leg_items,list) or len(leg_items)!=3:
        raise SourceTrainingError("Three_real_CAR3_uninterrupted_leg_proofs_required")
    for item,(name,begin,end,checkpoint) in zip(leg_items,[
            ("continuous",0,8,path.parent/"continuous/last_state.pt"),
            ("resumed_prefix",0,4,path.parent/"resumed_at_4.pt"),
            ("resumed_suffix",4,8,path.parent/"resumed/last_state.pt")]):
        leg_path=verify_artifact(cache.root,item)
        if leg_path!=path.parent/(name+"_single_call.json"):
            raise SourceTrainingError("CAR3_single_call_leg_path_mismatch")
        leg=json.loads(leg_path.read_text())
        if (leg.get("schema_version")!="paper3-CAR3-engineering-single-call-v1" or leg.get("name")!=name
                or leg.get("scope")!="CAR3_SOURCE_ENGINEERING_ONLY" or leg.get("start_completed_updates")!=begin
                or leg.get("end_completed_updates")!=end or leg.get("new_updates_this_invocation")!=end-begin
                or type(leg.get("observed_calls_before")) is not int
                or leg.get("observed_calls_after",-1)-leg["observed_calls_before"]!=end-begin
                or leg.get("single_uninterrupted_kernel_call") is not True
                or verify_artifact(cache.root,leg.get("checkpoint"))!=checkpoint):
            raise SourceTrainingError("CAR3_engineering_fixed_leg_uninterrupted_evidence_invalid")
    states=[];logits=[]
    for name,branch in zip(["continuous","resumed"],branches):
        cp=verify_artifact(cache.root,branch["checkpoint"])
        if cp!=path.parent/name/"last_state.pt":raise SourceTrainingError("CAR3_resume_branch_path_mismatch")
        verify_artifact(cache.root,branch["checkpoint_integrity"])
        saved=verified_checkpoint(cp)
        expected=fit_identity(cache,job,{**SOURCE_CONFIG,"effective_epochs":80},proof["runtime"])
        if saved.get("identity")!=expected or saved.get("identity_sha256")!=digest(expected) or saved.get("completed_updates")!=8:
            raise SourceTrainingError("Actual_CAR3_resume_branch_identity_or_cursor_invalid")
        states.append(saved)
        lp=verify_artifact(cache.root,branch["logits"])
        if lp!=path.parent/name/"engineering_logits.npy":raise SourceTrainingError("CAR3_resume_logits_path_mismatch")
        logits.append(np.load(lp,allow_pickle=False))
    from tools.run_controlled_pilot import _same_tree
    if any(not _same_tree(states[0][key],states[1][key]) for key in ("model","optimizer","rng")):
        raise SourceTrainingError("Saved_CAR3_resume_states_not_bitwise_equal")
    selected=job["train_indices"][:16]
    if (proof.get("prediction_indices")!=selected or proof.get("prediction_uids_sha256")!=digest(_uids(cache,selected))
            or logits[0].shape!=(len(selected),2) or not np.isfinite(logits[0]).all() or not np.array_equal(*logits)):
        raise SourceTrainingError("Saved_CAR3_resume_prediction_proof_invalid")
    return {"receipt":artifact(cache.root,path),"exact_resume_verified":True,
        "scope":"CAR3_SOURCE_ENGINEERING_ONLY","formal_fits_completed":0,
        "runtime":proof["runtime"],"observed_plus_uncertain_calls":total}


def install_plan(cache,plan,output,*,extend_confirmation=False):
    path=output/"training_plan.json"
    if not path.exists():immutable_json(path,plan);return
    previous=json.loads(path.read_text())
    if previous==plan:return
    if (not extend_confirmation or previous.get("profile")!="development" or plan.get("profile")!="confirmation"):
        raise SourceTrainingError("CAR3_plan_change_requires_completed_registered_confirmation_extension")
    validate_training_plan(cache,previous)
    by_id={j["job_id"]:j for j in plan["jobs"]}
    for job in previous["jobs"]:
        if by_id.get(job["job_id"])!=job:raise SourceTrainingError("CAR3_extension_changes_existing_trajectory")
        verify_complete_fit(cache,job,output/job["job_id"])
    immutable_json(output/"development_plan_completed.json",previous)
    immutable_json(output/"confirmation_extension.json",{"old_plan_sha256":previous["plan_sha256"],
        "new_plan_sha256":plan["plan_sha256"],"completed_fits_reused":15,"new_fits_planned":10})
    atomic_json(path,plan)


def export_f0_receipts(cache,plan,output,*,pilot):
    validate_training_plan(cache,plan)
    if verify_engineering_pilot(cache,pilot.get("receipt",{}).get("path"))!=pilot:
        raise SourceTrainingError("CAR3_export_requires_same_real_engineering_pilot")
    summaries={j["job_id"]:verify_complete_fit(cache,j,output/j["job_id"]) for j in plan["jobs"]}
    receipts=[]
    for seed in plan["seeds"]:
        folds=[j for j in plan["jobs"] if j["phase"]=="lodo" and j["seed"]==seed]
        epochs=[summaries[j["job_id"]]["selected_epoch"] for j in folds]
        final=next(j for j in plan["jobs"] if j["phase"]=="final" and j["seed"]==seed)
        identity=json.loads((output/final["job_id"]/"identity.json").read_text())
        if identity["configuration"]["effective_epochs"]!=final_epoch_from_lodo(epochs):
            raise SourceTrainingError("CAR3_final_epoch_selection_mismatch")
        if any(json.loads((output/j["job_id"]/"identity.json").read_text())["runtime"]!=identity["runtime"] for j in folds):
            raise SourceTrainingError("CAR3_LODO_and_final_runtime_mismatch")
        lodo=output/f"car3_lodo_seed-{seed}.json"
        immutable_json(lodo,{"status":"CAR3_SOURCE_LODO_SELECTION_COMPLETE","seed":seed,
            "selected_epochs":epochs,"final_epochs":final_epoch_from_lodo(epochs),
            "folds":[artifact(cache.root,output/j["job_id"]/"status.json") for j in folds],
            "rule":SOURCE_CONFIG["final_epochs"],"source_only_selection":True})
        resources=_freeze_resource_snapshot(cache,output,seed,identity["runtime"],scope="CAR3_SOURCE_F0_ONLY",
            statuses=[artifact(cache.root,output/j["job_id"]/"status.json") for j in folds+[final]])
        receipt={"status":"CAR3_SOURCE_F0_COMPLETE","architecture":ARCHITECTURE,"synthetic":False,
            "operational_resources":resources,
            "seed":seed,"pipeline_seed":seed,"input_contract_sha256":cache.input_contract_sha256,
            "preparation_contract_semantic_sha256":PREPARATION_SHA,"source_cache_manifest_sha256":cache.manifest_sha256,
            "training_dataset_ids":sorted(SOURCE_IDS),"Ma2022_in_source":False,"source_lodo_selection_verified":True,
            "checkpoint":artifact(cache.root,output/final["job_id"]/"selected.pt"),
            "final_fit_status":artifact(cache.root,output/final["job_id"]/"status.json"),
            "training_uid_manifest":artifact(cache.root,output/final["job_id"]/"training_uids.json"),
            "source_lodo_receipt":artifact(cache.root,lodo),"runtime":identity["runtime"],
            "configuration":identity["configuration"],"channels":["C3","Cz","C4"],
            "centered_dimension_d":2,"Rref_shape":[2,2],"Rref_status":"PENDING_INDEPENDENT_CAR3_SOURCE_ESTIMATOR",
            "independent_3G_meta_training_ready":False,"all14_methods_ready":False,
            "primary11_bridge_compatible":False,"engineering_pilot":pilot,
            "formal_fit_exact_resume_empirical_comparison_performed":False,"overall_W4_complete":False}
        path=output/f"car3_f0_receipt_seed-{seed}.json";immutable_json(path,receipt);receipts.append(artifact(cache.root,path))
    return receipts


def execute_plan(cache,plan,output,*,runtime,pilot_path,max_new_updates=None,max_wall_seconds=86400,
                 only_job=None,extend_confirmation=False,amend_budget_seconds=None,amendment_note=None):
    validate_training_plan(cache,plan)
    if runtime.get("device")!="cuda:0":raise SourceTrainingError("Real_CAR3_source_training_requires_CUDA")
    pilot=verify_engineering_pilot(cache,pilot_path,runtime=runtime)
    if max_new_updates is not None and (type(max_new_updates) is not int or max_new_updates<=0):
        raise SourceTrainingError("Positive_CAR3_update_bound_required")
    if only_job and only_job not in {j["job_id"] for j in plan["jobs"]}:raise SourceTrainingError("Unknown_CAR3_job")
    lock=_global_lock(cache)
    try:
        output,claim=_claim(cache,output,runtime,"CAR3_SOURCE_F0_ONLY",seeds=plan["seeds"])
        install_plan(cache,plan,output,extend_confirmation=extend_confirmation)
        folders=[output/j["job_id"] for j in plan["jobs"]]
        # Confirmation may explicitly enlarge only this operational replay cap.
        ledger_path=output/"optimizer_call_ledger.json"
        if extend_confirmation and ledger_path.exists():
            ledger_json=json.loads(ledger_path.read_text())
            if ledger_json.get("max_observed_plus_uncertain_calls")==15*80*64*2:
                ledger_json["max_observed_plus_uncertain_calls"]=plan["max_observed_plus_uncertain_optimizer_calls"]
                atomic_json(ledger_path,ledger_json)
        ledger,state,observe=_ledger(output,folders,limit=plan["max_observed_plus_uncertain_optimizer_calls"])
        deadline,heartbeat,close=_start_wall(output,max_wall_seconds,amend=amend_budget_seconds,note=amendment_note)
        completed=[];pending=[];budget=max_new_updates
        try:
            for job,folder in zip(plan["jobs"],folders):
                if (folder/"status.json").exists() and json.loads((folder/"status.json").read_text()).get("status")=="CAR3_SOURCE_FIT_COMPLETE":
                    verify_complete_fit(cache,job,folder)
                    if json.loads((folder/"identity.json").read_text())["runtime"]!=runtime:
                        raise SourceTrainingError("CAR3_complete_fit_runtime_changed")
                    completed.append(job["job_id"]);continue
                if (only_job and job["job_id"]!=only_job) or budget==0 or time.monotonic()>=deadline:
                    pending.append(job["job_id"]);continue
                epochs=80
                if job["phase"]=="final":
                    folds=[j for j in plan["jobs"] if j["phase"]=="lodo" and j["seed"]==job["seed"]]
                    try:epochs=final_epoch_from_lodo([verify_complete_fit(cache,j,output/j["job_id"])["selected_epoch"] for j in folds])
                    except (ValueError,OSError):pending.append(job["job_id"]);continue
                result=_run_with_ledger(cache,job,folder,output,runtime,epochs,state,ledger,observe,heartbeat,
                    limit=budget,deadline=deadline)
                if result is None:pending.append(job["job_id"]);continue
                if budget is not None:budget-=result["new_updates_this_invocation"]
                (completed if result["complete"] else pending).append(job["job_id"])
        finally:close()
        result={"status":"CAR3_SOURCE_F0_COMPLETE" if not pending else "CAR3_SOURCE_TRAINING_PENDING",
            "plan_sha256":plan["plan_sha256"],"completed_neural_fits":len(completed),"pending_jobs":pending,
            "observed_plus_uncertain_calls":state["observed_plus_uncertain_calls"],
            "independent_3G_and_all14_methods_ready":False,"overall_W4_complete":False}
        if not pending:result["f0_receipts"]=export_f0_receipts(cache,plan,output,pilot=pilot)
        atomic_json(output/"car3_training_progress.json",result)
        return result
    finally:lock.close()


def load_verified_f0(cache,receipt_path):
    """Independent source-Rref prerequisite bridge; old clinic11 bridge rejects it."""
    path=local(cache.root,str(receipt_path));receipt=json.loads(path.read_text())
    if (receipt.get("status")!="CAR3_SOURCE_F0_COMPLETE" or receipt.get("architecture")!=ARCHITECTURE
            or receipt.get("synthetic") is not False or receipt.get("input_contract_sha256")!=cache.input_contract_sha256
            or receipt.get("source_cache_manifest_sha256")!=cache.manifest_sha256
            or set(receipt.get("training_dataset_ids",[]))!=SOURCE_IDS or receipt.get("channels")!=["C3","Cz","C4"]):
        raise SourceTrainingError("Independent_real_CAR3_f0_receipt_required")
    _verify_resource_snapshot(cache,receipt.get("operational_resources"),path.parent,receipt["seed"],receipt["runtime"],
        scope="CAR3_SOURCE_F0_ONLY")
    cp=verify_artifact(cache.root,receipt["checkpoint"])
    status=verify_artifact(cache.root,receipt["final_fit_status"])
    saved=verified_checkpoint(cp);identity=saved.get("identity",{})
    job=identity.get("job",{})
    if job.get("phase")!="final" or job.get("seed")!=receipt.get("seed") or saved.get("channels")!=3:
        raise SourceTrainingError("Actual_CAR3_final_checkpoint_required")
    expected=next(j for j in build_training_plan(cache,profile="confirmation")["jobs"] if j["phase"]=="final" and j["seed"]==receipt["seed"])
    if job!=expected or cp!=status.parent/"selected.pt":raise SourceTrainingError("Registered_CAR3_final_job_required")
    verify_complete_fit(cache,job,status.parent)
    if verify_engineering_pilot(cache,receipt["engineering_pilot"]["receipt"]["path"],runtime=receipt["runtime"])!=receipt["engineering_pilot"]:
        raise SourceTrainingError("CAR3_f0_pilot_provenance_mismatch")
    lodo=json.loads(verify_artifact(cache.root,receipt["source_lodo_receipt"]).read_text())
    if lodo.get("status")!="CAR3_SOURCE_LODO_SELECTION_COMPLETE" or lodo.get("seed")!=receipt["seed"] or len(lodo.get("folds",[]))!=4:
        raise SourceTrainingError("Four_real_CAR3_LODO_folds_required")
    selected=[];held=set()
    for item in lodo["folds"]:
        foldstatus=verify_artifact(cache.root,item);foldidentity=json.loads((foldstatus.parent/"identity.json").read_text())
        foldjob=foldidentity["job"]
        fold_expected=next((j for j in build_training_plan(cache,profile="confirmation")["jobs"] if j["phase"]=="lodo" and j["seed"]==receipt["seed"] and j["heldout_dataset"]==foldjob.get("heldout_dataset")),None)
        if foldjob!=fold_expected or foldidentity["runtime"]!=receipt["runtime"]:
            raise SourceTrainingError("CAR3_LODO_fold_runtime_or_registered_job_mismatch")
        selected.append(verify_complete_fit(cache,foldjob,foldstatus.parent)["selected_epoch"]);held.add(foldjob["heldout_dataset"])
    if held!=SOURCE_IDS or selected!=lodo.get("selected_epochs") or identity["configuration"]["effective_epochs"]!=final_epoch_from_lodo(selected):
        raise SourceTrainingError("CAR3_LODO_median_final_epoch_binding_required")
    if verify_artifact(cache.root,receipt["training_uid_manifest"])!=status.parent/"training_uids.json":
        raise SourceTrainingError("CAR3_training_UID_path_mismatch")
    model=CompactEEGNet3();model.load_state_dict(saved["model"],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,receipt


def _validate_closed_resource_values(wall,ledger,*,scope):
    """Read-only acceptance of a final, closed, within-budget resource snapshot."""
    def number(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
    if (not isinstance(wall,dict) or wall.get("scope")!="operational_only_no_scientific_change"
            or type(wall.get("initial_seconds")) is not int or wall["initial_seconds"]<=0
            or type(wall.get("authorized_total_seconds")) is not int
            or not isinstance(wall.get("amendments"),list)
            or wall.get("revision")!=1+len(wall["amendments"])):
        raise SourceTrainingError("CAR3_final_wall_snapshot_schema_invalid")
    authorized=wall["initial_seconds"]
    for item in wall["amendments"]:
        if (item.get("previous_seconds")!=authorized or type(item.get("new_seconds")) is not int
                or item["new_seconds"]<=authorized or item.get("scientific_identity_changed") is not False
                or not isinstance(item.get("note"),str) or not item["note"].strip()):
            raise SourceTrainingError("CAR3_final_wall_amendment_chain_invalid")
        authorized=item["new_seconds"]
    clock=wall.get("clock_state",{})
    consumed=wall.get("consumed_active_seconds")
    if (authorized!=wall["authorized_total_seconds"] or wall.get("active_invocation") is not False
            or not number(consumed) or consumed>authorized
            or clock.get("schema_version")!="paper3-source-operational-clock-v1"
            or clock.get("interval") is not None or clock.get("clock_block") is not None
            or clock.get("wall_high_water_seconds")!=consumed):
        raise SourceTrainingError("CAR3_final_wall_snapshot_not_closed_within_budget")
    if (not isinstance(ledger,dict) or ledger.get("schema_version")!="paper3-CAR3-observed-optimizer-calls-v1"
            or not isinstance(ledger.get("observed_by_fit"),dict)
            or any(type(v) is not int or v<0 for v in ledger["observed_by_fit"].values())
            or type(ledger.get("uncertain_optimizer_calls")) is not int or ledger["uncertain_optimizer_calls"]<0
            or ledger.get("active_fit") is not None
            or type(ledger.get("invocations")) is not int or ledger["invocations"]<0
            or type(ledger.get("max_invocations")) is not int or ledger["invocations"]>ledger["max_invocations"]):
        raise SourceTrainingError("CAR3_final_optimizer_snapshot_invalid")
    total=sum(ledger["observed_by_fit"].values())+ledger["uncertain_optimizer_calls"]
    cap=ledger.get("max_observed_plus_uncertain_calls")
    legal={32} if scope=="CAR3_SOURCE_ENGINEERING_ONLY" else {15*80*64*2,25*80*64*2}
    if cap not in legal or total!=ledger.get("observed_plus_uncertain_calls") or total>cap:
        raise SourceTrainingError("CAR3_final_optimizer_snapshot_exceeds_registered_budget")


def _freeze_resource_snapshot(cache,output,seed,runtime,*,scope,statuses):
    name="engineering_resources.json" if scope=="CAR3_SOURCE_ENGINEERING_ONLY" else f"resource_snapshot_seed-{seed}.json"
    path=output/name
    if path.exists():
        item=artifact(cache.root,path)
        saved=_verify_resource_snapshot(cache,item,output,seed,runtime,scope=scope)
        if saved.get("verified_fit_statuses")!=statuses:
            raise SourceTrainingError("Immutable_CAR3_seed_resource_fit_binding_changed")
        return item
    wall=output/"operational_budget.json";ledger=output/"optimizer_call_ledger.json"
    wall_value=json.loads(wall.read_text());ledger_value=json.loads(ledger.read_text())
    _validate_closed_resource_values(wall_value,ledger_value,scope=scope)
    value={"schema_version":"paper3-CAR3-final-resource-snapshot-v1","scope":scope,"seed":seed,
        "output":output.relative_to(cache.root).as_posix(),"architecture":ARCHITECTURE,
        "cache_manifest_sha256":cache.manifest_sha256,"input_contract_sha256":cache.input_contract_sha256,
        "runtime":runtime,"wall_snapshot":wall_value,"optimizer_snapshot":ledger_value,
        "live_files_at_snapshot":{"wall":artifact(cache.root,wall),"optimizer":artifact(cache.root,ledger)},
        "verified_fit_statuses":statuses,"new_fits_started":0,
        "confirmation_extension_preserves_existing_seed_snapshot":True}
    immutable_json(path,value)
    return artifact(cache.root,path)


def _verify_resource_snapshot(cache,item,output,seed,runtime,*,scope):
    path=verify_artifact(cache.root,item)
    name="engineering_resources.json" if scope=="CAR3_SOURCE_ENGINEERING_ONLY" else f"resource_snapshot_seed-{seed}.json"
    if path!=output/name:raise SourceTrainingError("CAR3_final_resource_snapshot_path_mismatch")
    value=json.loads(path.read_text())
    if (value.get("schema_version")!="paper3-CAR3-final-resource-snapshot-v1" or value.get("scope")!=scope
            or value.get("seed")!=seed or value.get("output")!=output.relative_to(cache.root).as_posix()
            or value.get("architecture")!=ARCHITECTURE or value.get("cache_manifest_sha256")!=cache.manifest_sha256
            or value.get("input_contract_sha256")!=cache.input_contract_sha256 or value.get("runtime")!=runtime):
        raise SourceTrainingError("CAR3_final_resource_snapshot_identity_mismatch")
    _validate_closed_resource_values(value.get("wall_snapshot"),value.get("optimizer_snapshot"),scope=scope)
    statuses=value.get("verified_fit_statuses")
    if not isinstance(statuses,list) or len(statuses)!=(0 if scope=="CAR3_SOURCE_ENGINEERING_ONLY" else 5):
        raise SourceTrainingError("CAR3_final_resource_snapshot_fit_coverage_invalid")
    for status in statuses:verify_artifact(cache.root,status)
    live=value.get("live_files_at_snapshot",{})
    for key,filename,payload in [("wall","operational_budget.json",value["wall_snapshot"]),
                                 ("optimizer","optimizer_call_ledger.json",value["optimizer_snapshot"])]:
        descriptor=live.get(key,{})
        encoded=(json.dumps(payload,indent=2,allow_nan=False)+"\n").encode()
        if (descriptor.get("path")!=(output/filename).relative_to(cache.root).as_posix()
                or descriptor.get("bytes")!=len(encoded)
                or descriptor.get("sha256")!=hashlib.sha256(encoded).hexdigest()):
            raise SourceTrainingError("CAR3_closed_resource_snapshot_original_file_SHA_binding_invalid")
    # Snapshot SHA is immutable; historical live file SHA values are provenance,
    # not a demand that later confirmation work keep mutable live ledgers fixed.
    return value
