"""Pinned MOABB raw adapters; only named source/development datasets are accessible."""
from __future__ import annotations
import hashlib
import inspect
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from .core import (LABELS, array_hash, canonical, digest, environment, file_hash, natural_key,
                   read_json, save_shard, write_json)

REGISTRY = {
    "BNCI2014_001": {"role": "source", "kwargs": {}, "doi": "10.3389/fnins.2012.00055"},
    "Cho2017": {"role": "source", "kwargs": {}, "doi": "10.5524/100295"},
    "Lee2019_MI": {"role": "source", "kwargs": {"train_run": True, "test_run": False,
                                                "resting_state": False}, "doi": "10.5524/100542"},
    "PhysionetMI": {"role": "source", "kwargs": {"imagined": True, "executed": False},
                    "doi": "10.13026/C28G6P"},
    "Ma2022": {"role": "development", "kwargs": {}, "doi": "10.1038/s41597-022-01647-1",
               "availability": "Not exported by pinned MOABB 1.7.2; audited NPZ adapter required."},
    "HS-Stroke": {"role": "holdout", "available": False},
    "SEFMEID": {"role": "holdout", "available": False},
    "GuttmannFlury2025_MI": {"role": "holdout", "available": False},
    "Liu2024": {"role": "holdout", "available": False},
    "Kumar2024": {"role": "excluded_previous_paper", "available": False},
    "Yang2025": {"role": "excluded_paper1_active", "available": False},
    "Dreyer2023": {"role": "excluded_paper1_active", "available": False},
}

def instantiate(name):
    if name not in REGISTRY or REGISTRY[name]["role"] not in ("source", "development"):
        raise ValueError("Dataset is not approved for this source preparation package: " + name)
    import moabb
    import moabb.datasets as datasets
    if moabb.__version__ != "1.7.2": raise RuntimeError("Use pinned MOABB 1.7.2; create a new protocol for changes")
    cls = getattr(datasets, name, None)
    if cls is None:
        raise RuntimeError(f"{name} is not exported by MOABB 1.7.2. It remains registered only. "
                           "Supply an independently audited NPZ/JSON adapter; do not install moving develop.")
    dataset = cls(**REGISTRY[name]["kwargs"])
    if not set(LABELS) <= dataset.event_id.keys(): raise ValueError("No explicit left/right-hand labels")
    return dataset

def run_fingerprint(raw):
    """All EEG channels, in adapter coordinates; not a provider-file checksum."""
    channels=[c for c,t in zip(raw.ch_names,raw.get_channel_types()) if t=="eeg"]
    h = hashlib.sha256(canonical({"channels": channels, "sfreq": float(raw.info["sfreq"]),
                                 "n_times": int(raw.n_times), "units": "MNE SI volts"}))
    for start in range(0, raw.n_times, 65536):
        h.update(np.ascontiguousarray(raw.get_data(picks=channels,start=start, stop=min(start+65536,raw.n_times)),
                                     dtype="<f8").tobytes())
    return h.hexdigest()

def extract_run(raw, name, subject, session, run, session_order, run_order, dataset, config):
    import mne
    prep = config["preprocessing"]
    # get_data() sets canonical annotations at dataset.interval[0]. Its annotations
    # already include the provider adapter's interval shift; never add it twice.
    event_map = {key: int(dataset.event_id[key]) for key in LABELS}
    descriptions={str(d) for d in raw.annotations.description
                  if not str(d).lower().startswith("bad")}
    if not (descriptions & set(LABELS)):
        # The pinned PhysionetMI imagined-only adapter also returns both-hands/
        # feet runs 6/10/14. These are explicitly non-target MI, not missing labels.
        if (name=="PhysionetMI" and {"hands","feet"}<=descriptions
            and descriptions<={"hands","feet","rest"}):
            return [],[],[],{"session":str(session),"run":str(run),"excluded_run":True,
                  "reason":"Known Physionet both-hands/feet imagery run; left/right hand task absent",
                  "annotation_counts":dict(Counter(str(d) for d in raw.annotations.description)),
                  "n_binary_events":0,"n_retained":0,"exclusions":[],
                  "full_eeg_channel_names":[c for c,t in zip(raw.ch_names,raw.get_channel_types()) if t=="eeg"]}
        raise ValueError("No canonical left/right annotations in a run expected to contain the target task")
    events, _ = mne.events_from_annotations(raw, event_id=event_map, use_rounding=True, verbose="ERROR")
    inverse = {code: key for key, code in event_map.items()}
    if len({int(e[0]) for e in events}) != len(events): raise ValueError("Duplicate/ambiguous event sample")
    names = {ch.casefold(): ch for ch in raw.ch_names}
    if len(names) != len(raw.ch_names): raise ValueError("Ambiguous channel case normalization")
    missing = [ch for ch in prep["channels"] if ch.casefold() not in names]
    if missing: raise ValueError(f"{name}/{subject}/{session}/{run} missing channels {missing}; no interpolation")
    selected = [names[ch.casefold()] for ch in prep["channels"]]
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    if any(types[ch] != "eeg" for ch in selected): raise ValueError("Selected non-EEG channel")
    bad_selected = set(selected) & set(raw.info["bads"])
    if bad_selected: raise ValueError("Provider marks selected channels bad: " + str(sorted(bad_selected)))
    sfreq = float(raw.info["sfreq"])
    if not (0 < prep["fmin"] < prep["fmax"] < sfreq/2): raise ValueError("Invalid native bandpass")
    context_start, context_stop = prep["context_s"]
    crop_start, crop_stop = prep["crop_s"]
    if not (context_start <= crop_start < crop_stop <= context_stop): raise ValueError("Invalid time contract")
    if context_stop > float(dataset.interval[1] - dataset.interval[0]):
        raise ValueError("Context exceeds the adapter's task interval")
    factor = Fraction(str(prep["sfreq"])) / Fraction(str(sfreq))
    factor = factor.limit_denominator(100000)
    if abs(float(factor)*sfreq-prep["sfreq"]) > 1e-6: raise ValueError("Inexact resampling ratio")
    sos = butter(4, [prep["fmin"],prep["fmax"]], btype="bandpass", fs=sfreq, output="sos")
    n_times = round((crop_stop-crop_start)*prep["sfreq"])
    if n_times < 64: raise ValueError("Too short for configured baseline")
    fingerprint = run_fingerprint(raw)
    data=[]; labels=[]; rows=[]; excluded=[]
    bad_annotations = [(float(a["onset"])-raw.first_time, float(a["duration"]), str(a["description"]))
                       for a in raw.annotations if str(a["description"]).lower().startswith("bad")]
    for event in events:
        sample = int(event[0]); relative = sample-int(raw.first_samp); code=int(event[2])
        onset = relative/sfreq; label=LABELS[inverse[code]]
        identity={"dataset":name,"subject":str(subject),"session":str(session),"run":str(run),
                  "sample":sample,"event_code":code,"adapter_version":"moabb-1.7.2"}
        uid=digest(identity)
        start=relative+round(context_start*sfreq); stop=relative+round(context_stop*sfreq)
        if start<0 or stop>raw.n_times:
            excluded.append({**identity,"event_uid":uid,"reason":"incomplete_native_context",
                             "required_samples":[start,stop],"available_n_times":int(raw.n_times)})
            continue
        epoch=raw.get_data(picks=selected,start=start,stop=stop)
        if not np.isfinite(epoch).all(): raise ValueError("Nonfinite real EEG; no silent trial removal")
        qc=bool(np.any(np.ptp(epoch,axis=-1)>prep["qc_p2p_uv"]*1e-6))
        epoch=epoch-epoch.mean(axis=0,keepdims=True)
        epoch=sosfiltfilt(sos,epoch,axis=-1,padtype="odd")
        epoch=resample_poly(epoch,factor.numerator,factor.denominator,axis=-1,window=("kaiser",5.0))
        cut=round((crop_start-context_start)*prep["sfreq"])
        epoch=np.asarray(epoch[:,cut:cut+n_times],dtype=np.float32)
        if epoch.shape != (len(selected),n_times): raise ValueError("Unexpected output length")
        row={**identity,"event_uid":uid,"onset_s":onset,"label":label,
             "label_name":inverse[code],"session_order":session_order,"run_order":run_order,
             "native_sfreq":sfreq,"first_samp":int(raw.first_samp),"qc_p2p_flag":qc,
             "provider_bad_annotation_overlap":any(a<onset+context_stop and a+d>onset+context_start
                                                       for a,d,_ in bad_annotations),
             "signal_sha256":array_hash(epoch),"run_signal_sha256":fingerprint,
             "time_coordinate":"MOABB 1.7.2 canonical annotation in adapter Raw; not verified wall-clock"}
        data.append(epoch); labels.append(label); rows.append(row)
    audit={"session":str(session),"run":str(run),"full_eeg_channel_names":
           [c for c,t in types.items() if t=="eeg"],"selected_native_channels":selected,
           "native_sfreq":sfreq,"native_n_times":int(raw.n_times),"mne_first_samp":int(raw.first_samp),
           "mne_eeg_unit":"V by MNE Raw SI convention; upstream conversion is adapter supplied",
           "provider_bad_channels":list(raw.info["bads"]),"dataset_interval":list(dataset.interval),
           "event_mapping":event_map,"run_signal_sha256":fingerprint,"n_binary_events":len(events),
           "n_retained":len(rows),"exclusions":excluded}
    return data,labels,rows,audit

def flatten_paths(value):
    if isinstance(value,(str,Path)): return [Path(value)]
    if isinstance(value,dict): return sum((flatten_paths(v) for v in value.values()),[])
    if isinstance(value,(tuple,list)): return sum((flatten_paths(v) for v in value),[])
    return []

def prepare(root, names, subjects, config):
    import mne
    import moabb
    if mne.__version__ != "1.13.2": raise RuntimeError("Use pinned MNE 1.13.2")
    root=Path(root); raw_root=root/"raw"; raw_root.mkdir(parents=True,exist_ok=True)
    os.environ["MNE_DATA"] = str(raw_root.resolve())
    # MNE configuration is confined to this task directory, not the user's home.
    (root/"mne_config").mkdir(parents=True,exist_ok=True)
    os.environ["_MNE_FAKE_HOME_DIR"] = str((root/"mne_config").resolve())
    # An inherited dataset-specific variable must not redirect this new project
    # into a previous paper's cache. These are the pinned adapters' exact signs.
    for sign in ("BNCI","GIGADB","LEE2019-MI","EEGBCI"):
        os.environ[f"MNE_DATASETS_{sign}_PATH"] = str(raw_root.resolve())
    mne.set_config("MNE_DATA",str(raw_root.resolve()),set_env=True)
    for name in names:
        dataset=instantiate(name)
        selected=list(dataset.subject_list) if subjects is None else subjects
        if not set(selected)<=set(dataset.subject_list): raise ValueError("Unknown published participant ID")
        folder=root/"prepared"/name; folder.mkdir(parents=True,exist_ok=True)
        path=folder/"manifest.json"
        processing_implementation={"datasets_py":file_hash(__file__),
                                   "core_py":file_hash(Path(__file__).with_name("core.py")),
                                   "numpy":environment()["packages"]["numpy"],
                                   "scipy":environment()["packages"]["scipy"]}
        identity={"config":config["preprocessing"],"dataset":name,"kwargs":REGISTRY[name]["kwargs"],
                  "processing_implementation":processing_implementation,
                  "moabb_version":moabb.__version__,"mne_version":mne.__version__,
                  "adapter_file_sha256":file_hash(inspect.getfile(type(dataset)))}
        manifest=read_json(path) if path.exists() else {"dataset":name,"role":REGISTRY[name]["role"],
                  "identity":identity,"identity_sha256":digest(identity),"shards":[],
                  "biological_identity_independence_verified":False}
        if manifest["identity_sha256"] != digest(identity): raise ValueError("Existing cache has different protocol")
        for subject in selected:
            old=next((s for s in manifest["shards"] if s["subject"]==str(subject)),None)
            if old:
                if any(file_hash(folder/old[k])!=old[k+"_sha256"] for k in ("npz","json")):
                    raise ValueError("Existing cache changed; preserve and inspect")
                print(f"Verified reuse {name} subject {subject}",flush=True); continue
            print(f"Prepare {name} subject {subject}",flush=True)
            raw_data=dataset.get_data(subjects=[subject])
            sessions=raw_data[subject]
            data=[]; labels=[]; rows=[]; audits=[]
            for si,session in enumerate(sorted(sessions,key=natural_key)):
                for ri,run in enumerate(sorted(sessions[session],key=natural_key)):
                    raw=sessions[session][run]
                    if raw is None: raise ValueError("Missing provider run")
                    X,y,r,a=extract_run(raw,name,subject,session,run,si,ri,dataset,config)
                    data.extend(X); labels.extend(y); rows.extend(r); audits.append(a)
            if not rows or set(labels)!={0,1}: raise ValueError("Participant lacks qualified binary trials")
            local_files=sorted({p.resolve() for p in flatten_paths(dataset.data_path(subject)) if p.is_file()})
            sources=[{"path":str(p),"bytes":p.stat().st_size,"sha256":file_hash(p)} for p in local_files]
            header={"dataset":name,"role":REGISTRY[name]["role"],"units":"V",
                    "channels":config["preprocessing"]["channels"],"sfreq":config["preprocessing"]["sfreq"],
                    "n_times":np.shape(data)[-1],"preprocessing":config["preprocessing"],
                    "processing_implementation":processing_implementation,
                    "run_audits":audits,"source_files":sources,"environment":environment(),
                    "raw_retention":"Provider downloads stay in task raw cache; no original files deleted",
                    "chronology":"Natural adapter session/run order; not verified acquisition timestamps",
                    "provider_file_hash_verification":"Local complete SHA256; not an independent provider manifest check"}
            shard=save_shard(folder,subject,np.stack(data),np.array(labels),rows,header)
            manifest["shards"].append(shard); write_json(path,manifest)
            del raw_data, sessions, data
