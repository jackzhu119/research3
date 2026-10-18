"""Read-only planning by default; --execute explicitly starts preparation/fitting."""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from .core import (array_hash, code_identity, digest, environment, exclusive_lock, file_hash,
                   load_collection, participant_metrics, read_json, save_arrays, save_shard,
                   source_lodo, write_json)
from .datasets import REGISTRY, instantiate, prepare
from .models import fit_classical, predict_classical

DEFAULT_CONFIG=Path(__file__).resolve().parent.parent/"configs/default.json"

def doctor(root):
    report={"environment":environment(),"data_root":str(Path(root).resolve()),
            "code_sha256":code_identity(),"registered_datasets":REGISTRY,
            "actions":"Read-only check; no EEG download, fitting, or cloud operation"}
    available={}
    for name in ("mne","moabb","torch"):
        try:
            module=__import__(name);available[name]={"import":True,"version":module.__version__}
            if name=="torch": available[name]["cuda_available"]=module.cuda.is_available()
        except Exception as exc: available[name]={"import":False,"reason":str(exc)}
    report["imports"]=available
    if available["moabb"]["import"]:
        report["adapter_classes"]={}
        for name in REGISTRY:
            if REGISTRY[name]["role"] in ("source","development"):
                try:
                    ds=instantiate(name)
                    report["adapter_classes"][name]={"usable":True,"published_ids":len(ds.subject_list),
                          "event_id":ds.event_id,"interval":list(ds.interval)}
                except Exception as exc: report["adapter_classes"][name]={"usable":False,"reason":str(exc)}
    return report

def verify_complete(folder):
    record=read_json(folder/"artifact_manifest.json")
    for row in record["files"]:
        path=folder/row["path"]
        if path.stat().st_size!=row["bytes"] or file_hash(path)!=row["sha256"]:
            raise ValueError("Completed artifact changed: "+str(path))

def seal(folder):
    files=[{"path":str(p.relative_to(folder)),"bytes":p.stat().st_size,"sha256":file_hash(p)}
           for p in sorted(folder.rglob("*")) if p.is_file()
           and p.name not in ("artifact_manifest.json","status.json") and not p.name.endswith(".lock")]
    write_json(folder/"artifact_manifest.json",{"files":files,"complete_file_sha256_check":True})

def export_predictions(folder, role, indices, rows, y, probabilities, logits):
    save_arrays(folder/(role+"_predictions.npz"),indices=indices,y=y[indices],
                probabilities=probabilities,logits=logits)
    selected=[{**rows[i],"prediction_role":role} for i in indices]
    write_json(folder/(role+"_trial_metadata.json"),selected)
    write_json(folder/(role+"_metrics.json"),participant_metrics(selected,y[indices],probabilities))

def training_run(root,config,models,seeds,device,final=False,max_new_epochs=None,synthetic=False,only_fold=None):
    root=Path(root)
    X,y,rows,contract,inputs=load_collection(root/"prepared",allowed_roles=("source",))
    allowed={name for name,info in REGISTRY.items() if info["role"]=="source"}
    if not synthetic and not {r["dataset"] for r in rows}<=allowed:
        raise ValueError("Prepared input includes unapproved source dataset; holdout is never automatically opened")
    if config["preprocessing"]!=contract["preprocessing"]: raise ValueError("Training config and cache differ")
    folds=source_lodo(rows)
    if only_fold:
        if only_fold not in folds: raise ValueError("Unknown source LODO fold")
        folds={only_fold:folds[only_fold]}
    if final: folds={"ALL_SOURCE":(np.arange(len(rows)),np.array([],dtype=int))}
    inputs_digest=digest(inputs)
    for fold,(train,val) in folds.items():
        for kind in models:
            use_seeds=seeds if kind=="eegnet" else [0]
            for seed in use_seeds:
                job=root/"runs"/("SYNTHETIC_ONLY" if synthetic else "source_baselines")/fold/kind/f"seed-{seed}"
                job.mkdir(parents=True,exist_ok=True)
                neural_config=copy.deepcopy(config["neural"])
                if final and kind=="eegnet":
                    selections=[]
                    for source in sorted({r["dataset"] for r in rows}):
                        prior=job.parents[2]/source/kind/f"seed-{seed}"
                        # job.parents[2] is source_baselines, so source LODO jobs match this seed.
                        status=read_json(prior/"status.json")
                        if status["status"]!="COMPLETE": raise ValueError("Complete all LODO jobs before final fit")
                        verify_complete(prior)
                        old_identity=read_json(prior/"identity.json")
                        if (old_identity["input_sha256"]!=inputs_digest or old_identity["config"]!=config
                            or old_identity["code"]!=code_identity() or old_identity["runtime"]!=environment()):
                            raise ValueError("Final epoch selection has different inputs/config")
                        selections.append(read_json(prior/"fit_summary.json")["selected_epoch"])
                    neural_config["epochs"]=max(1,int(np.median(selections)))
                identity={"schema":"paper3-source-fit-v1","fold":fold,"model":kind,"seed":seed,
                          "input_sha256":inputs_digest,"config":config,"neural_effective":neural_config,
                          "code":code_identity(),"runtime":environment(),"synthetic_only":synthetic,
                          "train_uids":[rows[i]["event_uid"] for i in train],
                          "validation_uids":[rows[i]["event_uid"] for i in val],
                          "selection":("Source LODO only" if not final else
                              "Median completed source-LODO selected epoch, fixed before fit" if kind=="eegnet"
                              else "Fixed classical specification; fitted once on all sources")}
                identity_path=job/"identity.json"
                if identity_path.exists() and read_json(identity_path)!=identity:
                    raise ValueError("Existing job identity differs; create a new output root")
                write_json(identity_path,identity)
                status_path=job/"status.json"
                if status_path.exists() and read_json(status_path)["status"]=="COMPLETE":
                    verify_complete(job);print(f"Verified complete reuse {fold}/{kind}/{seed}",flush=True);continue
                write_json(status_path,{"status":"RUNNING","identity_sha256":digest(identity),"synthetic_only":synthetic})
                try:
                    if kind=="eegnet":
                        from .neural import train_eegnet,predict
                        model,summary=train_eegnet(job,X,y,rows,train,val,neural_config,seed,identity,
                                                  device=device,max_new_epochs=max_new_epochs)
                        write_json(job/"fit_summary.json",summary)
                        if not summary["complete"]:
                            write_json(status_path,{"status":"PAUSED_AT_EPOCH_BOUNDARY",**summary,"synthetic_only":synthetic})
                            continue
                        train_prob,train_logit=predict(model,X[train],device)
                        if len(val):val_prob,val_logit=predict(model,X[val],device)
                    else:
                        state_path=job/"classical_state.npz"; receipt=job/"classical_state_receipt.json"
                        if state_path.exists():
                            if (not receipt.exists() or file_hash(state_path)!=read_json(receipt)["sha256"]
                                or read_json(receipt)["identity_sha256"]!=digest(identity)):
                                raise ValueError("Incomplete/corrupt classical state; preserve and inspect")
                            with np.load(state_path,allow_pickle=False) as z: state=dict(z)
                        else:
                            state=fit_classical(kind,X[train],y[train],[rows[i] for i in train])
                            save_arrays(state_path,**state)
                            write_json(receipt,{"sha256":file_hash(state_path),"identity_sha256":digest(identity)})
                        train_prob,train_logit=predict_classical(kind,state,X[train])
                        if len(val):val_prob,val_logit=predict_classical(kind,state,X[val])
                        write_json(job/"fit_summary.json",{"complete":True,"seed_semantics":"Deterministic classical fit once, no replicated seeds",
                                   "tangent_reference":"log-Euclidean training mean" if kind=="ts_lr" else None,
                                   "classical_pipeline_contract":config["classical"]})
                    export_predictions(job,"train_diagnostic",train,rows,y,train_prob,train_logit)
                    if len(val):export_predictions(job,"source_validation",val,rows,y,val_prob,val_logit)
                    seal(job)
                    write_json(status_path,{"status":"COMPLETE","identity_sha256":digest(identity),
                                           "synthetic_only":synthetic,"final_holdout_evaluated":False})
                except Exception as exc:
                    write_json(status_path,{"status":"FAILED_PRESERVED","error_type":type(exc).__name__,
                                           "error":str(exc),"identity_sha256":digest(identity),"synthetic_only":synthetic})
                    raise

def synthetic_inputs(root,config):
    """No human EEG: API/workflow smoke, never a research performance result."""
    root=Path(root);rng=np.random.default_rng(20261018)
    channels=config["preprocessing"]["channels"];rate=config["preprocessing"]["sfreq"]
    n_times=round(np.diff(config["preprocessing"]["crop_s"])[0]*rate)
    for name in ("synthetic_A","synthetic_B"):
        folder=root/"prepared"/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/"manifest.json").exists():continue
        shards=[]
        for subject in range(1,4):
            rows=[];arrays=[];labels=[]
            for session in range(2):
                for run in range(2):
                    for trial in range(6):
                        label=trial%2;epoch=rng.normal(0,5e-6,(len(channels),n_times))
                        epoch[label]+=15e-6*np.sin(2*np.pi*(10+8*label)*np.arange(n_times)/rate)
                        epoch=epoch.astype(np.float32)
                        uid=f"SYNTHETIC:{name}:{subject}:{session}:{run}:{trial}"
                        rows.append({"dataset":name,"subject":str(subject),"session":str(session),"run":str(run),
                                     "event_uid":uid,"onset_s":float(trial*4),"sample":trial*4*rate,
                                     "session_order":session,"run_order":run,"label":label,
                                     "signal_sha256":array_hash(epoch),"synthetic_only":True})
                        arrays.append(epoch);labels.append(label)
            shards.append(save_shard(folder,subject,np.stack(arrays),labels,rows,
                         {"dataset":name,"role":"source","units":"V","channels":channels,"sfreq":rate,
                          "n_times":n_times,"preprocessing":config["preprocessing"],
                          "processing_implementation":{"synthetic_generator":"v1"},"synthetic_only":True}))
        write_json(folder/"manifest.json",{"dataset":name,"role":"source","shards":shards,"synthetic_only":True})

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("./paper3_data"))
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("doctor")
    prep=commands.add_parser("prep");prep.add_argument("--datasets",nargs="+",default=[k for k,v in REGISTRY.items() if v["role"]=="source"])
    prep.add_argument("--subjects",type=int,nargs="+");prep.add_argument("--execute",action="store_true")
    train=commands.add_parser("train");train.add_argument("--models",nargs="+",choices=["csp_lda","ts_lr","eegnet"],default=["csp_lda","ts_lr","eegnet"])
    train.add_argument("--seeds",type=int,nargs="+",default=[20261018,20261019,20261020])
    train.add_argument("--device",default="cpu");train.add_argument("--final",action="store_true")
    train.add_argument("--fold");train.add_argument("--max-new-epochs",type=int)
    train.add_argument("--execute",action="store_true")
    commands.add_parser("status")
    smoke=commands.add_parser("smoke");smoke.add_argument("--execute",action="store_true")
    smoke.add_argument("--with-torch",action="store_true")
    args=parser.parse_args(argv);config=read_json(args.config)
    if args.command=="doctor": print(json.dumps(doctor(args.root),indent=2,ensure_ascii=False));return
    if args.command=="status":
        result=[]
        for path in sorted((args.root/"runs").glob("**/status.json")):
            result.append({"job":str(path.parent.relative_to(args.root)),**read_json(path)})
        print(json.dumps(result,indent=2,ensure_ascii=False));return
    if not args.execute:
        print(json.dumps({"mode":"DRY_RUN_NO_DOWNLOAD_NO_FIT","command":args.command,"root":str(args.root),
                          "config":config,"requested":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                          "next_step":"Add --execute to this command after reviewing root, dataset scope and device."},indent=2))
        return
    args.root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(args.root/"paper3.lock"):
        if args.command=="prep":
            # This conservative capacity check is not a measured total-download estimate.
            if shutil.disk_usage(args.root).free < config["minimum_free_gib_prep"]*1024**3:
                raise RuntimeError("Insufficient free storage for configured preparation headroom")
            prepare(args.root,args.datasets,args.subjects,config)
        elif args.command=="train":
            if args.final and args.fold:raise ValueError("--final and --fold are mutually exclusive")
            training_run(args.root,config,args.models,args.seeds,args.device,args.final,args.max_new_epochs,only_fold=args.fold)
        elif args.command=="smoke":
            config=copy.deepcopy(config);config["neural"].update(epochs=2,patience=2,steps_per_epoch=2,batch_size=8,threads=1)
            write_json(args.root/"SYNTHETIC_ONLY_CONFIG.json",config)
            synthetic_inputs(args.root,config)
            training_run(args.root,config,["csp_lda","ts_lr"]+(["eegnet"] if args.with_torch else []),[123],"cpu",synthetic=True)
            print("SYNTHETIC WORKFLOW COMPLETE: no human EEG, no research result.")

if __name__=="__main__":main()
