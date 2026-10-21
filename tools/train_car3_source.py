#!/usr/bin/env python3
"""Independent CAR3 source f0. Dry-run by default; never boots a GPU/server.

First future GPU invocation: --execute --user-started-gpu --engineering-pilot.
Formal source invocation requires that separate, real CAR3 resume receipt.
Neither path implements 3-G/meta training, 14 methods or external evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work import car3_source_training as training


def cuda_runtime():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG",":4096:8") not in {":4096:8",":16:8"}:
        raise training.SourceTrainingError("Deterministic_CUBLAS_configuration_required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if not torch.cuda.is_available():
        raise training.SourceTrainingError("User_started_CUDA_unavailable_no_server_will_be_booted")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    p=torch.cuda.get_device_properties(0)
    try:
        driver=subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True,timeout=10,check=True).stdout.strip().splitlines()
    except (FileNotFoundError,subprocess.SubprocessError):driver=None
    return {"device":"cuda:0","device_name":p.name,"device_uuid":str(getattr(p,"uuid","not_exposed")),
        "device_total_memory":p.total_memory,"compute_capability":[p.major,p.minor],"driver_versions":driver,
        "torch":str(torch.__version__),"numpy":np.__version__,"python":sys.version,
        "cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"AMP":False,"TF32":False,
        "deterministic_algorithms":True,"cuda_rng_device_count":torch.cuda.device_count()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--profile",choices=("development","confirmation"),default="development")
    parser.add_argument("--output")
    parser.add_argument("--pilot-receipt",default="runs/car3_source_engineering_v1/engineering_receipt.json")
    parser.add_argument("--engineering-pilot",action="store_true")
    parser.add_argument("--extend-confirmation-plan",action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--max-new-updates",type=int)
    parser.add_argument("--max-wall-seconds",type=int)
    parser.add_argument("--amend-operational-budget-seconds",type=int)
    parser.add_argument("--operational-amendment-note")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--user-started-gpu",action="store_true")
    args=parser.parse_args(argv);entered=False
    try:
        if args.execute and not args.user_started_gpu:
            raise training.SourceTrainingError("Both_execute_and_user_started_gpu_flags_required")
        if args.engineering_pilot and (args.job or args.max_new_updates or args.extend_confirmation_plan):
            raise training.SourceTrainingError("CAR3_engineering_pilot_has_own_fixed_branch_and_update_budget")
        cache=training.verify_inputs(args.root)
        plan=training.build_training_plan(cache,profile=args.profile)
        if not args.execute:
            result={"status":"CAR3_SOURCE_F0_DRY_RUN","plan_sha256":plan["plan_sha256"],
                "profile":args.profile,"architecture":training.ARCHITECTURE,
                "source_provider_ids":224,"same_original_source_UIDs":len(cache.rows),
                "planned_neural_fits":plan["planned_neural_fits"],"max_committed_updates":plan["max_committed_updates"],
                "normalization":"float64_joint_RMS_then_float32","actual_fits_started":0,"GPU_checked":False,
                "CAR3_GPU_resume_pilot_verified":False,"engineering_pilot_required":True,
                "independent_3G_and_all14_methods_ready":False,"Rref_shape":[2,2],
                "server_started":False,"external_payloads_accessed":False}
        else:
            if not args.engineering_pilot:
                training.verify_engineering_pilot(cache,args.pilot_receipt)
            runtime=cuda_runtime();entered=True
            if args.engineering_pilot:
                result=training.engineering_pilot(cache,args.output or "runs/car3_source_engineering_v1",runtime=runtime,
                    max_wall_seconds=1800 if args.max_wall_seconds is None else args.max_wall_seconds,
                    amend_budget_seconds=args.amend_operational_budget_seconds,amendment_note=args.operational_amendment_note)
            else:
                result=training.execute_plan(cache,plan,args.output or "runs/car3_source_f0_v1",runtime=runtime,
                    pilot_path=args.pilot_receipt,max_new_updates=args.max_new_updates,
                    max_wall_seconds=86400 if args.max_wall_seconds is None else args.max_wall_seconds,only_job=args.job,
                    extend_confirmation=args.extend_confirmation_plan,
                    amend_budget_seconds=args.amend_operational_budget_seconds,amendment_note=args.operational_amendment_note)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result.get("status")=="CAR3_SOURCE_F0_DRY_RUN" or result.get("status")=="CAR3_SOURCE_F0_COMPLETE" or result.get("exact_resume_verified") is True else 3
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,FloatingPointError) as error:
        print(json.dumps({"status":"BLOCKED","reason":str(error),"execution_entered":entered,
            "actual_fits_started":None if entered else 0,"server_started":False,"external_payloads_accessed":False},indent=2))
        return 2


if __name__=="__main__":raise SystemExit(main())
