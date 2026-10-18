"""Synthetic W3 staging tests: no real download, credentials, or training."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
from botocore.exceptions import ClientError

from p3work.contracts import validate_signal_contract
from p3work.splits import freeze_subject_split
from tools import prepare_stieger_cache as cache
from tools import r2_backup, r2_restore


RAW = b"synthetic native raw object, never a study EEG result"


class FakeR2:
    def __init__(self):
        self.data = {}
        self.puts = []

    def put_object(self, **kw):
        key=kw["Key"]
        self.puts.append(key)
        if key in self.data:
            raise ClientError({"Error":{"Code":"PreconditionFailed"},"ResponseMetadata":{"HTTPStatusCode":412}},"PutObject")
        self.data[key]=kw["Body"].read()
        return {"ResponseMetadata":{"HTTPStatusCode":200}}

    def get_object(self, **kw):
        data=self.data[kw["Key"]]
        return {"ResponseMetadata":{"HTTPStatusCode":200},"ContentLength":len(data),"Body":io.BytesIO(data)}

    def backup(self, **kw):
        return r2_backup.run_backup(**kw,client=self,bucket="fixture-bucket")

    def restore(self, *, entry, root):
        return r2_restore.restore_one(self,"fixture-bucket",entry,root)


class FakeSession:
    def __init__(self,path,expected_sha256):
        self.path=path;self.source_sha256=expected_sha256
        self.participant_id="S1";self.session_id="1";self.native_sfreq=1000.
        self.calls=[];self.rows=[]
        for i,(label,length) in enumerate([(0,3.),(1,1.),(0,3.),(1,3.)]):
            self.rows.append({"dataset":"Stieger2021","dataset_version":1,"participant_id":"S1","session_id":"1",
                "source_record_id":"S1_Session_1.mat","source_file_sha256":expected_sha256,"original_trial_index":i,
                "provider_trialnumber":i+1,"physical_run_id":1,"physical_run_complete_verified":False,
                "tasknumber":1,"label_code":2 if label==0 else 1,"label":label,"trial_uid":f"synthetic-fixture:{expected_sha256}:{i}",
                "analysis_anchor_sample":4000,"feedback_stop_exclusive_sample":4001+int(length*1000),
                "feedback_end_verified":True,"available_feedback_seconds":length,
                "native_trial_samples_including_padding":5001+int(length*1000),"provider_artifact":0})

    def pure_lr_metadata(self):return [dict(r) for r in self.rows]

    def context_volts(self,i,seconds):
        self.calls.append(i)
        if i==2:raise ValueError("Nonfinite signal")
        t=np.arange(round(seconds*1000))/1000
        return np.array([(j+1)*1e-6*np.sin(2*np.pi*(11+i)*t+j*.1) for j in range(11)])

    def schema_audit(self):
        return {"participant_id":"S1","session_id":"1","source_file_sha256":self.source_sha256,
                "pure_lr_trials":4,"endpoint_errors":0,"mbsr_subject_raw":1}


class StiegerCacheTest(unittest.TestCase):
    def setup_fixture(self,root):
        contract={"status":"window_frozen","channels":list(cache.CLINIC11),"sfreq_hz":128,"selected_profile":"long",
                  "context_s":[0,3],"crop_s":[.5,2.5],"n_times":256,
                  "window_audit_receipt":{"decision_pool_role":"meta_train","split_sha256":"a"*64,
                    "raw_metadata_manifest_sha256":"b"*64,"long":{"participant_mean_coverage":.9,"participant_mean_abs_class_exclusion_gap":0.},
                    "short":{"participant_mean_coverage":.95,"participant_mean_abs_class_exclusion_gap":0.}}}
        entry={"id":123,"name":"S1_Session_1.mat","provider_subject":"S1","session_ordinal":1,"partition":"meta_train",
               "size":len(RAW),"computed_md5":hashlib.md5(RAW).hexdigest(),"download_url":"https://ndownloader.figshare.com/files/123"}
        for rel in ["p3work/adapters/stieger_native.py","p3work/preprocess.py"]:
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("synthetic code fixture\n")
        plan={"plan_sha256":"synthetic-fixture-plan","input_contract_sha256":validate_signal_contract(contract),"split_sha256":"a"*64,
              "adapter_code":cache.descriptor(root,root/"p3work/adapters/stieger_native.py"),
              "processing_code":cache.descriptor(root,root/"p3work/preprocess.py")}
        return contract,entry,plan

    @staticmethod
    def fake_download(entry,partial,final):
        final.parent.mkdir(parents=True,exist_ok=True);final.write_bytes(RAW)
        return {"sha256":hashlib.sha256(RAW).hexdigest(),"bytes":len(RAW),"md5":hashlib.md5(RAW).hexdigest()}

    def test_bounded_original_context_and_invalid_trials_remain_in_events(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root)
            session=FakeSession(root/entry["name"],hashlib.sha256(RAW).hexdigest())
            x,events,meta=cache.materialize_session(session,entry,contract,plan)
            self.assertEqual(x.shape,(2,11,256));self.assertEqual(x.dtype,np.float32)
            self.assertEqual(session.calls,[0,2,3])
            self.assertEqual(len(events),4)
            self.assertEqual([r["cache_index"] for r in events],[0,None,None,1])
            self.assertEqual(events[1]["exclusion_reason"],"insufficient_or_unverified_real_feedback")
            self.assertEqual(events[2]["exclusion_reason"],"nonfinite_context")
            self.assertGreater(events[1]["context_stop_sample"],events[1]["feedback_stop_exclusive_sample"])
            self.assertEqual(meta["original_lr_trials"],4)
            for row in (events[0],events[3]):
                self.assertEqual(row["context_start_sample"],4000)
                self.assertEqual(row["context_stop_sample"],7000)
                self.assertFalse(row["preprocessing_qc"]["normalization_applied"])
                self.assertFalse(row["preprocessing_qc"]["data_after_context_accessed"])
                self.assertEqual(row["preprocessing_qc"]["filter_context_ends_at_native_sample"],7000)
            np.testing.assert_allclose(x.mean(axis=-1),0,atol=1e-13)

    def test_new_owned_raw_removed_after_both_full_gets_and_resume_no_io(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2();output=root/"cache/stieger"
            with patch.object(cache.base,"check_space"):
                state=cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                       backup_fn=client.backup,load_fn=FakeSession,download_fn=self.fake_download)
                self.assertEqual(state["status"],"COMPLETE")
                self.assertFalse((root/state["raw_path"]).exists())
                self.assertTrue(cache.verify_session_state(state,root,plan))
                replay=cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                       backup_fn=lambda **kw:self.fail("must not repeat completed PUT/GET"),load_fn=lambda *a,**kw:self.fail("must not parse completed session"),
                       download_fn=lambda *a:self.fail("must not download completed original"))
                self.assertEqual(replay,state)

    def test_cache_backup_failure_preserves_raw_then_resumes_without_rewriting_raw_proof(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2();output=root/"cache/stieger";calls=[]
            def fail_cache_backup(**kw):
                calls.append(kw["prefix"])
                if "stieger-cache-v2" in kw["prefix"]:return {"status":"failed","execute":True}
                return client.backup(**kw)
            with patch.object(cache.base,"check_space"):
                with self.assertRaisesRegex(cache.CacheError,"r2_backup_full_get"):
                    cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                         backup_fn=fail_cache_backup,load_fn=FakeSession,download_fn=self.fake_download)
                state_path=output/"meta_train/S1/session-1/state.json";failed=json.loads(state_path.read_text())
                self.assertTrue((root/failed["raw_path"]).exists())
                original_proof=copy.deepcopy(failed["raw_backup_receipt"])
                state=cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                         backup_fn=client.backup,load_fn=FakeSession,download_fn=lambda *a:self.fail("raw already retained"))
            self.assertEqual(state["status"],"COMPLETE")
            self.assertEqual(state["raw_backup_receipt"],original_proof)
            self.assertFalse((root/state["raw_path"]).exists())

    def test_existing_schema_sample_preserved_and_verified_r2_raw_reused_for_new_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2();sample=root/"data/stieger2021/raw"/entry["name"]
            sample.parent.mkdir(parents=True);sample.write_bytes(RAW)
            with patch.object(cache.base,"check_space"):
                first=cache.process_session(entry,root=root,output=root/"cache/first",contract=contract,plan=plan,raw_catalog={},
                     backup_fn=client.backup,load_fn=FakeSession,download_fn=lambda *a:self.fail("sample exists"))
                self.assertEqual(sample.read_bytes(),RAW)
                raw_rows,_=r2_restore.load_verified_receipt(root/first["raw_backup_receipt"]["path"])
                raw=next(r for r in raw_rows if r["role"]=="source_development_raw")
                # Move an existing specimen out of the lookup, rather than
                # authorizing the worker to delete it.
                sample.rename(sample.with_suffix(".preserved"))
                next_plan=dict(plan,plan_sha256="synthetic-next-plan")
                second=cache.process_session(entry,root=root,output=root/"cache/second",contract=contract,plan=next_plan,
                     raw_catalog={123:{"entry":raw,"backup_receipt":first["raw_backup_receipt"]}},
                     backup_fn=client.backup,load_fn=FakeSession,restore_fn=client.restore,
                     download_fn=lambda *a:self.fail("verified R2 original must be restored"))
            self.assertEqual(second["status"],"COMPLETE")
            self.assertTrue(second.get("raw_restore_receipt"))
            self.assertTrue(sample.with_suffix(".preserved").exists())
            self.assertFalse((root/second["raw_path"]).exists())
            # Only the first stage uploads the original; second stage archives
            # its independently bound cache, and restores the original by GET.
            raw_puts=[k for k in client.puts if k.endswith(entry["name"])]
            self.assertEqual(len(raw_puts),1)

    def test_completed_cache_corruption_fails_without_downloading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2();output=root/"cache/stieger"
            with patch.object(cache.base,"check_space"):
                state=cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                      backup_fn=client.backup,load_fn=FakeSession,download_fn=self.fake_download)
            (root/state["session_record"]["events"]["path"]).write_text("[]")
            with self.assertRaisesRegex(cache.CacheError,"artifact_sha256"):
                cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                      download_fn=lambda *a:self.fail("corrupt completed cache must not trigger fresh download"))

    def test_w1_gate_blocks_before_touching_other_inputs_or_network(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"status").mkdir();(root/"status/tasks.json").write_text(json.dumps({"W1":{"status":"IN_PROGRESS"}}))
            with self.assertRaisesRegex(cache.CacheError,"w1_complete"):
                cache.read_frozen_inputs(root,root/"nonexistent-contract",root/"nonexistent-split",execute=True)

    def test_all598_plan_preserves_exact_frozen_roles_and_resource_bounds(self):
        root=cache.REPO;inventory=root/"data_catalog/stieger2021/official_inventory.json"
        if not inventory.exists():self.skipTest("Private local catalog unavailable; no network fallback")
        with tempfile.TemporaryDirectory() as d:
            contract,_,_=self.setup_fixture(Path(d))
        split=freeze_subject_split([{"participant_id":f"S{i}","identity_verified":True,"clinic11_present_verified":True,
                                     "pure_lr_present_verified":True,"intervention_group":"synthetic-fixture"} for i in range(1,63)])
        with patch.object(cache,"read_frozen_inputs",return_value=(contract,split,validate_signal_contract(contract))):
            plan=cache.build_plan(root,inventory_path=inventory,contract_path=root/"contracts/input_contract.json",
                                  split_path=root/"contracts/input_contract.json")
        self.assertEqual(plan["file_count"],598)
        self.assertEqual(plan["max_download_bytes"],376894087292)
        self.assertEqual(plan["max_r2_restore_bytes"],376894087292)
        self.assertEqual(plan["max_jobs"],598);self.assertEqual(plan["max_gpu_hours"],0)
        for role,people in split["partitions"].items():
            self.assertEqual({f["provider_subject"] for f in plan["files"] if f["partition"]==role},set(people))

    def test_unmarked_existing_temporary_raw_is_never_adopted_for_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2()
            raw=root/"data/stieger-cache-raw"/plan["plan_sha256"] /str(entry["id"])/entry["name"]
            raw.parent.mkdir(parents=True);raw.write_bytes(RAW)
            with patch.object(cache.base,"check_space"):
                state=cache.process_session(entry,root=root,output=root/"cache/stieger",contract=contract,plan=plan,raw_catalog={},
                        backup_fn=client.backup,load_fn=FakeSession,download_fn=lambda *a:self.fail("raw already exists"))
            self.assertFalse(state["raw_owned"])
            self.assertEqual(raw.read_bytes(),RAW)

    def test_interruption_after_verified_raw_removal_resumes_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2();output=root/"cache/stieger"
            original_cleanup=cache.cleanup_owned_raw
            def interrupt(state,**kw):
                original_cleanup(state,**kw)
                raise cache.CacheError("synthetic_interruption_after_raw_removal")
            with patch.object(cache.base,"check_space"),patch.object(cache,"cleanup_owned_raw",side_effect=interrupt):
                with self.assertRaisesRegex(cache.CacheError,"synthetic_interruption"):
                    cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                         backup_fn=client.backup,load_fn=FakeSession,download_fn=self.fake_download)
            state=cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                         backup_fn=lambda **kw:self.fail("proof already committed"),
                         download_fn=lambda *a:self.fail("verified original must not be downloaded twice"))
            self.assertEqual(state["status"],"COMPLETE")

    def test_corrupted_full_get_cannot_trigger_owned_raw_deletion(self):
        class CorruptCacheGet(FakeR2):
            def get_object(self,**kw):
                response=super().get_object(**kw)
                if kw["Key"].endswith("/X.npy"):
                    data=self.data[kw["Key"]]
                    response["Body"]=io.BytesIO(bytes([data[0]^1])+data[1:])
                return response
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=CorruptCacheGet();output=root/"cache/stieger"
            with patch.object(cache.base,"check_space"):
                with self.assertRaisesRegex(cache.CacheError,"r2_backup_full_get"):
                    cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                        backup_fn=client.backup,load_fn=FakeSession,download_fn=self.fake_download)
            failed=json.loads((output/"meta_train/S1/session-1/state.json").read_text())
            self.assertEqual(failed["status"],"FAILED_CLOSED")
            self.assertEqual((root/failed["raw_path"]).read_bytes(),RAW)

    def test_official_byte_budget_and_interrupted_partial_are_accounted_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root);client=FakeR2();output=root/"cache/stieger"
            with patch.object(cache.base,"check_space"):
                with self.assertRaisesRegex(cache.CacheError,"official_provider_download_budget"):
                    cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                        remaining_source_bytes=1,download_fn=lambda *a:self.fail("budget rejected before network"))
                def interrupted(entry,partial,final):
                    partial.write_bytes(RAW[:5])
                    raise TimeoutError("sensitive text must be suppressed")
                with self.assertRaisesRegex(cache.CacheError,"operation_failed_details_suppressed"):
                    cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                        remaining_source_bytes=len(RAW),download_fn=interrupted)
                state_path=output/"meta_train/S1/session-1/state.json"
                failed=json.loads(state_path.read_text())
                self.assertEqual(failed["official_provider_body_bytes"],len(RAW))
                self.assertNotIn("sensitive text",state_path.read_text())
                def resume(entry,partial,final):
                    self.assertEqual(partial.read_bytes(),RAW[:5])
                    info=self.fake_download(entry,partial,final)
                    partial.unlink();info["resumed_from_bytes"]=5
                    return info
                done=cache.process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog={},
                    remaining_source_bytes=len(RAW)-5,backup_fn=client.backup,load_fn=FakeSession,download_fn=resume)
            self.assertEqual(done["official_provider_body_bytes"],2*len(RAW)-5)
            self.assertEqual(done["official_provider_verified_body_bytes"],len(RAW)-5)

    def test_hard_kill_transfer_intents_consume_full_reservations_before_retry(self):
        for kind in ("official_get","r2_restore"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);contract,entry,plan=self.setup_fixture(root)
                obj=root/"cache/meta_train/S1/session-1";obj.mkdir(parents=True)
                raw=root/"data/stieger-cache-raw"/plan["plan_sha256"]/str(entry["id"])/entry["name"]
                raw.parent.mkdir(parents=True)
                state={"status":"RAW_VERIFY_OR_FETCH","plan_sha256":plan["plan_sha256"],"file_id":entry["id"],"name":entry["name"],
                       "raw_path":raw.relative_to(root).as_posix(),"raw_owned":True}
                cache.base.atomic_json(obj/"raw_ownership.json",{**{k:state[k] for k in ("plan_sha256","file_id","raw_path")},
                                                                "created_by":"tools/prepare_stieger_cache.py"})
                sp=obj/"state.json"
                cache.begin_transfer(state,kind=kind,reserve=len(RAW),available=len(RAW),root=root,object_dir=obj,plan=plan,state_path=sp)
                # SIGKILL equivalent: only durable intent survives; the worker
                # has no finally/report opportunity after half a response.
                partial=raw.with_suffix(".mat.partial") if kind=="official_get" else raw.with_name(raw.name+".random-interrupted.partial")
                partial.write_bytes(RAW[:26])
                state=json.loads(sp.read_text())
                cache.recover_open_transfer(state,root=root,object_dir=obj,plan=plan,state_path=sp)
                counter="official_provider_body_bytes" if kind=="official_get" else "r2_restore_body_bytes"
                self.assertEqual(state[counter],len(RAW))
                self.assertEqual(state["transfer_history"][-1]["status"],"HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED")
                retry=len(RAW)-26 if kind=="official_get" else len(RAW)
                with self.assertRaisesRegex(cache.CacheError,"budget_exhausted"):
                    cache.begin_transfer(state,kind=kind,reserve=retry,available=retry-1,root=root,object_dir=obj,plan=plan,state_path=sp)
                cache.begin_transfer(state,kind=kind,reserve=retry,available=retry,root=root,object_dir=obj,plan=plan,state_path=sp)
                cache.finish_transfer(state,actual=retry,root=root,object_dir=obj,plan=plan,state_path=sp)
                self.assertEqual(state[counter],len(RAW)+retry)
                self.assertTrue(partial.exists())  # No unverified partial deleted.

    def test_unknown_partial_ownership_cannot_reopen_transfer_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,entry,plan=self.setup_fixture(root);obj=root/"cache/session";obj.mkdir(parents=True)
            raw=root/"data/stieger-cache-raw"/plan["plan_sha256"]/str(entry["id"])/entry["name"]
            raw.parent.mkdir(parents=True)
            state={"status":"RAW_VERIFY_OR_FETCH","plan_sha256":plan["plan_sha256"],"file_id":entry["id"],"name":entry["name"],
                   "raw_path":raw.relative_to(root).as_posix(),"raw_owned":True}
            marker={**{k:state[k] for k in ("plan_sha256","file_id","raw_path")},"created_by":"tools/prepare_stieger_cache.py"}
            cache.base.atomic_json(obj/"raw_ownership.json",marker)
            cache.begin_transfer(state,kind="r2_restore",reserve=len(RAW),available=len(RAW),root=root,object_dir=obj,plan=plan,state_path=obj/"state.json")
            partial=raw.with_name(raw.name+".random.partial");partial.write_bytes(RAW[:26])
            (obj/"raw_ownership.json").unlink()
            recovered=json.loads((obj/"state.json").read_text())
            with self.assertRaisesRegex(cache.CacheError,"ownership_unverified"):
                cache.recover_open_transfer(recovered,root=root,object_dir=obj,plan=plan,state_path=obj/"state.json")
            self.assertEqual(recovered["r2_restore_body_bytes"],len(RAW));self.assertTrue(partial.exists())

    def test_hard_kill_wall_interval_is_charged_and_clock_changes_close_budget(self):
        boot="0"*36;start={"utc":"2026-10-18T00:00:00+00:00","monotonic_ns":1000000000,"boot_id":boot}
        now={"utc":"2026-10-18T00:00:37+00:00","monotonic_ns":38000000000,"boot_id":boot}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"progress.json";progress={"active_wall_elapsed_s":100.,"wall_intent":start}
            cache.base.atomic_json(path,progress)
            progress=json.loads(path.read_text())
            with patch.object(cache,"clock_snapshot",return_value=now):cache.wall_checkpoint(path,progress,reopen=False)
            self.assertEqual(progress["active_wall_elapsed_s"],137.)
            self.assertNotIn("wall_intent",progress)
            for changed in (dict(now,utc="2026-10-17T23:59:59+00:00"),dict(now,boot_id="1"*36),dict(now,monotonic_ns=1)):
                progress={"active_wall_elapsed_s":100.,"wall_intent":start}
                with patch.object(cache,"clock_snapshot",return_value=changed):
                    with self.assertRaisesRegex(cache.CacheError,"clock_changed_budget_closed"):
                        cache.wall_checkpoint(path,progress,reopen=True)
                self.assertEqual(progress["status"],"UNKNOWN_ELAPSED_TIME_BUDGET_CLOSED")
                self.assertIn("wall_intent",progress)

    def test_recovered_wall_exhaustion_stops_before_catalog_or_any_transfer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract,entry,plan=self.setup_fixture(root)
            plan.update(output="cache/stieger",files=[entry],max_wall_seconds=60,max_download_bytes=len(RAW),
                        max_r2_restore_bytes=len(RAW),max_cache_bytes=100000)
            (root/"status").mkdir();(root/"status/tasks.json").write_text(json.dumps({"W1":{"status":"COMPLETE"}}))
            output=root/plan["output"];output.mkdir(parents=True)
            cache.base.atomic_json(output/"ownership.json",{"schema_version":cache.SCHEMA,"created_by":"tools/prepare_stieger_cache.py","plan_sha256":plan["plan_sha256"]})
            start={"utc":"2026-10-18T00:00:00+00:00","monotonic_ns":1000000000,"boot_id":"0"*36}
            now={"utc":"2026-10-18T00:00:20+00:00","monotonic_ns":21000000000,"boot_id":"0"*36}
            cache.base.atomic_json(output/"progress.json",{"plan_sha256":plan["plan_sha256"],"active_wall_elapsed_s":50.,"wall_intent":start})
            with patch.object(cache,"validate_plan",return_value=(contract,{})),patch.object(cache,"clock_snapshot",return_value=now),\
                 patch.object(cache,"raw_receipt_catalog",side_effect=AssertionError("budget must stop first")):
                result=cache.prepare(root,plan,execute=True)
            self.assertEqual(result["status"],"RESOURCE_BUDGET_STOP_NO_W3_COMPLETE")
            self.assertEqual(result["active_wall_elapsed_s"],70.)

    def test_execute_error_does_not_falsely_report_that_no_data_execution_occurred(self):
        with tempfile.TemporaryDirectory() as d:
            plan_path=Path(d)/"plan.json";plan_path.write_text("{}")
            out=io.StringIO()
            with patch.object(cache.sys,"argv",["prepare_stieger_cache.py","--plan",str(plan_path),"--execute"]),\
                 patch.object(cache,"prepare",side_effect=cache.CacheError("synthetic_stop_after_checkpoint")),redirect_stdout(out):
                self.assertEqual(cache.main(),1)
            result=json.loads(out.getvalue())
            self.assertEqual(result["status"],"EXECUTION_STOPPED_NOT_COMPLETE")
            self.assertTrue(result["execution_requested"])
            self.assertFalse(result["training_started"])

    def test_cross_split_identical_signal_closes_manifest_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/"cache";output.mkdir();states=[]
            for i,partition in enumerate(("meta_train","meta_validation")):
                p=output/f"events{i}.json";p.write_text(json.dumps([{"event_uid":f"fixture-{i}","valid":True,"signal_sha256":"same-synthetic-signal"}]))
                states.append({"file_id":i,"session_record":{"partition":partition,"events":cache.descriptor(root,p)}})
            states.extend({"file_id":i} for i in range(2,598))
            plan={"files":[{"id":i} for i in range(598)]}
            with patch.object(cache,"verify_session_state",return_value=True):
                with self.assertRaisesRegex(cache.CacheError,"crosses_frozen_person_split"):
                    cache.assemble_manifest(root,output,plan,states)
            self.assertEqual(json.loads((output/"duplicate_audit.json").read_text())["status"],"SPLIT_INVALID")


if __name__=="__main__":unittest.main()
