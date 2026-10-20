"""Staged native eligibility only: captured TAL and marked synthetic EDFs, no fit/IO service."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import ast
import importlib.util
import json
import math
from pathlib import Path
import shutil

import mne
import numpy as np
import pytest

from p3work.contracts import CHANNELS, digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "audit/data_plan_20261020/dreyer_structure_case_review"
CANDIDATE = CASE / "native_v5_candidate/dreyer2023.py"
FROZEN = CASE / "frozen_dreyer_native_ec2f7dbc.py"
REPORT = CASE / "final_native_TAL_and_backup_review.json"
REPORT_SHA = "2a769a20353f1f478a57fd06333c6b1745938e99b820d88e417df96891d5cc37"
FROZEN_SHA = "ec2f7dbc10903c308cd30564c30b04c4653c64173fdb0331ede318fd1c3b1489"


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


BASE_TESTS=load(ROOT/"tests/test_dreyer2023_extension.py","_v5_fixture_factory")
native_fixture=BASE_TESTS.native_fixture
profile=BASE_TESTS.profile


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    # Preserve production relative preprocessing provenance without altering DSP.
    folder=tmp_path_factory.mktemp("STAGED_V5_SOFTWARE_MODULE")/"p3work/source_extensions"
    folder.mkdir(parents=True)
    path=folder/"dreyer2023.py";shutil.copyfile(CANDIDATE,path)
    shutil.copyfile(ROOT/"p3work/preprocess.py",folder.parent/"preprocess.py")
    return load(path,"_staged_v5_candidate")


@pytest.fixture(scope="module")
def captured():
    assert file_digest(REPORT)==REPORT_SHA
    report=json.loads(REPORT.read_bytes())
    assert report['status']=='ALL520_CAPTURED_NATIVE_TAL_REVIEW_COMPLETE'
    assert report['qualification_final_proof_review']['status']=='PASS_ALL520_NATIVE_PROOFS_522_FINAL_GET_AND_CLOSED_CONSERVED_LEDGER'
    rows=[]
    for proof in report['observed_record_proofs']:
        path=ROOT/proof['native']['path']
        assert file_digest(path)==proof['native']['sha256'] and path.stat().st_size==proof['native']['bytes']
        audit=json.loads(path.read_bytes())
        source=audit.get('source_file',{})
        sha=source.get('sha256',audit.get('source_file_sha256'))
        record=Path(source.get('path',audit.get('source_record_id',''))).name
        if not record:
            subject,run=proof['record_id'].split('_run-')
            record=f'{subject}_task-R{run}{"acquisition" if int(run)<=2 else "online"}_eeg.edf'
        native={'annotations':[{**r,'onset':Fraction(r['onset_token']),'duration':Fraction(r['duration_token'])}
                              for r in audit['native_annotations']],
                'n_times':audit['native_n_times'],'record_count':audit['native_record_count']}
        rows.append((proof,audit,native,sha,record))
    assert len(rows)==520
    return report,rows


def projected_bounds(rows):
    keys=('original_trial_index','original_annotation_index','raw_label_code','native_cue_onset_token',
          'native_end_onset_token','analysis_anchor_sample','task_end_sample','original_context_samples')
    return [{k:r[k] for k in keys} for r in rows]


def uids(record,sha,rows):
    subject=int(record[4:6]);run=int(record.split('_task-R')[1][0])
    run_id=f'R{run}{"acquisition" if run<=2 else "online"}'
    return [digest({'dataset':'Dreyer2023','provider_subject':str(subject),'physical_run_id':run_id,
        'source_file_sha256':sha,'original_annotation_index':r['original_annotation_index'],
        'native_cue_onset_token':r['native_cue_onset_token'],'raw_label_code':r['raw_label_code']}) for r in rows]


def test_frozen_files_and_numerical_loop_are_unchanged():
    assert file_digest(FROZEN)==FROZEN_SHA
    assert file_digest(ROOT/'p3work/source_extensions/dreyer2023.py') in (FROZEN_SHA,file_digest(CANDIDATE))
    old_tree=ast.parse(FROZEN.read_text());new_tree=ast.parse(CANDIDATE.read_text())
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    old,new=functions(old_tree),functions(new_tree)
    def trial_tail(node):
        start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='rows' for t in n.targets))
        return [ast.dump(n,include_attributes=False) for n in node.body[start:]]
    assert trial_tail(old['_trials'])==trial_tail(new['_trials'])
    for name in ('_number','_fraction','_validate_profile','_identity','_read_native_metadata'):
        assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False)
    # Materialization changes only the new native_eligibility metadata value.
    updated=deepcopy(new['materialize'])
    for n in ast.walk(updated):
        if isinstance(n,ast.Dict):
            pairs=[(k,v) for k,v in zip(n.keys,n.values) if not (isinstance(k,ast.Constant) and k.value=='native_eligibility')]
            n.keys=[k for k,v in pairs];n.values=[v for k,v in pairs]
    assert ast.dump(old['materialize'],include_attributes=False)==ast.dump(updated,include_attributes=False)


def test_all520_captured_native_membership_clock_labels_and_UIDs(candidate,captured):
    report,records=captured
    all_uids=[];labels=Counter();eligible=0;old_pass=0;reused=0
    frozen=load(FROZEN,'_immutable_v4_for_candidate')
    for proof,audit,native,sha,record in records:
        rows=candidate._trials(native,source_sha256=sha,source_record_id=record)
        assert digest(projected_bounds(rows))==proof['grammar']['original_MI_bounds_sha256']
        actual_uids=uids(record,sha,rows)
        assert actual_uids==proof['grammar']['eligible_trial_UIDs']
        assert all(r['analysis_anchor_sample']+768<=r['task_end_sample']<=native['n_times'] for r in rows)
        if proof['existing_V4_gate']['status']=='EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS':
            assert rows==frozen._trials(native);old_pass+=1
        if proof['reused_frozen_parent_complete']:
            assert actual_uids==[r['trial_uid'] for r in audit['trials']];reused+=1
            assert all(all(prior[k]==value for k,value in row.items()) for prior,row in zip(audit['trials'],rows))
        labels.update(r['label'] for r in rows);all_uids.extend(actual_uids);eligible+=len(rows)
    assert old_pass==497 and reused==128
    assert eligible==20792 and labels=={0:10394,1:10398}
    assert len(set(all_uids))==20792 and digest(all_uids)==report['native_eligible_UIDs_sha256']


def test_whitelist_is_exactly_all22_actual_missing_stop_records(candidate,captured):
    _,records=captured
    observed={}
    for proof,audit,native,sha,record in records:
        if proof['record_id']=='sub-40_run-3':continue
        if not any(e['description']=='32770' for e in native['annotations']):
            observed[(sha,record)]=tuple(proof['grammar']['suffix'])
    assert candidate.MISSING_STOP_RECORDS==observed and len(observed)==22
    assert Counter(observed.values())=={('1010','33281'):13,('1010',):1,('1010','33281','33284'):8}


def test_actual_partial_metadata_retains_only32_and_separate_original_prefix(candidate,captured):
    _,records=captured
    proof,audit,native,sha,record=next(r for r in records if r[0]['record_id']=='sub-40_run-3')
    rows=candidate._trials(native,source_sha256=sha,source_record_id=record)
    assert len(rows)==32 and Counter(r['label'] for r in rows)=={0:14,1:18}
    assert rows[-1]['original_annotation_index']==190 and rows[-1]['native_end_onset_token']=='+349'
    eligibility=candidate._native_eligibility(native,rows,source_sha256=sha,source_record_id=record)
    assert eligibility['native_original_trial_start_markers']==33
    assert eligibility['eligible_original_MI_trials']==32
    assert eligibility['eligible_original_class_counts']=={'left_hand':14,'right_hand':18}
    assert eligibility['experiment_stop_observed'] is False
    prefix=eligibility['terminal_incomplete_prefixes'][0]
    assert prefix['original_annotations']==audit['native_annotations'][193:]
    assert prefix['cue_present'] is prefix['paired800_present'] is prefix['canonical_label_available'] is False
    assert prefix['MI_trial_UID_available'] is prefix['included_in_MI_trials'] is prefix['included_in_waveform_QC_exclusions'] is False
    assert eligibility['labels_or_trials_synthesized'] is eligibility['padding_applied'] is eligibility['class_balancing_applied'] is False


def test_normal_synthetic_materialize_is_bitwise_equal_to_frozen_math(candidate,native_fixture,profile):
    old=load(FROZEN,'_frozen_v4_synthetic_math')
    # Old staged copy needs identical sibling preprocessing provenance too.
    provenance=FROZEN.parents[1]/'preprocess.py'
    # Do not mutate the immutable audit tree just to satisfy old relative layout.
    original_digest=old.file_digest
    old.file_digest=lambda path:original_digest(ROOT/'p3work/preprocess.py' if Path(path)==provenance else path)
    sha=file_digest(native_fixture)
    before=old.materialize(native_fixture,expected_sha256=sha,profile=profile)
    after=candidate.materialize(native_fixture,expected_sha256=sha,profile=profile)
    assert np.array_equal(before[0],after[0]) and np.array_equal(before[1],after[1])
    for left,right in zip(before[2],after[2]):
        assert {k:v for k,v in left.items() if k!='adapter_sha256'}=={k:v for k,v in right.items() if k!='adapter_sha256'}
    changed={'adapter_sha256','independent_native_metadata_semantic_sha256','native_eligibility'}
    assert {k:v for k,v in before[3].items() if k not in changed}=={k:v for k,v in after[3].items() if k not in changed}
    assert after[0].shape==(40,11,128) and after[3]['native_eligibility']['eligible_original_MI_trials']==40
    assert after[3]['native_eligibility']['terminal_incomplete_prefixes']==[]


@pytest.fixture(scope='module')
def partial_synthetic(tmp_path_factory,captured):
    _,records=captured
    proof,audit,native,sha,record=next(r for r in records if r[0]['record_id']=='sub-40_run-3')
    folder=tmp_path_factory.mktemp('DREYER_PARTIAL_SYNTHETIC_SOFTWARE_FIXTURE')/'sub-40/eeg';folder.mkdir(parents=True)
    path=folder/record
    records_count,annotation_samples=354,57
    names=BASE_TESTS.NAMES+['EDF Annotations'];ns=len(names);field=BASE_TESTS._field
    fixed=b''.join(field(v,w) for v,w in [('0',8),('SYNTHETIC_SOFTWARE_FIXTURE',80),('SYNTHETIC_SOFTWARE_FIXTURE',80),
        ('01.01.26',8),('00.00.00',8),(256*(ns+1),8),('EDF+C',44),(records_count,8),('1',8),(ns,4)])
    arrays=[(names,16),(['']*ns,80),(['uV']*32+[''],8),(['-100']*ns,8),(['100']*ns,8),
        (['-32768']*ns,8),(['32767']*ns,8),(['']*ns,80),([512]*32+[annotation_samples],8),(['']*ns,32)]
    header=b''.join(field(v,w) for values,w in arrays for v in values)
    adc=(12000*np.sin(2*np.pi*10*np.arange(512)[None,:]/512+np.arange(32)[:,None]*.1)).astype('<i2')
    with path.open('wb') as stream:
        stream.write(fixed+header)
        for index in range(records_count):
            stream.write(adc.tobytes());block=f'+{index}\x14\x14\x00'.encode()
            for event in audit['native_annotations']:
                if math.floor(Fraction(event['onset_token']))==index:
                    block+=f"{event['onset_token']}\x15{event['duration_token']}\x14{event['description']}\x14\x00".encode()
            assert len(block)<=annotation_samples*2
            stream.write(block.ljust(annotation_samples*2,b'\x00'))
    assert path.stat().st_size==11648932
    return path


def test_partial_synthetic_native_no_EEG_and_materialization32_contexts(candidate,partial_synthetic,profile,monkeypatch):
    # This substitution permits only this marked synthetic software fixture,
    # never a real-data hash bypass in production.
    sha=file_digest(partial_synthetic);monkeypatch.setattr(candidate,'PARTIAL_SOURCE_SHA256',sha)
    with monkeypatch.context() as guard:
        guard.setattr(mne.io.BaseRaw,'get_data',lambda *a,**k:pytest.fail('native audit read EEG'))
        native=candidate.audit_native(partial_synthetic,expected_sha256=sha,profile=profile)
    assert native['original_trials']==32 and native['EEG_array_samples_read']==0
    reads=[];original=mne.io.read_raw_edf
    def opened(*args,**kwargs):
        raw=original(*args,**kwargs);get=raw.get_data
        def bounded(*,picks,start,stop):
            assert picks==list(CHANNELS) and stop-start==768 and stop<352*512
            reads.append((start,stop));return get(picks=picks,start=start,stop=stop)
        raw.get_data=bounded;return raw
    monkeypatch.setattr(mne.io,'read_raw_edf',opened)
    X,y,events,metadata=candidate.materialize(partial_synthetic,expected_sha256=sha,profile=profile)
    assert X.shape==(32,11,128) and np.isfinite(X).all() and len(reads)==len(events)==32
    assert y.dtype==np.int64 and Counter(y.tolist())=={0:14,1:18}
    assert metadata['original_trials']==metadata['valid_trials']==32
    assert metadata['excluded_trials']==0 and metadata['exclusions']==[]
    assert metadata['native_eligibility']==native['native_eligibility']
    assert metadata['actual_fits']==metadata['GPU_actions']==metadata['network_operations']==metadata['R2_operations']==0
    assert len({r['trial_uid'] for r in events})==32
    assert all(r['original_annotation_index']<=190 for r in events)


@pytest.mark.parametrize('mutation',['raw_sha','record_name','record_parent','raw_bytes'])
def test_partial_audit_identity_and_full_bytes_gate_before_numeric_decode(candidate,partial_synthetic,profile,monkeypatch,tmp_path,mutation):
    path=partial_synthetic;sha=file_digest(path)
    monkeypatch.setattr(candidate,'PARTIAL_SOURCE_SHA256',sha)
    if mutation=='raw_sha':
        monkeypatch.setattr(candidate,'_read_native_metadata',lambda *a:pytest.fail('wrong SHA reached native parser'))
        sha='0'*64
    elif mutation=='record_name':
        path=tmp_path/'sub-41/eeg/sub-41_task-R3online_eeg.edf';path.parent.mkdir(parents=True);shutil.copyfile(partial_synthetic,path)
    elif mutation=='record_parent':
        path=tmp_path/'sub-41/eeg'/partial_synthetic.name;path.parent.mkdir(parents=True);shutil.copyfile(partial_synthetic,path)
    else:
        monkeypatch.setattr(candidate,'PARTIAL_SOURCE_BYTES',11648933)
    with pytest.raises(ValueError):candidate.audit_native(path,expected_sha256=sha,profile=profile)
