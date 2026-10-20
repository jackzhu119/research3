"""Synthetic EDF and permission tests; no download, real fit or GPU operation."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil

import mne
import numpy as np
import pytest

from p3work.contracts import CHANNELS, digest, file_digest
from p3work.source_extensions import dreyer2023 as adapter


NAMES = "Fz FCz Cz CPz Pz C1 C3 C5 C2 C4 C6 EOG1 EOG2 EOG3 EMGg EMGd F4 FC2 FC4 FC6 CP2 CP4 CP6 P4 F3 FC1 FC3 FC5 CP1 CP3 CP5 P3".split()


def _field(value, width):
    encoded = str(value).encode("ascii")
    assert len(encoded) <= width
    return encoded.ljust(width,b" ")


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("DREYER_SYNTHETIC_SOFTWARE_FIXTURE") / "sub-01" / "eeg"
    directory.mkdir(parents=True)
    path = directory / "sub-01_task-R1acquisition_eeg.edf"
    records, annotation_samples = 405, 256
    names = NAMES + ["EDF Annotations"]
    ns = len(names)
    fixed = b"".join(_field(v,w) for v,w in [
        ("0",8),("SYNTHETIC_SOFTWARE_FIXTURE",80),("SYNTHETIC_SOFTWARE_FIXTURE",80),
        ("01.01.26",8),("00.00.00",8),(256*(ns+1),8),("EDF+C",44),(records,8),("1",8),(ns,4)])
    arrays = [(names,16),([""]*ns,80),(["uV"]*32+[""],8),(["-100"]*ns,8),(["100"]*ns,8),
        (["-32768"]*ns,8),(["32767"]*ns,8),([""]*ns,80),([512]*32+[annotation_samples],8),([""]*ns,32)]
    header = b"".join(_field(v,w) for values,w in arrays for v in values)
    entries = [("0","0.0019","32769")]
    for index in range(40):
        trial = Fraction(1+10*index) + Fraction(3,10000)
        def token(value):
            return f"{float(value):.4f}"
        entries += [(token(trial),"0.0019","768"),(token(trial),"0.0019","786"),
            (token(trial+2),"0.0019","33282"),(token(trial+3),"0.0019","769" if index%2 == 0 else "770"),
            (token(trial+Fraction(17,4)),"0.0019","781"),(token(trial+8),"0.0019","800")]
    entries.append(("404.9000","0","32770"))
    adc = (12000*np.sin(2*np.pi*10*np.arange(512)[None,:]/512 + np.arange(32)[:,None]*.1)).astype("<i2")
    with path.open("wb") as stream:
        stream.write(fixed+header)
        for record in range(records):
            stream.write(adc.tobytes())
            block = f"+{record}\x14\x14\x00".encode()
            for onset,duration,code in entries:
                if math.floor(Fraction(onset)) == record:
                    block += f"+{onset}\x15{duration}\x14{code}\x14\x00".encode()
            assert len(block) <= annotation_samples*2
            stream.write(block.ljust(annotation_samples*2,b"\x00"))
    return path


@pytest.fixture
def profile():
    contract = json.loads((Path(__file__).parents[1]/"contracts/input_contract.json").read_text())
    return {"schema_version":"paper3-dreyer2023-native-source-profile-v1", "profile_id":adapter.PROFILE_ID,
        "role":"exposed_source_extension", "dataset_id":"Dreyer2023", "source_version":adapter.SOURCE_VERSION,
        "native_anchor":"original_769_770_cue", "native_end":"paired_800", "sampling_policy":"ceil_onset_floor_end",
        "native_sfreq_hz":512, "native_units":["uV","µV"], "raw_label_to_y":{"769":0,"770":1},
        "raw_label_to_hand":{"769":"left_hand","770":"right_hand"},
        "signal_contract_semantic_sha256":adapter.CONTRACT_SHA256, "signal_contract":contract,
        "external_query_access":False,"training_started":False}


def test_native_audit_reads_no_EEG_and_preserves_original_fractional_clock(native_fixture, profile, monkeypatch):
    monkeypatch.setattr(mne.io.BaseRaw,"get_data",lambda *a,**k:pytest.fail("native audit read EEG"))
    result = adapter.audit_native(native_fixture, expected_sha256=file_digest(native_fixture), profile=profile)
    assert result["EEG_array_samples_read"] == 0
    assert result["original_trials"] == 40 and result["waveform_QC_applied"] is False
    rows = result["trials"]
    assert [r["label"] for r in rows] == [0,1]*20
    assert [r["original_trial_index"] for r in rows] == list(range(1,41))
    assert [r["original_annotation_index"] for r in rows] == list(range(4,240,6))
    assert len({r["trial_uid"] for r in rows}) == 40
    first = rows[0]
    assert first["native_cue_onset_token"] == "+4.0003"
    assert first["analysis_anchor_sample"] == math.ceil(Fraction("4.0003")*512)
    assert first["task_end_sample"] == math.floor(Fraction("9.0003")*512)
    assert first["original_context_samples"] == [2049,2817]
    assert 0 <= first["inward_start_delay_seconds"] < 1/512
    assert first["native_cue_impulse_duration_token"] == "0.0019"
    assert first["task_end_sample"]-first["analysis_anchor_sample"] > 2500
    assert result["native_marker_counts"]["800"] == 40


def test_documented_terminal_EndOfFile_retains_native_index_without_changing_MI_trials(tmp_path,native_fixture,profile):
    path=tmp_path/"sub-01"/"eeg"/native_fixture.name
    path.parent.mkdir(parents=True);shutil.copyfile(native_fixture,path)
    annotation_bytes=512
    offset=256*34+404*2*(32*512+256)+2*32*512
    with path.open("r+b")as stream:
        stream.seek(offset);block=stream.read(annotation_bytes)
        block=block.rstrip(b"\x00")+b"\x00+404.9500\x150\x1433284\x14\x00"
        assert len(block)<annotation_bytes
        stream.seek(offset);stream.write(block.ljust(annotation_bytes,b"\x00"))
    old=adapter.audit_native(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)
    new=adapter.audit_native(path,expected_sha256=file_digest(path),profile=profile)
    assert new["native_marker_counts"]["33284"]==1 and new["terminal_eof_annotation_indices"]==[242]
    assert len(new["native_annotations"])==243 and new["original_trials"]==40
    for a,b in zip(old["trials"],new["trials"]):
        for key in ["original_annotation_index","raw_label_code","analysis_anchor_sample","task_end_sample",
                    "original_context_samples","native_cue_onset_token"]:
            assert a[key]==b[key]
    X,y,events,metadata=adapter.materialize(path,expected_sha256=file_digest(path),profile=profile)
    assert X.shape==(40,11,128) and len(events)==40 and y.tolist()==[0,1]*20
    assert metadata["original_trials"]==40


@pytest.mark.parametrize("bad",["mid_run","duplicate","nonzero_duration","before_stop","outside_file"])
def test_EndOfFile_is_never_silently_ignored_or_used_as_trial_boundary(native_fixture,bad):
    native=adapter._read_native_metadata(native_fixture)
    eof=dict(native["annotations"][-1],description="33284",onset=Fraction("404.95"),duration=Fraction(0),
             original_annotation_index=242)
    if bad=="mid_run":native["annotations"].insert(40,eof)
    else:
        native["annotations"].append(eof)
        if bad=="duplicate":native["annotations"].append(dict(eof))
        elif bad=="nonzero_duration":eof["duration"]=Fraction(1,512)
        elif bad=="before_stop":eof["onset"]=Fraction(400)
        elif bad=="outside_file":eof["onset"]=Fraction(native["n_times"],512)
    with pytest.raises(ValueError,match="EndOfFile"):
        adapter._trials(native)


def with_online_controls(native):
    annotations=native["annotations"]
    template=annotations[-1]
    annotations[-1:-1]=[
        dict(template,description="1010",onset=Fraction(400),original_annotation_index=241),
        dict(template,description="33281",onset=Fraction(402),original_annotation_index=242)]
    annotations[-1]["original_annotation_index"]=243
    return native


def test_documented_online_post_trial_controls_do_not_change_any_native_MI_binding(native_fixture):
    original=adapter._read_native_metadata(native_fixture)
    before=adapter._trials(original,run_phase="online")
    after=adapter._trials(with_online_controls(deepcopy(original)),run_phase="online")
    assert before==after and len(after)==40
    assert [r["original_annotation_index"]for r in after]==list(range(4,240,6))
    assert [r["label"]for r in after]==[0,1]*20


@pytest.mark.parametrize("bad",["middle","duplicate","missing_half","wrong_order","before_last_end",
                                 "same_time_as_last_end","after_stop","outside_file","acquisition_phase"])
def test_online_control_suffix_is_strictly_posttrial_and_ordered(native_fixture,bad):
    native=with_online_controls(adapter._read_native_metadata(native_fixture));rows=native["annotations"]
    phase="online"
    if bad=="middle":rows.insert(40,rows.pop(-3))
    elif bad=="duplicate":rows.insert(-2,dict(rows[-2]))
    elif bad=="missing_half":del rows[-2]
    elif bad=="wrong_order":rows[-3],rows[-2]=rows[-2],rows[-3]
    elif bad=="before_last_end":rows[-3]["onset"]=rows[-4]["onset"]-1
    elif bad=="same_time_as_last_end":rows[-3]["onset"]=rows[-4]["onset"]
    elif bad=="after_stop":rows[-2]["onset"]=rows[-1]["onset"]+1
    elif bad=="outside_file":rows[-3]["onset"]=Fraction(native["n_times"],512)+1
    else:phase="acquisition"
    with pytest.raises(ValueError,match="post-trial"):
        adapter._trials(native,run_phase=phase)


def test_materialize_reads_only_bounded_clinic11_context_once_in_volts(native_fixture, profile, monkeypatch):
    reads = []
    opened = mne.io.read_raw_edf
    def tracked(*args, **kwargs):
        raw = opened(*args,**kwargs)
        get = raw.get_data
        def bounded(*,picks,start,stop):
            assert picks == list(CHANNELS) and stop-start == 768
            reads.append((start,stop))
            return get(picks=picks,start=start,stop=stop)
        raw.get_data = bounded
        return raw
    monkeypatch.setattr(mne.io,"read_raw_edf",tracked)
    prepare = adapter.prepare_trial
    def volts_only(signal, **kwargs):
        assert kwargs["native_unit"] == "V"
        assert kwargs["task_sample"] == 0 and kwargs["task_end_sample"] == 768
        assert np.max(np.abs(signal)) < .001
        return prepare(signal,**kwargs)
    monkeypatch.setattr(adapter,"prepare_trial",volts_only)
    X,y,events,meta = adapter.materialize(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)
    assert X.shape == (40,11,128) and X.dtype == np.float32
    assert y.dtype == np.int64 and y.tolist() == [0,1]*20 and np.isfinite(X).all()
    assert len(reads) == len(events) == 40
    assert meta["valid_trials"] == 40 and meta["excluded_trials"] == 0
    assert meta["normalization_applied"] is False and meta["actual_fits"] == 0
    assert all(row["preprocessing"]["native_unit"] == "V" and row["preprocessing"]["si_conversion_factor"] == 1.0 for row in events)
    native = adapter.audit_native(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)
    assert meta["independent_native_metadata_semantic_sha256"] == digest(native)
    for event,original,read in zip(events,native["trials"],reads):
        assert all(event[k] == v for k,v in original.items())
        assert list(read) == original["original_context_samples"]
        assert read[0] >= Fraction(original["native_cue_onset_token"])*512
        assert read[1] <= original["task_end_sample"]


@pytest.mark.parametrize("mutation",["anchor","end","labels","role","unit","sfreq","contract","query","fit"])
def test_changed_profile_rejected_before_native_read(native_fixture,profile,monkeypatch,mutation):
    changes = {"anchor":("native_anchor","feedback"), "end":("native_end","cue_duration"),
        "labels":("raw_label_to_y",{"769":1,"770":0}),"role":("role","unseen_external"),
        "unit":("native_units",["V"]),"sfreq":("native_sfreq_hz",128),
        "query":("external_query_access",True),"fit":("training_started",True)}
    if mutation == "contract":
        profile["signal_contract"]["crop_s"] = [0,1]
    else:
        key,value = changes[mutation]
        profile[key] = value
    monkeypatch.setattr(adapter,"_read_native_metadata",lambda *a:pytest.fail("modified profile read native"))
    with pytest.raises((ValueError,PermissionError)):
        adapter.materialize(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)


def test_wrong_full_SHA_rejected_before_native_parse(native_fixture,profile,monkeypatch):
    monkeypatch.setattr(adapter,"_read_native_metadata",lambda *a:pytest.fail("wrong SHA read native"))
    with pytest.raises(ValueError,match="SHA256"):
        adapter.materialize(native_fixture,expected_sha256="0"*64,profile=profile)


@pytest.mark.parametrize("change",["missing_end","feedback_first","cue_class_missing","short_task","trial_order"])
def test_bad_native_trial_structure_never_invents_trials(native_fixture,change):
    native = adapter._read_native_metadata(native_fixture)
    rows = native["annotations"]
    if change == "missing_end":
        del rows[6]
    elif change == "feedback_first":
        rows[4],rows[5] = rows[5],rows[4]
    elif change == "cue_class_missing":
        rows[4]["description"] = "770"
    elif change == "short_task":
        rows[6]["onset"] = rows[4]["onset"]+1
    else:
        rows[3]["onset"] = rows[4]["onset"]+1
    with pytest.raises(ValueError):
        adapter._trials(native)


@pytest.mark.parametrize("header_change",["missing_channel","bad_unit","bad_clock","truncated"])
def test_bad_original_header_or_record_clock_blocks_before_EEG(tmp_path,native_fixture,profile,header_change):
    p=tmp_path/"sub-01"/"eeg"/native_fixture.name
    p.parent.mkdir(parents=True)
    shutil.copyfile(native_fixture,p)
    with p.open("r+b") as f:
        if header_change == "missing_channel":
            f.seek(256+16*NAMES.index("C3"));f.write(_field("X3",16))
        elif header_change == "bad_unit":
            f.seek(256+96*33+8*NAMES.index("C3"));f.write(_field("mystery",8))
        elif header_change == "bad_clock":
            f.seek(256*(33+1)+2*(32*512+256)+2*32*512)
            f.write(b"+2")  # record1 must carry +1, not +2
        else:
            f.truncate(p.stat().st_size-1)
    with pytest.raises(ValueError):
        adapter.audit_native(p,expected_sha256=file_digest(p),profile=profile)


def test_waveform_exclusion_keeps_native_UID_and_full_denominator(native_fixture,profile,monkeypatch):
    opened=mne.io.read_raw_edf
    count=0
    def changed(*a,**kw):
        raw=opened(*a,**kw);get=raw.get_data
        def bounded(**kwargs):
            nonlocal count
            x=get(**kwargs);count+=1
            if count==1:x[0,0]=np.nan
            return x
        raw.get_data=bounded
        return raw
    monkeypatch.setattr(mne.io,"read_raw_edf",changed)
    native=adapter.audit_native(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)
    X,y,events,meta=adapter.materialize(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)
    assert X.shape[0] == y.size == len(events) == 39
    assert meta["original_trials"] == 40 and meta["excluded_trials"] == 1
    assert meta["exclusions"][0]["trial_uid"] == native["trials"][0]["trial_uid"]
    assert "Nonfinite" in meta["exclusions"][0]["exclusion_reason"]
    assert set(r["trial_uid"] for r in events+meta["exclusions"]) == set(r["trial_uid"] for r in native["trials"])


def test_MNE_annotation_rewrite_rejected_before_array_read(native_fixture,profile,monkeypatch):
    opened=mne.io.read_raw_edf
    def changed(*a,**kw):
        raw=opened(*a,**kw)
        raw.annotations.description[4]="770"
        raw.get_data=lambda **kwargs:pytest.fail("rewritten MNE annotation read EEG")
        return raw
    monkeypatch.setattr(mne.io,"read_raw_edf",changed)
    with pytest.raises(ValueError,match="MNE annotation"):
        adapter.materialize(native_fixture,expected_sha256=file_digest(native_fixture),profile=profile)


def test_known_absent_subject59_runs_are_not_invented(tmp_path,profile):
    path=tmp_path/"sub-59"/"eeg"/"sub-59_task-R5online_eeg.edf"
    with pytest.raises(ValueError,match="no subject59"):
        adapter.materialize(path,expected_sha256="0"*64,profile=profile)
