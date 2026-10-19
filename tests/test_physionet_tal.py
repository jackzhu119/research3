"""Synthetic EDF software tests plus one explicitly labelled local CPU audit.

Synthetic files contain zero ADC samples, not real EEG or scientific evidence.
The optional real-file test reads only the one ROOT-restored diagnostic file;
it performs no network request, preprocessing, cache creation or fit.
"""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

import mne
import numpy as np
import pytest

from p3work.adapters import source_dispatch as dispatch, source_native as legacy
from p3work.adapters import physionet_tal_v1 as tal
from p3work.contracts import CHANNELS, digest, file_digest


def edf_fixture(path, *, sfreq=128, records=10, events=None):
    events = events or [("0", "1", "T0"), ("1.01", "4", "T1"),
                        ("4.99", "1", "T0"), ("7.01", "4", "T2")]
    names = [name + "." for name in legacy.CHO_EEG_CHANNELS] + ["EDF Annotations"]
    ns = len(names); header_bytes = 256*(ns+1); annotation_samples = 128
    def field(value, width):
        encoded = str(value).encode("ascii")
        assert len(encoded) <= width
        return encoded.ljust(width, b" ")
    fixed = b"".join(field(v,w) for v,w in [("0",8),("SOFTWARE_TEST_DOUBLE",80),("SOFTWARE_TEST_DOUBLE",80),
        ("01.01.26",8),("00.00.00",8),(header_bytes,8),("EDF+C",44),(records,8),("1",8),(ns,4)])
    arrays = [([*names],16), ([""]*ns,80), (["uV"]*64+[""],8), (["-1"]*ns,8), (["1"]*ns,8),
        (["-32768"]*ns,8), (["32767"]*ns,8), ([""]*ns,80), ([sfreq]*64+[annotation_samples],8), ([""]*ns,32)]
    header = b"".join(field(value,width) for values,width in arrays for value in values)
    assert len(fixed+header) == header_bytes
    with path.open("wb") as stream:
        stream.write(fixed+header)
        for record in range(records):
            stream.write(np.zeros((64,sfreq),dtype="<i2").tobytes())
            block = f"+{record}\x14\x14\x00".encode()
            for onset,duration,code in events:
                if math.floor(Fraction(onset)) == record:
                    block += f"+{onset}\x15{duration}\x14{code}\x14\x00".encode()
            assert len(block) <= annotation_samples*2
            stream.write(block.ljust(annotation_samples*2,b"\x00"))
    return events


@pytest.fixture(params=[128,160])
def synthetic(tmp_path, request):
    path = tmp_path / "S088R04.edf"
    events = edf_fixture(path, sfreq=request.param)
    return path, request.param, events


def audit(path, *, revision=True):
    return dispatch.audit_native_source("PhysionetMI",path,provider_subject="88",session_id="provider_single_visit",
        source_version="SOFTWARE_TEST_DOUBLE",expected_sha256=file_digest(path),
        revision=dispatch.current_revision() if revision else None)


def software_contract(monkeypatch, seconds=1.5):
    # Only the signal-contract authority is replaced. The actual EDF parser,
    # registration, raw SHA, metadata replay and bounded get_data remain real.
    monkeypatch.setattr(tal,"validate_signal_contract",lambda c:digest(c))
    return {"SOFTWARE_TEST_DOUBLE":True,"context_s":[0,seconds],"crop_s":[.25,1.25],"n_times":128,
        "channels":list(CHANNELS),"sfreq_hz":128}


def test_fractional_native_clock_preserves_TAL_MNE_and_actual_component_evidence(synthetic):
    path, sfreq, _ = synthetic
    with pytest.raises(ValueError,match="annotation clock"):
        audit(path,revision=False)
    result = audit(path); rows = result["trials"]; revision = dispatch.current_revision()
    assert [r["original_trial_index"] for r in rows] == [2,4]
    assert [r["raw_label_code"] for r in rows] == ["T1","T2"]
    assert rows[0]["analysis_anchor_sample"] == math.ceil(Fraction("1.01")*sfreq)
    assert rows[0]["task_end_sample"] == math.floor(Fraction("4.99")*sfreq)
    assert rows[0]["annotation_end_after_next_onset"] is True
    assert rows[0]["annotation_onset_token"] == "+1.01" and rows[0]["annotation_duration_token"] == "4"
    assert rows[-1]["annotation_duration_seconds"] == 4
    assert rows[-1]["mne_annotation_duration_seconds"] == pytest.approx(2.99)
    assert rows[-1]["record_clipping_seconds"] == pytest.approx(1.01)
    assert rows[-1]["task_end_sample"] == 10*sfreq
    assert all(0 <= r["inward_onset_delay_seconds"] < 1/sfreq for r in rows)
    assert all(0 <= r["inward_stop_rounding_loss_seconds"] < 1/sfreq for r in rows)
    summary = result["summary"]
    assert summary["adapter_file_sha256"] == dispatch.FROZEN_BASE_SHA256
    assert summary["adapter_role"] == "frozen_shared_metadata_helpers_component"
    assert summary["active_variant_file_sha256"] == revision["variant_reader"]["sha256"]
    assert all(r["native_reader_revision"] == revision for r in rows)
    assert summary["native_reader_revision"] == revision


@pytest.mark.parametrize("sfreq",[128,160])
def test_exact_grid_native_UIDs_match_unchanged_legacy_reader(tmp_path,sfreq):
    path=tmp_path/"S088R04.edf"
    edf_fixture(path,sfreq=sfreq,events=[("0","1","T0"),("1","2","T1"),("4","1","T0"),("7","2","T2")])
    before=file_digest(legacy.__file__)
    old,new=audit(path,revision=False),audit(path)
    assert [r["trial_uid"] for r in old["trials"]] == [r["trial_uid"] for r in new["trials"]]
    assert [(r["analysis_anchor_sample"],r["task_end_sample"]) for r in old["trials"]] == [
        (r["analysis_anchor_sample"],r["task_end_sample"]) for r in new["trials"]]
    assert all(r["onset_was_exact_native_sample"] for r in new["trials"])
    assert file_digest(legacy.__file__) == before == dispatch.FROZEN_BASE_SHA256


def test_context_iterator_replays_revision_then_reads_no_before_onset_or_after_bound(synthetic,monkeypatch):
    path,sfreq,_=synthetic; rows=audit(path)["trials"]; calls=[]
    original=mne.io.read_raw_edf
    def opened(*args,**kwargs):
        raw=original(*args,**kwargs); get=raw.get_data
        def bounded(*,picks,start,stop):
            row=rows[len(calls)]
            assert Fraction(start,sfreq) >= Fraction(row["annotation_onset_token"])
            assert stop <= row["task_end_sample"]
            calls.append((start,stop))
            return get(picks=picks,start=start,stop=stop)
        raw.get_data=bounded
        return raw
    monkeypatch.setattr(mne.io,"read_raw_edf",opened)
    contexts=list(dispatch.iter_native_contexts("PhysionetMI",path,rows=rows,
        contract=software_contract(monkeypatch),expected_sha256=file_digest(path)))
    assert len(calls)==len(contexts)==2
    assert all(c["signal"].shape == (11,round(1.5*sfreq)) for c in contexts)
    assert all(c["native_unit"]=="V" and c["task_sample"]==0 for c in contexts)
    assert all(c["original_context_samples"]==list(call) for c,call in zip(contexts,calls))


@pytest.mark.parametrize("change",["anchor","token","duration","end","uid","mixed_revision","missing_revision","reader_sha"])
def test_changed_metadata_cannot_be_relabelled_or_read_any_signal(synthetic,monkeypatch,change):
    path,_,_=synthetic; rows=deepcopy(audit(path)["trials"])
    if change=="anchor":rows[0]["analysis_anchor_sample"]-=1
    elif change=="token":rows[0]["annotation_onset_token"]="+1.0"
    elif change=="duration":rows[0]["annotation_duration_seconds"]=9
    elif change=="end":rows[0]["task_end_sample"]+=1
    elif change=="uid":rows[0]["trial_uid"]="a"*64
    elif change=="mixed_revision":rows[1]["native_reader_revision"]["revision_id"]="random"
    elif change=="missing_revision":rows[0].pop("native_reader_revision")
    else:rows[0]["physionet_tal_reader_sha256"]="a"*64
    monkeypatch.setattr(mne.io.BaseRaw,"get_data",lambda *a,**k:pytest.fail("unverified rows read EEG"))
    with pytest.raises(ValueError):
        list(dispatch.iter_native_contexts("PhysionetMI",path,rows=rows,
            contract=software_contract(monkeypatch),expected_sha256=file_digest(path)))


def test_resealed_wrong_revision_rejected_before_raw_read(synthetic,monkeypatch):
    path,_,_=synthetic; revision=dispatch.current_revision()
    revision["variant_reader"]["sha256"]="a"*64
    revision.pop("revision_sha256");revision["revision_sha256"]=digest(revision)
    monkeypatch.setattr(mne.io,"read_raw_edf",lambda *a,**k:pytest.fail("wrong code revision opened EDF"))
    with pytest.raises(ValueError,match="revision"):
        dispatch.audit_native_source("PhysionetMI",path,provider_subject="88",session_id="provider_single_visit",
            source_version="SOFTWARE_TEST_DOUBLE",expected_sha256=file_digest(path),revision=revision)


def test_legacy_rows_keep_the_old_dispatch_and_cannot_be_upgraded(monkeypatch):
    rows=[{"SOFTWARE_TEST_DOUBLE":True}]
    monkeypatch.setattr(legacy,"iter_native_contexts",lambda *a,**k:iter([{"old_reader":True}]))
    assert list(dispatch.iter_native_contexts("PhysionetMI","unused",rows=rows,contract={},expected_sha256="a"*64)) == [{"old_reader":True}]
    with pytest.raises(ValueError,match="Legacy Physio"):
        list(dispatch.iter_native_contexts("PhysionetMI","unused",rows=rows,contract={},expected_sha256="a"*64,
            revision=dispatch.current_revision()))
    assert list(dispatch.iter_native_contexts("Cho2017","unused",rows=rows,contract={},expected_sha256="a"*64,
        revision=dispatch.current_revision())) == [{"old_reader":True}]


def test_short_contexts_preserve_both_original_trials_as_exclusions(tmp_path,monkeypatch):
    path=tmp_path/"S088R04.edf"
    edf_fixture(path,events=[("0","1","T0"),("1.01","1","T1"),("4","1","T0"),("7.01","1","T2")])
    rows=audit(path)["trials"]
    monkeypatch.setattr(mne.io.BaseRaw,"get_data",lambda *a,**k:pytest.fail("short context was padded/read"))
    result=list(dispatch.iter_native_contexts("PhysionetMI",path,rows=rows,contract=software_contract(monkeypatch),expected_sha256=file_digest(path)))
    assert [c["row"]["trial_uid"] for c in result] == [r["trial_uid"] for r in rows]
    assert all(c["excluded"]=="insufficient_original_task_context" for c in result)


DIAGNOSTIC=Path('/workspace/scratch/paper3-physio128-diagnostic/data/source_preparation/staging/56b245baf8e2a537aac261c49cac2f44ad7353b791575e227676231fbb5e7e34/S100R08.edf')
@pytest.mark.skipif(not DIAGNOSTIC.is_file(),reason="ROOT-restored real diagnostic raw is not packaged or fetched by tests")
def test_real_restored_S100_all_twelve_bounded_contexts_cpu_audit_only():
    """REAL local bounded array read; no preprocessing, training or pilot claim."""
    sha="8371eec06d0b1ff62d10e016124e3bc2db411ab6764fea572fb2d1f25a6afd97"
    assert file_digest(DIAGNOSTIC)==sha
    revision=dispatch.current_revision()
    result=dispatch.audit_native_source("PhysionetMI",DIAGNOSTIC,provider_subject="100",session_id="provider_single_visit",
        source_version="1.0.0",expected_sha256=sha,revision=revision)
    rows=result["trials"]
    contract=json.loads((Path(__file__).resolve().parents[1]/"contracts/input_contract.json").read_text())
    assert contract["context_s"]==[0,1.5]
    contexts=list(dispatch.iter_native_contexts("PhysionetMI",DIAGNOSTIC,rows=rows,contract=contract,
        expected_sha256=sha,revision=revision))
    assert len(rows)==len(contexts)==12 and all(c["signal"].shape==(11,192) for c in contexts)
    assert all(np.isfinite(c["signal"]).all() for c in contexts)
    for row,context in zip(rows,contexts):
        assert row["trial_uid"]==digest({k:row[k] for k in ("dataset","source_version","provider_subject","session_id",
            "source_file_sha256","source_record_id","physical_run_id","original_trial_index","raw_label_code")})
        assert Fraction(context["original_context_samples"][0],128)>=Fraction(row["annotation_onset_token"])
        assert context["original_context_samples"][1]<=row["task_end_sample"]
    assert rows[-1]["annotation_duration_token"]=="5.125" and rows[-1]["mne_annotation_duration_seconds"]==5.1
