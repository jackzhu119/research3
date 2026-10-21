"""Temporary code-only fixture: the historical transport rejects V5 native."""
import hashlib

import pytest

from tools import resume_dreyer_qualification_transport as work


def test_historical_qualification_entry_rejects_actual_frozen_v5_native(tmp_path):
    historical = work.ROOT / "audit/data_plan_20261020/dreyer_structure_case_review/v5_historical_code/dreyer2023.py"
    old_body = historical.read_bytes()
    assert hashlib.sha256(old_body).hexdigest() == work.NATIVE_SHA == "ec2f7dbc10903c308cd30564c30b04c4653c64173fdb0331ede318fd1c3b1489"
    qualifier_body = (work.ROOT / work.ENTRY).read_bytes()
    assert hashlib.sha256(qualifier_body).hexdigest() == work.QUALIFIER_SHA
    for name, body in ((work.ENTRY, qualifier_body), (work.NATIVE, old_body)):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    assert work.ref(tmp_path, tmp_path / work.NATIVE)["sha256"] == work.NATIVE_SHA
    current_body = (work.ROOT / work.NATIVE).read_bytes()
    assert hashlib.sha256(current_body).hexdigest() == "5c31e969eeabd17634864f6783d8473a0a79f1888faf274d758cd9c5ba4cef77"
    (tmp_path / work.NATIVE).write_bytes(current_body)
    # No plan or scientific data are present: the unchanged pre-import native
    # guard must stop before the historical qualifier could inspect either.
    with pytest.raises(work.RecoveryError, match="canonical_ec2f_must_remain_unchanged_during_qualification"):
        work.load(tmp_path)
