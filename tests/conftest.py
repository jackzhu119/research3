"""Narrow compatibility for an immutable historical code-identity test.

The original test bytes are themselves evidence in the all520 qualification
proof. Only its two guard cases need the old parser in their temporary root;
the current production parser and the historical test remain unchanged.
"""
import hashlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def preserve_historical_dreyer_guard_fixture(request):
    if getattr(request.node, "originalname", None) != "test_real_load_rejects_changed_frozen_qualifier_and_native_bytes":
        return
    historical_test = Path(__file__).with_name("test_resume_dreyer_qualification_transport.py").resolve()
    if Path(request.module.__file__).resolve() != historical_test:
        return
    software = request.getfixturevalue("software")
    work = request.module.work
    assert software.root.resolve() != work.ROOT.resolve()
    archive = work.ROOT / "audit/data_plan_20261020/dreyer_structure_case_review/v5_historical_code/dreyer2023.py"
    body = archive.read_bytes()
    expected = "ec2f7dbc10903c308cd30564c30b04c4653c64173fdb0331ede318fd1c3b1489"
    assert work.NATIVE_SHA == expected and hashlib.sha256(body).hexdigest() == expected
    (software.root / work.NATIVE).write_bytes(body)
