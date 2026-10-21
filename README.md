# Paper3 research code

This is a curated source snapshot for the frozen Paper3 research release
`603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7`.
The original 197 software input hashes are preserved. The snapshot also contains
dependency locks, the one-click launcher, and its regression tests.

To launch, use the separately delivered `Paper3_OneClick_20261021.zip` on the
Linux GPU machine the user has manually started, then run `bash START.sh` in
that extracted bundle. Its pinned SDK wheels and fixed package descriptors are
part of that delivery. This code-only repository intentionally omits datasets,
private protocol attachments, credentials, and the original repository history.
The launcher asks for GitHub and existing R2 credentials through hidden terminal
input; tokens must never be committed or supplied as command arguments.

The frozen bootstrap downloads only approved objects, verifies complete GET
SHA256, installs pinned offline dependencies, and enforces the real 200-update
source pilot and resume gate before the registered main campaign. Mainline
freeze precedes the separate Yang and Dreyer source arms. External W7 query
remains sealed. `legacy_baseline_v1` is a historical control, not a new method.
Scientific completion and result backup are established by actual producer and
full-GET receipts after execution, not by this source snapshot.

The one-click test suite uses local fixtures and intercepted network actions:
`python -B -m pytest -q tests/test_oneclick_launcher.py tests/test_oneclick_transport.py tests/test_oneclick_github.py`.
