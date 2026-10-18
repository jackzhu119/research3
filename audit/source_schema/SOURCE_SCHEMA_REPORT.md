# Four-source native schema preparation

This receipt covers one pre-registered specimen from each approved source. It is
not a full source cohort manifest, a frozen numerical input contract, or a fit.
No raw file came from a previous paper's NPZ cache.

| Source specimen | Complete bytes | Native LR metadata trials | What was actually verified |
|---|---:|---:|---|
| BNCI `A01T.mat` | 42,806,453 | 144: 72 left, 72 right | Native 25-column arrays; three EOG records with no events followed by six physical MI runs; 250 Hz; native four-class labels and artifact flags. |
| Cho `s01.mat` | 203,446,256 | 200: 100 left, 100 right | Original class-separated 68-channel streams, 512 Hz, 100 event pulses per class, native frame and task-array identity. |
| OpenBMI `sess01_subj01_EEG_MI.mat` | 609,007,751 | 100 offline train: 50 left, 50 right | Actual 62-channel strings including clinic11, 1,000 Hz, native labels and four-second epochs; all 100 `x[t:t+4000]` slices exactly equal provider `smt`. |
| PhysioNet `S001R04.edf` | 2,596,896 | 15: 8 left, 7 right | Actual 64-channel EDF labels, physical units µV, 160 Hz, native run 4 and task annotations. Local SHA256 equals the provider's published SHA256SUMS entry. |

Total complete raw bytes: **857,857,356**. The 459 rows are source schema
observations, not a claim about post-window eligible training trials. Complete
raw SHA256 values are in `raw_inventory.json`. Native rows preserve a content
identity derived from source SHA, dataset/version, participant, record, run and
original trial index; class filtering never renumbers a provider trial.

BNCI contains the required measured C5 and C6 columns. Its MAT lacks channel
strings; the order is supported by the provider's montage and a retained,
fixed MOABB v1.7.2 column map. Cho likewise has coordinates rather than channel
strings, so its Biosemi64 order is labelled as external mapping evidence. No
interpolation, zero fill or virtual montage supplies a missing measurement.

Three MAT variants lack an explicit unit/calibration field. Their retained
fixed MOABB adapters declare numeric µV and a common `1e-6` EEG conversion; the
audit records that evidence separately from native header evidence. It does
not call these MAT headers unit-verified or authorize absolute amplitude QC.
The EDF has an actual physical-dimension header; MNE returns that signal in V.

The BNCI native trial positions are MATLAB one-based trial-start indices; cue
and MI start are 2 seconds later according to the original provider document.
OpenBMI sample alignment is proved by equality to the provider epoch arrays,
not inferred from MATLAB conventions: the correct offset in this native
release is exactly zero. Its one offline training block is not relabelled as
a newly recovered physical run, and no calendar date is invented.

Cho's paper explicitly defines `imagery_event == 1` as MI onset. The actual
pulse is sample 1023 of every 3584-sample provider epoch, one sample earlier
than a zero inferred from `frame=[-2000,5000]`. The adapter preserves the pulse
and records that discrepancy. Class-separated order does not establish
acquisition order: physical run and chronological episode permission remain
null/false, and Cho is usable only as a supervised source.

The provider OpenBMI container contains both train and test variables. An
initial exploratory full `loadmat` inspected their shapes and field structure;
this source-container access is retained in `source_container_access_log.json`.
No model or selection used its test data. The formal reader loads only
`EEG_MI_train`. These source variables are distinct from the sealed HS/Wang
external cohorts, neither of which was accessed by these tools.

Provider evidence is preserved verbatim and hashed in `evidence_manifest.json`:
BNCI currently states CC BY-ND 4.0; Cho and OpenBMI DataCite records state CC0;
PhysioNet states ODC-By 1.0. This package does not authorize public redistribution
of processed BNCI signals. Private backup decisions belong to the root task.

The remaining steps are the complete source inventories/identity rosters,
metadata-only Stieger split and its meta-train-only window decision, then the
frozen preprocessing contract. Only after that gate may new numerical caches
and controlled pilot inputs be materialized. No filtering, cache generation,
fitting, GPU job, RunPod operation, R2 upload, HS or Wang payload access was
performed by this preparation subtask.

The W2 implementation is now available in `tools/prepare_all_sources.py`, with
serial resumable native preparation, per-participant NPY caches, and separate
raw/cache R2 receipts. `plans/source_preparation_candidates.json` lists exactly
505 required candidate objects but explicitly does not verify their existence
or full archive bytes. Only the four specimens above are currently confirmed
by this subtask. A candidate dry run returns a non-completion status; W1 and its
frozen window still block numerical execution. The 36 software checks use only
temporary mathematical arrays or mocked network receipts. They are not EEG
training, source cohort completion, or scientific pilot results.
