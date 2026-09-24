# AI-SPM Guardrail Eval

A three-stage AI Security Posture Management (AI-SPM) guardrail for enterprise
LLM pipelines — semantic prompt-injection shielding, RAG context control, and
output-side entailment auditing — sharing one lightweight embedding
representation instead of paying for a second generative-LLM call at any
checkpoint. Benchmarked against a regex filter, a heavyweight secondary
zero-shot transformer, and a fine-tuned open-source injection classifier, with
full reproducible code, data-generation scripts, and results.

This repo is the code and data release accompanying the paper
"Regime-Adaptive Conformal Guardrails for Prompt Injection Attacks under
Distribution Shift." (Link to be added once published.)

## The central contribution

Regime-Adaptive Drift-Triggered Conformal Recalibration (RA-DTCR) is a
stateful operating-point controller for frozen prompt-injection guardrails.
Label-hidden windows are tested against the deployed regime under a declared
horizon-wide error budget. A trigger buys labels and refreshes both the regime
and benign conformal references. The implementation exposes finite-sample
marginal benign-FPR and horizon-wide false-trigger assumptions directly.

The released study evaluates a semantic-similarity shield, ProtectAI-v2,
and PIGuard on four public targets using 100 overlapping split-sensitivity
runs per pair. Frozen thresholds fail in both directions. DTCR moves mean
FPR to 0.045–0.052 across all 12 transfers. Because every shifted transfer
triggers, one-shot DTCR matches periodic target conformalization and spends
the same labels in all twelve; all three disjoint same-distribution controls
spend zero labels.
In twelve-window constructed regime streams, RA-DTCR cuts labels by
63.9–69.0% versus periodic conformalization with pooled FPR 0.045–0.059
(periodic: 0.046–0.049) and recall changes from −0.011 to +0.036. The paper
and result JSON report
per-pair realized-FPR exceedance rates and disclose
the exchangeability, finite-sample, detector-quality, and overlapping-split
limits explicitly.

The hard Stage 3 transfer test has AUROC 0.995 and recall 1.000, but the
easy-set threshold blocks 163/180 faithful responses (FPR 0.906). This is a
ranking/operating-point failure, not deployable performance.

## Repository layout

```
guardrail/              Pipeline source code (the thing being evaluated)
  stage1_shield.py         Stage 1: semantic injection shield + NER/PII masking
  stage2_rag_control.py    Stage 2: RAG context control (whole-chunk + windowed)
  stage3_entailment.py     Stage 3: NLI-based entailment auditor
  conformal_recalibration.py KS drift trigger + conformal decision utilities
  baselines.py             Comparison systems (regex, secondary transformer, OSS classifier)
  pipeline.py              End-to-end 3-stage orchestrator with per-stage latency tracing

data/                    Evaluation datasets + the scripts that generate them
  generate_dataset.py       Builds all synthetic *.jsonl files (fixed seed)
  fetch_public_benchmark.py Fetches + caches the public external benchmark
  fetch_shift_benchmarks.py Fetches + standardizes four public shift targets
  reference_bank.json       Stage 1/2 injection reference vectors (bank D)
  *_eval.jsonl               The evaluation sets themselves (see Datasets below)

experiments/             Benchmark harness + figure generation
  run_benchmark.py          Runs every experiment reported in the paper
  run_shift_recalibration.py Runs the 3-detector x 4-target DTCR study
  run_regime_stream.py       Runs the 12-window stateful prequential study
  sweep_stage2_window.py    Stage 2 window-size/stride hyperparameter sweep
  make_figures.py           Generates every figure in the paper from results.json
  results/                  Pipeline and DTCR result JSON/NPZ artifacts

requirements.txt         Minimum Python dependencies
requirements-lock.txt    Exact environment used for the reported run
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Python 3.9+ (developed and tested on 3.9.6). Everything runs on CPU — no GPU
required or used anywhere in this repo; all latency numbers reported in the
paper are CPU-only, unbatched, single-request measurements (Apple M3
Pro laptop), disclosed as such rather than presented as production-serving
numbers.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

63 unit tests covering the core logic and provenance controls: L2 normalization
and cosine-similarity scoring, the NER/PII span-overlap resolution algorithm
(including a regression test for the leading-parenthesis phone-number bug
disclosed in the paper), the `_word_windows` windowing/striding arithmetic,
the entailment softmax and label-index lookup, the three-stage pipeline's
control flow (early exit at each stage, decision labeling), and every
baseline's label-extraction logic, KS drift detection, conformal p-values,
stratified partitions, the tied-score threshold regression, cross-source
de-duplication, deterministic target selection, content-addressed score-cache
invalidation, horizon spending, regime refresh, and stream-split disjointness.
None load the real
embedding/NLI/classification models -- model-backed classes are instantiated
via `__new__` with the model-dependent calls stubbed, so the suite runs in
well under a second and exercises the same code paths `run_benchmark.py`
does without any network access or model-download cost. This checks the
mechanism `run_benchmark.py`'s numbers depend on; it does not re-verify
those numbers themselves, which come from the real models via the
reproduction steps below.

## Reproducing everything, end to end

```bash
# 1. Generate the synthetic datasets (deterministic, fixed seed=42)
python3 data/generate_dataset.py

# 2. Fetch + cache the public external benchmark (requires network + `datasets` lib)
python3 data/fetch_public_benchmark.py
python3 data/fetch_shift_benchmarks.py

# 3. Run every benchmark reported in the paper
python3 experiments/run_benchmark.py
#    -> writes experiments/results/results.json and raw_scores.npz

# 4. Run DTCR across three frozen detectors and four public targets
python3 experiments/run_shift_recalibration.py
#    -> writes shift_scores.npz and shift_recalibration_results.json

# 5. Run the stateful 12-window regime-stream evaluation
python3 experiments/run_regime_stream.py
#    -> writes regime_stream_results.json

# 6. (optional) Stage 2 window/stride sweep, on the dev split only
python3 experiments/sweep_stage2_window.py
#    -> writes experiments/results/stage2_window_sweep.json

# 7. Regenerate every figure from results.json
python3 experiments/make_figures.py
#    -> writes ROC/PR curves and confusion matrices as PDF + PNG
```

Every number reported in the paper is produced by this exact sequence, with
the seed recorded in `results.json`'s `meta` block along with the platform,
PyTorch version, and thread count the run used.

## The pipeline, briefly

Formalized as `F(x) -> y`: input prompt in, validated output out, three
sequential decision points, each independently thresholded.

1. **Stage 1 — Semantic Shielding** (`stage1_shield.py`): embeds the input
   with `all-MiniLM-L6-v2` and computes max cosine similarity against a
   reference bank of known injection strings. Above `tau_injection`, the
   prompt is dropped. Prompts that pass are then run through NER + regex-based
   PII redaction (email, phone, SSN, credit card) before being forwarded.
2. **Stage 2 — RAG Context Control** (`stage2_rag_control.py`): the same
   mechanism applied to *retrieved* context chunks rather than user input,
   since indirect injection hides instructions in third-party content a RAG
   loop pulls in. Two scoring modes are implemented and compared: whole-chunk
   embedding (the original design) and windowed sub-chunk scoring (overlapping
   20-word windows, stride 10 by default) — the development-selected setting
   raises recall from 0.333 to 0.875
   at added latency cost; see the paper for the full trade-off.
3. **Stage 3 — Output Entailment Auditing** (`stage3_entailment.py`): scores
   `P(entailment | response, context)` with a DeBERTa-v3 NLI cross-encoder.
   Below `tau_entailment`, the response is treated as unsupported/hallucinated
   relative to its own retrieved context.

All three thresholds are tuned (not asserted). Stages 2 and 3 use 40%/60%
development/test splits; Stage 3 keeps both rows from each factual context on
the same side. Stage 1 uses a balanced 240-row development set drawn only from
benign and reference-represented families, while all 75 attacks from the three
reference-omitted families remain test-only in the 360-row test set. Thresholds
are swept for the most permissive value keeping development FPR at or below 5%.

## Datasets

Core datasets live in `data/` as JSONL. Five are synthetic and four public
shift targets are standardized under `data/external_shift/`:

| File | Examples | What it tests |
|---|---|---|
| `stage1_eval.jsonl` | 600 | Injection detection: 300 benign enterprise queries + 300 adversarial prompts across 12 attack families. The reference bank is built from only 9 families; all 75 examples from the other 3 are excluded from threshold development and reserved for test. |
| `stage2_eval.jsonl` | 400 | Indirect injection: 200 clean + 200 poisoned RAG context chunks, each paired with a topically matched query so poison detection can't be reduced to a relevance heuristic. |
| `stage3_eval.jsonl` | 400 | Hallucination detection ("easy" set): 200 faithful paraphrases + 200 either numeric contradictions or unsupported additions, with large (30–100%) perturbations. |
| `stage3_hard_eval.jsonl` | 360 | Hallucination detection ("hard" set): the same construct with subtle perturbations (5–15% numeric drift, direction flips at unchanged magnitude, topically-consistent unsupported additions) — built specifically to stress-test the easy set's ceiling effect. |
| `pii_eval.jsonl` | 150 sentences, 227 spans | PII masking: synthetic support-ticket sentences with inserted names/emails/phones/SSNs/credit-card numbers, with exact character-offset ground truth. |
| `public_benchmark_eval.jsonl` | 662 | The one non-synthetic source: deepset's `prompt-injections` dataset (HF Hub), fetched via `fetch_public_benchmark.py`. Used to test whether already-tuned thresholds transfer to data this project had no hand in constructing. Its label definition differs slightly from ours — see the docstring in `fetch_public_benchmark.py` for the full caveat. |
| `external_shift/xtram.jsonl` | 2,048 | xTRam public test split after normalized cross-source de-duplication. |
| `external_shift/neuralchemy.jsonl` | 942 | Neuralchemy core test split with benign hard negatives. |
| `external_shift/safeguard.jsonl` | 1,977 | Deterministic near-balanced subset of jcanode Safeguard's held-out test split. |

`reference_bank.json` holds the ~30 canonical injection-pattern strings (9 of
the 12 attack families) that Stage 1/2's cosine-similarity check scores
against.

Third-party target text is fetched locally and ignored by Git rather than
redistributed. `data/shift_benchmarks_manifest.json` pins upstream revisions
and records licenses, selection rules, counts, duplicate removals, and SHA-256
checksums. Generated targets have zero normalized exact overlap; cached model
scores are invalidated whenever a source checksum, row count, model revision,
or semantic reference bank changes.

## Experiments / what `run_benchmark.py` actually computes

`run_benchmark.py` runs the motivating pipeline experiments:

- **Stage 1 injection detection** — the proposed shield vs. three baselines
  (`baselines.py`): a legacy regex/keyword filter, a secondary heavyweight
  zero-shot transformer (`facebook/bart-large-mnli`, standing in for "call a
  second general-purpose model on every input"), and an existing open-source
  fine-tuned injection classifier (`deepset/deberta-v3-base-injection`).
  Recall, precision, F1, FPR, AUROC, P99 latency, and throughput for all four,
  plus 95% bootstrap confidence intervals (2,000 resamples) and McNemar's
  test for every head-to-head comparison.
- **Reference-bank generalization** — shield recall split by whether the
  attack family had a representative in the reference bank.
- **Public benchmark transfer** — the same, already-tuned thresholds (no
  retuning) applied to the external deepset dataset.
- **Stage 2 RAG context control** — whole-chunk vs. windowed scoring,
  including a topical-relevance control (confirming the detector isn't just
  exploiting an off-topic shortcut) and a dedicated-vs-shared-embedder
  ablation (`all-mpnet-base-v2` as an independently-loaded alternative to
  reusing Stage 1's embedder).
- **Stage 3 entailment auditing** — both the easy and hard sets, at the same
  threshold, broken down by hallucination subtype (numeric contradiction,
  unsupported addition, direction flip).
- **PII masking** — entity-level precision/recall/F1, broken down by entity
  type (regex-based fields vs. the NER-based person-name detector).
- **End-to-end pipeline latency** — full 3-stage sequential worst-case timing.
- **Incremental pipeline coverage ablation** — what each stage actually
  contributes, measured on a combined attack set spanning all three vectors
  plus a benign negative control, using every stage's already-tuned threshold
  with no per-configuration retuning.

`sweep_stage2_window.py` separately sweeps Stage 2's window size/stride over
a small grid on the *development* split only (the test split used everywhere
else is never touched by this sweep) — it exists specifically to answer
"were 20/10 chosen empirically or just asserted," honestly.

Results land in `experiments/results/results.json` (all metrics) and
`raw_scores.npz` (raw per-example scores, used by `make_figures.py` to plot
ROC/PR curves and confusion matrices without re-running the models).

`run_shift_recalibration.py` independently runs DTCR and four comparators
over three frozen detectors and the four public target files. It uses
disjoint 20/40/40 monitor/calibration/test partitions, 100 fixed seeds,
10/20/40% label-budget sensitivity, and disjoint same-distribution controls.
Its outputs are `shift_scores.npz` and
`shift_recalibration_results.json`; the seeds overlap rows and are therefore
sensitivity runs, not independent replications.

`run_regime_stream.py` evaluates RA-DTCR, periodic target conformalization,
source-referenced DTCR, and a frozen conformal bank over twelve ordered
windows with five public-regime transitions and a recurrence. Rows are
disjoint within each seed; because the source datasets have no timestamps,
the order is a constructed prequential stress test rather than a production
chronology. Its output is `regime_stream_results.json`.

## Reproducibility notes

- Every synthetic dataset is generated with a fixed seed (42); regenerating
  via `generate_dataset.py` is deterministic.
- Accuracy/detection metrics are deterministic given the recorded seeds. Latency
  numbers are not — they vary run-to-run with machine load, which is
  measured and reported (not hidden) across multiple runs in the paper.
- `results.json`'s `meta` block records the exact platform, PyTorch version,
  thread count, and seed used to produce the numbers currently checked in.

## Limitations (short version)

The pipeline corpus is mostly synthetic; the recalibration targets are public,
but the regime order is constructed rather than a chronological production
trace. The conformal statement is marginal
for a future benign score under exchangeability, not a batch-FPR or
continuing-drift guarantee. Released classifier training data may overlap
targets. Latency numbers come from one CPU machine, and guardrail-aware
adaptive attackers are out of scope. Read the paper before treating any
number as a production guarantee.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Sagar Pradip Chaudhari — schaud16@binghamton.edu
