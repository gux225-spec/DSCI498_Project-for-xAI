# DSCI498 Filter Surrogate Explanation Project

Repository: [gux225-spec/DSCI498_Project-for-xAI](https://github.com/gux225-spec/DSCI498_Project-for-xAI).

This project tests whether explanations of a microstrip band-pass filter surrogate are faithful to the model, consistent with its ADS circuit-simulation source, and useful for selecting geometry changes. An independent openEMS full-wave stress test checks whether circuit-confirmed tuning decisions transfer to another modeling level.

PR-Net is the primary model because its pole/residue representation supports concept checks. M1 is a more accurate frequency-conditioned MLP comparison; M0 adapts the network layout reported by Suslova et al.; B1 is PCA plus Gaussian-process regression. M0 differs from the paper in data, feature count and outputs. Published paper numbers are contextual comparisons, not a controlled benchmark.

Experiments and saved results are available. Final materials were prepared on **November 22, 2026**: an [8-page final report](report/final/DSCI498_FinalReport.pdf), a [2-page AI-use reflection](report/ai_use_reflection/DSCI498_AI_Use_Reflection.pdf), and a [requirement-by-requirement checklist](docs/submission_checklist.md). Original draft files and preparation notes are retained as history.

## Verified findings

See [the archived-result audit](results/submission_audit/verification.md) and its [numeric details](results/submission_audit/verification.json). The verifier recalculates results from saved predictions and simulator responses without retraining or launching new simulations.

| Model | S21 MAE dB | S11 MAE dB | Centre-frequency MAE MHz | Bandwidth R2 |
|---|---:|---:|---:|---:|
| PR-Net | 0.083928 | 0.078133 | 2.728483 | 0.910204 |
| M1 | 0.021145 | 0.025027 | 0.374020 | 0.999872 |
| M0 | 0.266388 | 0.811044 | 29.296504 | -0.342229 |

Rows are three-seed means. Curve MAE uses all **216 test designs**. Centre frequency and bandwidth use **202 valid single-band designs**, with peak-relative -3 dB edges.

- PR-Net single-reference SHAP rankings agree with ADS Shapley at approximately 0.98–0.99. This refers to a particular simulator game; it is not a hardware-truth claim or a universal criterion for IG/LIME faithfulness.
- Of 30 preregistered failing test designs, SHAP and IG selected parameter pairs allowing ADS-confirmed tuning in 25 cases; LIME-delta in 24; random pairs in 6. Six-parameter surrogate search succeeded in all 30. SHAP versus random has a Holm-adjusted McNemar p value of 4.1962e-5.
- Three selected SHAP-tuned designs passed in ADS and failed the mask in openEMS. Two improved and one worsened. These examples establish counterexamples to transfer, not a population failure rate.
- Interviews with one RF practitioner on November 5 and November 21 inform workflow requirements. They do not measure diagnostic time savings or improved human decisions.

## Code and evidence archives

Download the paired archives and final PDFs from the [submission release](https://github.com/gux225-spec/DSCI498_Project-for-xAI/releases/tag/submission-2026-11-22). The code repository keeps source and lightweight records; the release carries binary reproduction evidence.

| Archive | Contents |
|---|---|
| `DSCI498_code.zip` | Source, experiment scripts, documentation, requirements, baseline netlist, draft/final reports, AI-use reflection and figures |
| `DSCI498_evidence.zip` | Complete dataset release, checkpoints, saved predictions/attributions, ADS response caches, tuning records and full-wave responses |

Extract **both ZIPs into the same parent directory**, merging their `DSCI498_Project/` folders. The code ZIP can be inspected alone; computation requires evidence assets. Every file in the original dataset hash manifest is retained. Python caches, solver working directories and intermediate openEMS `sim/` outputs are excluded; `response.npz`, summaries and experiment records remain. Native ADS workspaces outside this project and installed commercial/native solvers are not included.

Each archive contains a file-size/SHA256 manifest under `manifests/`. The separate `submission_package_index.json` lists archive contents/exclusions, and `SHA256SUMS.txt` provides archive checksums. The third-party full-text working paper is excluded; its bibliography and citations remain. Verify an archive with the standard library:

```powershell
python scripts/build_submission.py --verify ../DSCI498_evidence.zip
```

The `.gitignore` excludes large arrays/checkpoints from ordinary Git commits. Distribute the evidence ZIP separately; a code-only Git clone will lack those assets. The archive builder only writes local ZIPs and does not initialize or push Git repositories.

## Environment and quick verification

Python **3.11** was checked. The minimal route uses NumPy and SciPy, without PyTorch, ADS, openEMS or a GPU:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-check.txt
python scripts/verify_submission.py
```

Use `requirements.txt` for model inference/training, PCA/GPR and plotting. Use `requirements-xai.txt` for the complete paper-protocol SHAP/LIME runs:

```powershell
python -m pip install -r requirements-xai.txt
python scripts/verify_submission.py --live-models
```

The checked training environment has PyTorch `2.11.0+cu128`. Requirements pin public version `2.11.0` without selecting a CUDA build. The model code chooses CUDA if available, otherwise CPU. Small floating-point changes are possible across builds/devices/batch sizes, especially near deep dB notches. Installed versions and sample-inference errors are in the audit; a fresh pip installation and full retraining were not performed during release preparation.

New solver runs require separate environments:

- **ADS 2026:** its Python, `keysight.ads`, `keysight.edatoolbox`, circuit simulator and suitable license. The original installation is `E:/ADS2026`. `scripts/ads_sim.py` contains that root; `step4_tune.py` and `step4b_explore.py` contain the ADS Python path. Configure these in a separate copy for another machine; historical scripts are preserved for provenance.
- **openEMS 0.37.0 and CSXCAD:** native libraries/Python bindings, originally in `E:/anaconda3/envs/openems`, with `CSXCAD_INSTALL_PATH=E:/openEMS`. These are not installed by pip requirements. An existing environment variable overrides the default in `oems_filter.py`.
- **Momentum:** stored diagnostic responses and setup files are included. Full native regeneration can require the external ADS workspaces/paths recorded in the dataset release.

## Dataset

The release contains **1,440 unique geometries**: 480 legacy four-variable designs on a fixed-width slice and 960 new six-variable designs. They are not 1,440 six-dimensional LHS samples. Fixed design-level splits are **1008 train / 216 validation / 216 test**.

Geometry inputs are `L1, L2, S1, S2, Wc1, Wc2`, in mil. Three synthetic negative controls are appended for model/explanation checks and never enter ADS. Core responses cover **9–15 GHz, 601 points**; wideband responses cover the same designs with 2,101 points. Overlapping frequency points and control replicates are not extra independent designs. Labels are near-lossless ideal-conductor **circuit simulations**. Full-wave diagnostics are separate from training data.

```python
import sys
sys.path.insert(0, "src")
from filterxai import data

train = data.load("train")
print(train["X"].shape)       # (1008, 9), train-only normalization
print(train["S"].shape)       # (1008, 601, 2), complex [S21, S11]
print(train["Y_dB"].shape)    # (1008, 601, 2), [S21_dB, S11_dB]
```

`data.load("test")` returns inputs without labels. Explicit test-label access requires `results/step3/freeze.json`. The loader checks its existence; it does not automatically validate historical hashes. The submission verifier checks those hashes separately.

## Project map

```text
src/filterxai/                 Data, models, training, metrics, targets and explanations
scripts/                      Experiments, read-only verification and packaging
ads/baseline.ckt               Hashed baseline circuit netlist
data/filter_supplement_v2/     Dataset, loader, release hashes and Momentum diagnostics
results/step0/                 Split checks and vector-fitting concepts
results/step1/                 Validation selection, power penalty and model registry
results/step2/                 Validation explanations
results/step3/                 Freeze, test accuracy/XAI and ADS reference games
results/step4/                 Preregistered tuning and ADS confirmations
results/step4b/                Exploratory exhaustive-pair and M0 comparisons
results/fullwave_openems/      Mesh study, selected cases and paired responses
results/submission_audit/      Newly generated result verification
docs/decision_log.md           Decisions, evidence and timing qualifications
docs/reproducibility.md        Experiment commands and output locations
docs/stakeholder_interview*    Original and follow-up summaries
docs/ai_use_reflection_notes.md Confirmed AI-use facts and reflection material
report/                       Original draft, final report and AI-use reflection projects
```

## Recompute and rerun

Start with `python scripts/verify_submission.py`; add `--live-models` to check three designs with each reference-seed neural model. Use `--output PATH` to write audit files elsewhere. The verifier does not overwrite old experimental results.

[docs/reproducibility.md](docs/reproducibility.md) lists stage commands and outputs. Run original experiment/report scripts in a separate working copy: many overwrite existing results for the same tag. Preserve the original freeze/preregistration records; their creation scripts refuse to overwrite them.

Rebuild packages after a successful audit:

```powershell
python scripts/build_submission.py --output ../DSCI498_submission
```

The package destination must be outside the project.

## Interpretation and retained historical records

Three historical numerical/provenance caveats are recorded in the audit:

1. **Historical hashes:** four of 30 freeze hashes differ: the evolving plan/protocol and two scripts adapted for test paths. Plan section 12 documents changes and a validation rerun. Original prior versions are not reconstructed. All nine frozen neural weights match.
2. **Bandwidth definitions:** model metrics use peak-relative -3 dB; old full-wave summaries use absolute -3 dB. Tuned openEMS relative BW reductions are about **80–87 MHz**, while absolute reductions are **124–142 MHz**. These are different quantities; hard-mask verdicts are unchanged.
3. **Violation error:** the old approximately 0.062 dB median compares surrogate smooth violation with ADS hard violation. It is not a same-target prediction error. The single-case smooth error for `test_0005` is approximately 0.0749 dB; its LIME/IG quality values are in the audit.
The original draft's TODOs and PDF/TeX mismatch are retained as historical files. The separate final source has no TODO placeholders; its compiled report is 8 pages and the AI-use reflection is 2 pages. Their hashes/page checks are recorded in `results/submission_audit/final_artifacts.json`.

The dataset's own README/status files describe its original pretraining state. The plan's original RQ3 means tuning; the report's Q3 means full-wave robustness. Use stage names to avoid confusing these labels. Baseline mesh sensitivity does not establish convergence of every tuned geometry; missing two-port power does not uniquely isolate radiation.
