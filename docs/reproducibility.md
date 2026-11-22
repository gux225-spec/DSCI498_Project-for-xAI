# Reproducing archived filter experiments

Prepared November 22, 2026. Commands below are run from the project root. Extract both release ZIPs first; code and binary evidence are distributed separately. The README describes Python and solver environments.

## Read only verification

```powershell
python -m pip install -r requirements-check.txt
python scripts/verify_submission.py
```

This creates `results/submission_audit/verification.json` and `verification.md`. It checks dataset hashes, design splits, normalization, all saved neural/B1 test prediction errors, ADS coalition references, exact SHAP, tuning counts/statistics, exploratory baselines, paired full-wave responses, citation keys and figure availability. Calculations use independent NumPy/SciPy implementations rather than importing the experiment's metric/report helper functions. The dataset loader is shared for input selection and scaling.

For a sample of current-weight inference:

```powershell
python -m pip install -r requirements.txt
python scripts/verify_submission.py --live-models
```

The additional check uses the first three test designs and seed 2026 for PR-Net, M1 and M0. It is a sampled inference check, not a rerun of every prediction. Test curves/metrics for all saved seeds are checked separately. A 0.02 dB maximum sample-difference tolerance accommodates numerical differences near deep notches; actual errors are saved. This tolerance is not the model-accuracy acceptance gate.

Use `--output PATH` to keep separate audit runs. Verification invokes no retraining, new solver runs, vector-fitting refits, LaTeX compilation or human performance assessment.

## Original experiment stages

The original scripts often write their documented paths directly. Use a separate working copy before running them. Existing outputs are the released evidence, and simply rerunning a script does not recreate a past preregistration timestamp.

| Stage | Commands | Output and prerequisites |
|---|---|---|
| Input and vector fitting | `python scripts/step0_split_checks.py` then `python scripts/step0_vector_fit.py` | `results/step0/`; main Python environment with PyTorch and plotting |
| Validation sweep | `python scripts/step1_sweep.py` | `results/step1/runs/`, sweep log; reuses runs with saved metrics |
| PCA and GPR | `python scripts/step1_b1_gpr.py` | B1 runs and validation predictions; scikit-learn |
| PR-Net power penalty | `python scripts/step1b_passivity.py` | Passive/lossless comparisons and passivity report; selected registry already supplied |
| Validation explanations | `python scripts/step2_setup.py`; `python scripts/step2_xai.py PRNet`; repeat XAI for `M1` and `M0`; then `python scripts/step2_eval.py` | `results/step2/`; complete XAI requirements, potentially substantial runtime |
| Test prediction | `python scripts/step3_test.py` and `python scripts/step3_b1_test.py` | `results/step3/test/`; requires historical freeze record |
| Test explanation | `python scripts/step2_xai.py PRNet test`; repeat for `M1` and `M0`; `python scripts/step2_eval.py test` | `results/step3/attr/`, test XAI report |
| ADS reference jobs | `python scripts/step3_ads_jobs.py` | `results/step3/ads/jobs.json`, mapping; does not itself run ADS |
| ADS reference analysis | `python scripts/step3_ads_eval.py` then `python scripts/step3_report.py` | Uses cached ADS responses; vector fitting is recomputed by the first script |
| Post-hoc controls | `python scripts/step3b_controls.py` | `results/step3/controls.*`; explicitly exploratory |
| Tuning | `python scripts/step4_tune.py validation`, `python scripts/step4_tune.py test` | PR-Net search and ADS confirmation; existing preregistration; configured ADS Python path and license |
| Tuning summary | `python scripts/step4_report.py` | `results/step4/step4_report.*`; also accepts `validation` |
| Exhaustive exploratory tuning | `python scripts/step4b_explore.py PRNet`; repeat for `M0`; `python scripts/step4b_report.py` | `results/step4b/`; additional solver calls if caches are missing |

`step3_freeze.py` and `step4_prereg.py` deliberately refuse to overwrite their historical records. Preserve those records. A genuinely new study needs a new output/version arrangement and a new prospective protocol, rather than replacing dates in the original files.

## ADS batch execution

The original `scripts/ads_sim.py` expects jobs in this format:

```json
[
  {
    "id": "example_baseline",
    "geometry": {
      "L1": 138.5, "L2": 136.0, "S1": 3.3,
      "S2": 18.0, "Wc1": 25.5, "Wc2": 31.5
    }
  }
]
```

For the recorded installation:

```powershell
& 'E:\ADS2026\tools\python\python.exe' scripts/ads_sim.py results/step3/ads/jobs.json results/step3/ads --workers 4
```

Completed jobs are skipped. Failures are logged to `failures.jsonl`; check that every requested case exists before analysis. If concurrent licenses are exhausted, rerun with `--workers 1`. Parallelism affects throughput and license demand, not the scientific sample size. The original runner checks its baseline-netlist SHA256, solver completion, frequency grid and finite S-parameter arrays.

Paths in the archived scripts identify the original machine. For another installation, adjust paths in a separate working copy and record the change. The dataset generation/Momentum helper scripts can also reference original workspaces under `D:/Downloads/ADS_XAI_runs`; the release's analysis does not require those external workspaces when the stored evidence is used.

## Full wave stress test

Use the separately configured openEMS/CSXCAD environment. Commands for the original setup are:

```powershell
$env:CSXCAD_INSTALL_PATH = 'E:\openEMS'
& 'E:\anaconda3\envs\openems\python.exe' scripts/oems_mesh_study.py m1 m2 m3 m4
& 'E:\anaconda3\envs\openems\python.exe' scripts/oems_batch.py select
& 'E:\anaconda3\envs\openems\python.exe' scripts/oems_batch.py run m4
& 'E:\anaconda3\envs\openems\python.exe' scripts/oems_report.py m4
```

The selector walks preregistered test cases in order, retaining the first three ADS-successful SHAP cases with distinct parameter pairs. Its CLI supports `select`, `run` and `ads_jobs`. Despite the historical docstring's `report` example, report generation is through **`oems_report.py`**, not `oems_batch.py report`.

The paired ADS responses needed by `oems_report.py` are already included. For fresh pairs, generate `results/fullwave_openems/ads_jobs.json` with `oems_batch.py ads_jobs`, then run `ads_sim.py` in the ADS environment with that job file and `results/fullwave_openems/ads` as output.

`scripts/report_figures.py` recreates the six report figure PDFs from stored data. It imports `oems_filter.py`, so it requires the openEMS environment even though it launches no simulation. Running it replaces existing figure files; use a separate copy. Existing report figures are already packaged.

## Metric definitions

The hard mask violation is the sum of four nonnegative worst-case exceedances in dB: passband S21 below -0.5 dB, passband S11 above -10 dB, low-stopband S21 above -20 dB and high-stopband S21 above -15 dB. Regions are 11.6–12.4, 9–10.8 and 13.2–15 GHz respectively. Hard violation zero is pass on the sampled grid.

The smooth explanation target uses log-sum-exp/softplus with temperature 0.05 dB. Its absolute value and zeros need not equal those of the hard metric. Compare smooth to smooth or hard to hard when stating prediction error.

Model frequency/BW scores use the connected region around the sampled S21 peak at **peak minus 3 dB**. Historical full-wave summaries use a connected region above **absolute -3 dB**. Both definitions are reproduced in the audit. This affects the numerical bandwidth difference and some centre-frequency values, but not the separately computed specification-mask verdict.

The 512 model coalitions include nine inputs; ADS's 64 coalitions use six geometry inputs. The simulator never sees negative controls. Exact SHAP comparisons refer to the stated reference game; local slopes and path-integrated gradients have different definitions.

## Provenance and package verification

The final report and reflection are separate LaTeX projects in `report/final/` and `report/ai_use_reflection/`. Their compiled PDFs are included. To compile their `main.tex` files with Tectonic, run `tectonic main.tex` inside each project directory; BibTeX/reference passes are handled automatically. The checked compiler is Tectonic 0.17.0. First use may download its TeX bundle, so compile one project first if the format cache has not yet been initialized. The original draft source/PDF remain in the parent `report/` directory.

`data/filter_supplement_v2/artifact_hashes.json` covers the dataset release. `results/step3/freeze.json` records 30 historical hashes and the test protocol. Four current files differ from that snapshot as documented in plan section 12; the audit lists them. This gate is not an independently time-stamped public registration, and the unmodified historical source versions are not reconstructed in this release.

Build and verify release ZIPs with:

```powershell
python scripts/build_submission.py --output ../DSCI498_submission
python scripts/build_submission.py --verify ../DSCI498_submission/DSCI498_code.zip
python scripts/build_submission.py --verify ../DSCI498_submission/DSCI498_evidence.zip
```

The builder verifies every packaged file against its SHA256/size manifest. It does not alter or delete source experiments. Intermediate openEMS port/time-series files are omitted, while response arrays and summaries are retained. Complete native regeneration remains dependent on the separately installed solvers and original workspace assets described above.
