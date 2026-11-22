# Quick guide

## The problem

A model predicts a filter's S21/S11 response from six geometry parameters. When a design violates a specification mask, an engineer needs to know which parameters to investigate. An explanation should help narrow that investigation while remaining consistent with the model and the simulator.

The code checks three questions:

1. **Is the explanation faithful?** Compare SHAP, Integrated Gradients and LIME with model diagnostics and ADS parameter interventions.
2. **Does it help select a change?** Select two parameters, search for a fix and confirm the candidate in ADS.
3. **Does the fix transfer?** Re-evaluate selected ADS-passing designs with openEMS full-wave simulation.

PR-Net is the primary model because its poles/residues support physical concept checks. M1 is a more accurate MLP comparison; M0 is an adapted literature architecture; B1 is PCA plus Gaussian-process regression.

## Script order

Run from the repository root after restoring the Drive archives. Use a separate working copy for new experiments: the original scripts can overwrite saved outputs.

| Stage | Entry points |
|---|---|
| Check inputs and fit modal concepts | `step0_split_checks.py`, `step0_vector_fit.py` |
| Train and select models on validation data | `step1_sweep.py`, `step1_b1_gpr.py`, `step1b_passivity.py` |
| Validation explanations | `step2_setup.py`, `step2_xai.py PRNet`, `step2_eval.py` |
| Saved test predictions | `step3_test.py`, `step3_b1_test.py` |
| Test explanations | `step2_xai.py PRNet test`, `step2_eval.py test` |
| ADS attribution reference | `step3_ads_jobs.py`, `ads_sim.py`, `step3_ads_eval.py` |
| Restricted tuning | `step4_tune.py test`, `step4_report.py` |
| Exploratory pair comparisons | `step4b_explore.py PRNet`, `step4b_report.py` |
| Full-wave check | `oems_mesh_study.py`, `oems_batch.py select`, `oems_batch.py run m4`, `oems_report.py m4` |

All entry points above are in `scripts/`. Repeat explanation runs for M1 and M0 when comparing models. Exact commands and solver setup are in `docs/reproducibility.md` after extracting the course-materials ZIP.

## Common issues

- **Missing arrays or checkpoints:** restore both data and results ZIPs. A Git clone contains code only.
- **Missing report/protocol files during verification:** also restore the course-materials ZIP.
- **ADS paths or license errors:** `ads_sim.py` uses `E:/ADS2026`; tuning scripts also contain the original ADS Python path. Configure these for your installation. Reduce `--workers` if licenses limit parallel jobs.
- **openEMS import errors:** use an environment with native openEMS and CSXCAD bindings; pip requirements do not install the solver.
- **Different bandwidth values:** model metrics use peak-relative −3 dB; old full-wave summaries use absolute −3 dB. Compare matching definitions.
- **Deep-notch IG instability:** inspect integration residuals; for LIME, inspect local R². High aggregate agreement is not a confidence score for one design.
- **Historical freeze differences:** four current protocol/path files differ from their recorded hashes. The archived audit documents them. Preserve the old freeze/preregistration records rather than recreating their timestamps.
