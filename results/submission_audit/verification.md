# Archived result verification

Recorded UTC: 2026-11-22T16:50:18.225802+00:00

Status: **verified_with_documented_caveats**

Archived-data recomputation and optional sample inference; no retraining or new simulation

## Recomputed test accuracy

| Model | S21 MAE dB | S11 MAE dB | f0 MAE MHz | BW R2 | f0 and BW designs |
|---|---:|---:|---:|---:|---:|
| PRNet | 0.083928 | 0.078133 | 2.728483 | 0.910204 | 202 |
| M1 | 0.021145 | 0.025027 | 0.374020 | 0.999872 | 202 |
| M0 | 0.266388 | 0.811044 | 29.296504 | -0.342229 | 202 |
| B1 | 0.131020 | 0.130505 | 5.884942 | 0.869247 | 202 |

Curve errors use all 216 test designs; frequency and bandwidth errors use the valid single-band subset.

## Recomputed tuning results

| Method | Success | ADS calls including reference attribution |
|---|---:|---:|
| shap_ref | 25/30 | 65 |
| ig | 25/30 | 70 |
| lime_delta | 24/30 | 74 |
| random | 6/30 | 141 |
| grad_x_span | 23/30 | 78 |
| lime_slope_x_span | 23/30 | 76 |
| ads_true_shap | 24/30 | 1992 |
| all6_surrogate | 30/30 | 30 |
| shap_revert_no_search | 17/30 | 30 |
| ads_coordinate | 30/30 | 360 |

Calls follow the original per-method accounting, including shared cached trajectories as independent method calls; these are not unique new solver launches.

## Findings and retained caveats

- **WARNING — Historical freeze is not an immutable current snapshot**: Four historical hash differences are documented in plan section 12; original prior versions are not reconstructed.
- **WARNING — Bandwidth conventions require separate labels**: Historical model metrics use peak-relative -3 dB; full-wave summaries use absolute -3 dB. Both are retained and recomputed here; the hard-mask verdict is unaffected.
- **WARNING — Draft still contains TODOs and is not the final report**: {'lines': [34, 285, 295, 400]}
- **WARNING — Historical violation-error statistic mixes smooth and hard targets**: step4b_report.py compares surrogate smooth violation with ADS hard violation. Its 0.062 dB median is not a same-target prediction error; the case card also uses a population median for one design.

## Full-wave bandwidth definitions

| Tuned design | ADS hard V dB | openEMS hard V dB | Relative bandwidth reduction MHz |
|---|---:|---:|---:|
| test_0005 | 0.000000 | 10.194002 | 80.100 |
| test_0015 | 0.000000 | 7.284766 | 82.343 |
| test_0034 | 0.000000 | 4.998917 | 86.865 |

## Scope limits

The following were outside this verification run:

- Model retraining
- New ADS/openEMS/Momentum simulations
- Vector-fitting refits
- LaTeX compilation
- Practitioner performance study

## Detailed checks

| Check | Status |
|---|---|
| Python source parses | pass |
| Dataset release SHA256 manifest | pass |
| Frozen model weights and other unchanged artifacts | pass |
| Historical freeze is not an immutable current snapshot | warning |
| train shapes and finite values | pass |
| train unique design IDs | pass |
| train frequency grid | pass |
| train dB derived from complex response | pass |
| validation shapes and finite values | pass |
| validation unique design IDs | pass |
| validation frequency grid | pass |
| validation dB derived from complex response | pass |
| test shapes and finite values | pass |
| test unique design IDs | pass |
| test frequency grid | pass |
| test dB derived from complex response | pass |
| train/validation disjoint designs and geometry | pass |
| train/test disjoint designs and geometry | pass |
| validation/test disjoint designs and geometry | pass |
| Train normalization | pass |
| Default test access does not return labels | pass |
| PRNet_lam0.3_s2026_b16_lossless10.0 prediction IDs | pass |
| PRNet_lam0.3_s2026_b16_lossless10.0 saved S-parameter MAE | pass |
| PRNet_lam0.3_s2026_b16_lossless10.0 frequency and bandwidth metrics | pass |
| PRNet_lam0.3_s2027_b16_lossless10.0 prediction IDs | pass |
| PRNet_lam0.3_s2027_b16_lossless10.0 saved S-parameter MAE | pass |
| PRNet_lam0.3_s2027_b16_lossless10.0 frequency and bandwidth metrics | pass |
| PRNet_lam0.3_s2028_b16_lossless10.0 prediction IDs | pass |
| PRNet_lam0.3_s2028_b16_lossless10.0 saved S-parameter MAE | pass |
| PRNet_lam0.3_s2028_b16_lossless10.0 frequency and bandwidth metrics | pass |
| M1_k4_lam0.1_s2026 prediction IDs | pass |
| M1_k4_lam0.1_s2026 saved S-parameter MAE | pass |
| M1_k4_lam0.1_s2026 frequency and bandwidth metrics | pass |
| M1_k4_lam0.1_s2027 prediction IDs | pass |
| M1_k4_lam0.1_s2027 saved S-parameter MAE | pass |
| M1_k4_lam0.1_s2027 frequency and bandwidth metrics | pass |
| M1_k4_lam0.1_s2028 prediction IDs | pass |
| M1_k4_lam0.1_s2028 saved S-parameter MAE | pass |
| M1_k4_lam0.1_s2028 frequency and bandwidth metrics | pass |
| M0_s2026 prediction IDs | pass |
| M0_s2026 saved S-parameter MAE | pass |
| M0_s2026 frequency and bandwidth metrics | pass |
| M0_s2027 prediction IDs | pass |
| M0_s2027 saved S-parameter MAE | pass |
| M0_s2027 frequency and bandwidth metrics | pass |
| M0_s2028 prediction IDs | pass |
| M0_s2028 saved S-parameter MAE | pass |
| M0_s2028 frequency and bandwidth metrics | pass |
| B1_r32 prediction IDs | pass |
| B1_r32 saved S-parameter MAE | pass |
| Frequency metric denominator disclosed | pass |
| ADS curve Shapley rebuilt from coalition values | pass |
| ADS Shapley efficiency | pass |
| ADS mapping agrees with truth array IDs | pass |
| All 80 ADS coalition tables available | pass |
| Full ADS coalitions reproduce original responses | pass |
| ADS target tables recomputed from cached complex responses | pass |
| PRNet_lam0.3_s2026_b16_lossless10.0 exact SHAP rebuilt | pass |
| PRNet_lam0.3_s2027_b16_lossless10.0 exact SHAP rebuilt | pass |
| PRNet_lam0.3_s2028_b16_lossless10.0 exact SHAP rebuilt | pass |
| PRNet test SHAP vs ADS curve rank correlations | pass |
| M1_k4_lam0.1_s2026 exact SHAP rebuilt | pass |
| M1_k4_lam0.1_s2027 exact SHAP rebuilt | pass |
| M1_k4_lam0.1_s2028 exact SHAP rebuilt | pass |
| M1 test SHAP vs ADS curve rank correlations | pass |
| M0_s2026 exact SHAP rebuilt | pass |
| M0_s2027 exact SHAP rebuilt | pass |
| M0_s2028 exact SHAP rebuilt | pass |
| M0 test SHAP vs ADS curve rank correlations | pass |
| Shapley reference scope disclosed | pass |
| Tuning cases agree with preregistration | pass |
| shap_ref success and call totals | pass |
| ig success and call totals | pass |
| lime_delta success and call totals | pass |
| random success and call totals | pass |
| grad_x_span success and call totals | pass |
| lime_slope_x_span success and call totals | pass |
| ads_true_shap success and call totals | pass |
| all6_surrogate success and call totals | pass |
| shap_revert_no_search success and call totals | pass |
| ads_coordinate success and call totals | pass |
| shap_ref primary statistics | pass |
| ig primary statistics | pass |
| lime_delta primary statistics | pass |
| PRNet exhaustive parameter pair count | pass |
| PRNet exploratory baselines | pass |
| M0 exhaustive parameter pair count | pass |
| M0 exploratory baselines | pass |
| test_0005 x0 ads mask and archived absolute bandwidth | pass |
| test_0005 x0 oems mask and archived absolute bandwidth | pass |
| test_0005 tuned ads mask and archived absolute bandwidth | pass |
| test_0005 tuned oems mask and archived absolute bandwidth | pass |
| test_0015 x0 ads mask and archived absolute bandwidth | pass |
| test_0015 x0 oems mask and archived absolute bandwidth | pass |
| test_0015 tuned ads mask and archived absolute bandwidth | pass |
| test_0015 tuned oems mask and archived absolute bandwidth | pass |
| test_0034 x0 ads mask and archived absolute bandwidth | pass |
| test_0034 x0 oems mask and archived absolute bandwidth | pass |
| test_0034 tuned ads mask and archived absolute bandwidth | pass |
| test_0034 tuned oems mask and archived absolute bandwidth | pass |
| Three ADS-passing tuned cases fail archived full-wave mask | pass |
| Bandwidth conventions require separate labels | warning |
| LaTeX citation keys resolve | pass |
| LaTeX figure files exist | pass |
| Draft still contains TODOs and is not the final report | warning |
| Historical violation-error statistic mixes smooth and hard targets | warning |
| Conclusion and generalization limits | pass |
| PRNet current-weight sample inference | pass |
| M1 current-weight sample inference | pass |
| M0 current-weight sample inference | pass |

Full numeric values, settings, environment and check details are in `verification.json` alongside this file.
