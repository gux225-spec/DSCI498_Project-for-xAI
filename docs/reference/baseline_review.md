# ADS baseline filter review

Readiness: **Share with caveats**. Review completeness: **Complete for the supplied CSVs and static plot, plus the retrieved baseline netlist**. Original CSVs and plot were preserved. No ADS sweep or EM simulation was run. The original extraction source code, full complex two-port S matrix, and physical measurements were unavailable.

## Verified values

The S-parameter CSV has 601 finite rows, strictly increasing frequency from 9 to 15 GHz at 0.01 GHz intervals. No missing or duplicate frequencies were found. Standard-library recomputation of `20 log10(hypot(S21_real,S21_imag))` agrees with `S21_dB` within 1.5e-14 dB. A second calculation using NumPy/pandas agrees to within 4e-13 dB; the difference is floating-point parsing/rounding.

| Quantity | Recalculated result | Meaning |
|---|---:|---|
| Sampled peak frequency | 11.620 GHz | Frequency of the highest sampled S21 value |
| Peak S21 | -0.00000379940753585 dB | Essentially 0 dB under the ideal loss settings |
| Lower relative -3 dB edge | 11.4083705554 GHz | Linear interpolation between bracketing samples |
| Upper relative -3 dB edge | 12.6951056272 GHz | Linear interpolation between bracketing samples |
| Arithmetic center | 12.0517380913 GHz | `(f_lower+f_upper)/2` |
| Geometric center | 12.0345531381 GHz | `sqrt(f_lower*f_upper)`, a separate convention |
| Bandwidth | 1.28673507177 GHz | `f_upper-f_lower` |
| Fractional bandwidth | 10.6767593357% | Bandwidth / arithmetic center |
| Best sampled insertion loss | 0.00000379940753585 dB | `-max(S21_dB)` |
| IL at arithmetic center | approximately 0.03109490 dB | Interpolation in dB; not a new ADS point |

All four values in the supplied metrics CSV exactly match the independent standard-library recomputation. This establishes numerical agreement, not proof of which algorithm the unavailable original extraction script used. The results are consistent with a threshold of `max(S21_dB)-3.000 dB`, linear crossing interpolation, arithmetic band center, and best sampled IL.

The arithmetic-center convention is also used in Keysight's bandwidth-search documentation: https://helpfiles.keysight.com/csg/e5063a/measurement/analyzing_data/determining_the_bandwidth_of_the_trace_bandwidth_search.htm

Absolute -3 dB gives bandwidth 1.28673481995 GHz, virtually identical here because the peak loss is tiny. Exact half-power (-3.01029995664 dB relative to peak) gives 1.28741772952 GHz. Keep the chosen threshold and center convention in the dataset schema rather than silently changing it between runs. The displayed precision does not imply GHz measurements accurate to all shown digits; the ADS grid is 10 MHz and convergence has not been established by finer simulation.

## Findings and proposed changes

1. **Use band center as the center-frequency target.** The three sampled local maxima are 11.62, 11.98, and 12.46 GHz. Their S21 values differ by only about 2.2e-6 dB. A winner selected by `argmax` can change peak branches as parameters or grid alignment change. Thinning the existing grid from 10 to 50 MHz changes the sampled winning peak from 11.62 to 12.45 GHz while arithmetic center changes by only about 0.262 MHz. This is a resampling diagnostic, not a finer ADS convergence test. **Fix (Proposed):** retain `f_peak_GHz` for diagnostics; train center-frequency models on an explicitly named band-center quantity.

2. **Define insertion loss by location and physical model.** The actual netlist has `Cond=1e306`, `TanD=0`, and `Rough=0`. The CSV satisfies `|S11|^2+|S21|^2=1` to approximately 1.7e-15. This supports a lossless response for port-1 excitation; it does not establish full two-port passivity or reciprocity. The near-zero best IL is unsuitable as the principal loss-regression target in this dataset. **Fix (Proposed):** use f0 and BW as initial targets, then matching, ripple, and specified stopband rejection. For practical IL prediction, use documented conductor/substrate loss settings and actual copper thickness, followed by a separate lossy dataset and selected EM validation. Do not invent a laminate loss tangent without its material data.

3. **Match the input features to the actual hierarchy and simulator.** The retrieved netlist instantiates `halffil` twice in `cmstpfil`, with reversed terminal orientation. Editing the shared halffil design changes both halves. `CLin1` and `CLin2` are MCFIL components; their electrical simulation uses W, S, and L. Their UI W1/W2 values do not appear in this circuit netlist. Keysight documents W1/W2 as layout connection widths: https://edadownload.software.keysight.com/eedl/ads/2011_01/pdf/ccdist.pdf (printed page 48). **Fix (Proposed):** use the six W/S/L inputs below, preserve symmetry, and keep layout-only W1/W2 out of the circuit-surrogate feature set. A later layout/EM surrogate can have a different feature schema.

4. **The plot agrees with the CSV, but should expose the training definitions.** Its dashed line is correctly labeled Peak: 11.620 GHz. S21 forms a connected bandpass, and S11 notches align with the three near-unity transmission peaks. There is no numerical reason to relabel that dashed line as f0. **Fix (Proposed):** add separate f0 and edge markers and a passband zoom. Between the outer sampled transmission peaks (11.62-12.46 GHz), sampled ripple is approximately 0.21404 dB and worst return loss approximately 13.1795 dB. That interval is a descriptive baseline region, not a user-supplied passband specification. Deep individual S11 notches do not establish good matching across the full band.

## Next sweep

This filter is suitable as a small, interpretable, lossless circuit-surrogate benchmark. It has six meaningful geometry controls and both frequency and coupling behavior to explain. A single design does not establish learnability across the domain, and the circuit model alone cannot establish fabricated-filter performance or publication novelty.

These ranges are proposed local exploration bounds, not manufacturing tolerances or validated feasibility limits.

| Parameter in halffil | Baseline (mil) | Proposed range (mil) |
|---|---:|---:|
| CLin1.L | 138.5 | 131.575-145.425 |
| CLin2.L | 136.0 | 129.2-142.8 |
| CLin1.S | 3.3 | 2.64-3.96 |
| CLin2.S | 18.0 | 14.4-21.6 |
| CLin1.W | 25.5 | 22.95-28.05 |
| CLin2.W | 31.5 | 28.35-34.65 |

Keep the substrate and TL1 fixed initially. Lengths are expected primarily to move frequency; spacing primarily changes coupling and bandwidth; width affects impedance, matching, and effective permittivity. These are physical hypotheses for the diagnostic sweep to check, not conclusions established from one baseline.

1. Run 25 OFAT diagnostic designs: one baseline plus four non-baseline levels for each of the six parameters. Restore the baseline before each change. Verify requested versus netlisted values and watch for missing crossings or new bands. If a supposedly active parameter produces an identical curve, inspect netlisting and stale outputs before dropping the feature.
2. If that domain behaves properly, run 256 Latin-hypercube training/validation designs (seed 42) and 64 independent held-out designs (seed 31415). This is a starting sample budget; use learning curves and error localization to decide expansion. The supplied JSON contains all 345 proposed designs and marks them `not_run`.
3. Keep the baseline 9-15 GHz / 10 MHz grid for comparability. A proposed label-search window is 10.5-13.5 GHz for this local sweep, to be checked during diagnostics. Both -3 dB edges must be bracketed. Return an invalid/truncated status, not fabricated band edges, when they are missing. Keep multi-band responses flagged. Re-simulate representative baseline/extreme designs with 1-2 MHz spacing around the relevant band edges/peaks to quantify convergence; extend the simulation window if responses shift outside it.

## Implementation and XAI

`scripts/filter_tools.py` uses only the Python standard library. Its default root is `D:/Downloads/ADS_XAI_runs`. It validates the baseline and creates the sweep proposal; it does not modify ADS designs or launch simulations.

In an ADS Python console, load it without assuming an undocumented ADS setter API:

```python
import runpy
from pathlib import Path

root = Path(r'D:\Downloads\ADS_XAI_runs')
tools = runpy.run_path(str(root / 'scripts' / 'filter_tools.py'))
result = tools['extract'](tools['read_csv'](root / 'baseline' / 'sparams_baseline.csv'))
plan = tools['make_plan'](root)
```

For command-line use, `python scripts/filter_tools.py validate` writes the validation JSON, and `python scripts/filter_tools.py plan` writes the plan JSON. Run these from the project root, or provide the full script path. `--root` overrides the default. For a non-baseline CSV use `validate --csv <path>`; specify `--metrics <path>` only when the comparison metrics belong to that design.

Connect the plan to the ADS mutation/netlist/simulation workflow already working in the original project. Before a batch, verify the actual API names in the installed ADS version. For each design:

1. Start from baseline geometry; apply the six parameters to `mw_filter_lib:halffil:schematic`, preserving units and symmetry.
2. Save/netlist `mw_filter_lib:spar_sim:schematic`. Check that the emitted MCFIL W/S/L values equal the requested inputs.
3. Run into a unique `sweeps/<design_id>/` directory; never reuse a stale baseline dataset after a failed simulation.
4. Save netlist, solver log/status, requested and actual parameter values, frequency grid, port impedances, material model, ADS/model version, source hashes, and full complex S11/S21/S12/S22 (or Touchstone).
5. Extract labels only from a completed run and retain invalid responses with reason codes. Keep all new files beneath the project root.

Start with separate small regressors for f0 and BW, for example ExtraTrees, and compare a simple linear or quadratic baseline. Report held-out MAE for f0 and BW in MHz; for near-constant IL, R-squared is not informative. If training a frequency-conditioned response model, split by geometry/design_id before expanding frequency rows, and keep all replicates of a geometry in the same split. Use the 64-design holdout only after tuning.

Compute permutation importance or SHAP only after satisfactory held-out prediction. Compare explanations with the OFAT slopes and selected ADS reruns; explanations of a surrogate are not independent physical proofs. Keep parameters independently sampled for this first study; if later tying or correlating dimensions, use grouped/conditional explanations and state the sampling domain. Scikit-learn explicitly cautions that importances depend on predictive quality and that correlated features can distort ordinary permutation importance: https://scikit-learn.org/stable/modules/permutation_importance

## Review coverage

Counts below refer to scoped source/component/question inventory, not universal verification. Categories overlap and should not be combined into an accuracy percentage. No numerical corrections to source files were required. Important proposed schema/model changes above remain proposals.

### Artifact quality

| Category | Observed defects | Assessment |
|---|---:|---|
| Usefulness/completeness | 0 / 3 | Supplied evidence answers the three extraction questions; project suitability and sweep proposals are bounded by the circuit model. |
| Analytical clarity | 0 / 1 | The original figure explicitly identifies a peak. Separate band-center/edge markers would improve reuse. |
| Visual/interaction consistency | 0 / 1 | Original static plot inspected; axes and legend agree. Interactive behavior is inapplicable. |

### Analytical correctness and robustness

| Category | Observed defects | Assessment |
|---|---:|---|
| Source authority/confidence | 0 / 4 | Original S-parameter CSV, metrics CSV, plot and local netlist inspected. Uploaded CSVs/plot hash-match the D-drive copies. |
| Value accuracy | 0 / 4 | Peak frequency, band center, bandwidth and best IL independently recomputed. Original extraction implementation unavailable. |
| Within-chart agreement | 0 / 3 | S21 shape, S11 notches and peak annotation agree with CSV samples. |
| Complete source details | N/A | Original artifact is a static PNG and CSV export, with no interactive source-detail surface. Definitions and caveats are supplied in this review. |
| Cross-artifact consistency | 0 / 2 | CSV-to-metrics comparison and CSV-to-plot comparison passed within their stated scopes. |
| Data-quality controls | 0 / 1 | All CSV rows checked for finite values, increasing unique frequencies, complex/dB agreement and port-1 power balance. |
| Conclusion support | 0 / 3 | Extraction, ideal-model suitability and hierarchy/feature conclusions supported. EM accuracy, actual fabrication performance and original script internals unverified. |

## File location

The D-drive project contains the complete eight-file baseline set and the new scripts/analysis/sweep proposal. The baseline files were preserved. Windows initially blocked the move because files were in use; subsequent automated approval review blocked cleanup of the remaining C-drive files, giving only `blocked by policy`. Three files and their directories remain at the old location: `baseline/spar_sim.ds`, `baseline/spectra.raw`, and `baseline/spar_sim_data/logFile.txt`. Their D-drive copies were SHA-256 matched before the cleanup attempt. The relocation is therefore complete as a usable D-drive copy, but not as removal of the old folder. Do not use the old path for new runs.
