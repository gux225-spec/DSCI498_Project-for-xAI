# Filter explanation project decision log

Compiled on **November 22, 2026**, from retained project artifacts and the author's subsequently supplied interview information. This is a retrospective consolidation, not a claim that this document existed before the experiments.

## Evidence and timing

Recorded event dates below come from dated project records, local freeze/preregistration timestamps, or the author's explicit account. The local timestamps are not independent public registrations. The analysis plan was extended with results after its initial date; its current contents are not automatically evidence of what every earlier version contained. When an exact decision date cannot be established, it is left unspecified.

The audit and package manifests are current integrity checks. They preserve the original freeze/preregistration records rather than replacing their dates or hashes.

## Dataset and model decisions

### D01 Use design level splits and train only scaling

**Timing:** dataset release dated October 24, 2026; consolidated here retrospectively.

**Decision:** use the 1,440-design v2 release with fixed 1008/216/216 train/validation/test membership. Keep all frequency samples of one geometry in one split, use training-only scalers, and treat synthetic negative controls as checks rather than physical inputs.

**Reason:** frequency-wise splitting would mix responses from the same design across development and evaluation. The six geometry variables and planted controls permit distinct tests of response prediction and irrelevant-input dependence. The release combines 480 legacy designs on a fixed-width slice with 960 new six-dimensional designs; it is not a single 1,440-point six-dimensional LHS.

**Evidence:** [release plan](../data/filter_supplement_v2/plan.json), [split manifest](../data/filter_supplement_v2/split_manifest.json), [train scalers](../data/filter_supplement_v2/scalers_train_only.json), [original split checks](../results/step0/split_checks.json). Current checks confirm unique/disjoint designs and geometry.

### D02 Use complex responses and five conjugate pole pairs for PR Net

**Timing:** model reasoning recorded in the October 24–25 plan; exact time of the structural choice is not independently established here.

**Decision:** predict complex S21/S11 with a pole/residue representation and five shared conjugate pole pairs, deriving dB afterward. Compare with a frequency-conditioned complex MLP, an adapted direct-dB M0 and PCA/GPR.

**Reason:** deep dB notches are difficult direct regression targets. Train-side representation checks and vector fitting supported a compact rational representation. Poles were learned end-to-end, allowing later checks against independently fitted concepts. PR-Net remained primary for concept analysis once it passed validation gates, even though M1 was more accurate.

**Evidence:** [plan sections 2.1–2.5](algorithm_plan_v2.md), [vector-fitting summary](../results/step0/vf_summary.json), [model code](../src/filterxai/models.py), [validation report](../results/step1/validation_report.md). The final registry also retains B1; optional B2 and proposed extra control-retraining experiments are not claimed as completed.

### D03 Extend training and adopt a soft power balance penalty

**Timing:** validation-stage revisions recorded October 25, 2026.

**Decision:** allow up to 2,000 epochs with patience 50 and plateau learning-rate reduction; select validation hyperparameters before test evaluation. Adopt PR-Net's lossless power-balance penalty with mu=10, lambda=0.3 and batch 16 after validation comparisons across three seeds.

**Reason:** the original training schedule caused early stopping before convergence and uneven update budgets. The selected soft penalty improved power balance while meeting the predefined validation eligibility rule. It is a data-specific soft loss, not a general guarantee of passivity or an appropriate lossless constraint for radiating full-wave structures.

**Evidence:** [plan section 2.4 and 10.1](algorithm_plan_v2.md), [penalty report](../results/step1/passivity_report.md), [penalty driver and eligibility rule](../scripts/step1b_passivity.py), [final registry](../results/step1/final_models.json). Exact prior plan snapshots are not reconstructed.

## Explanation and evaluation decisions

### D04 State the target and reference for each explanation

**Timing:** settings appear in the plan/setup used before the October 25 test freeze. The later stakeholder-oriented framing is consolidated retrospectively.

**Decision:** use the ADS baseline geometry with controls at the training mean as the primary reference. Explain defined scalar response targets, with smooth mask violation as the main nonlinear target. Compare exact 512-coalition SHAP, adaptive IG and bounded-neighborhood LIME slopes multiplied by reference displacement.

**Reason:** the reference specifies the contrast being explained; the target supplies engineering units. Background changes answer a different comparison. LIME-delta is not an exact additive decomposition, and IG's path allocation is not identical to Shapley allocation.

**Evidence:** [setup](../results/step2/setup.json), [target definitions](../src/filterxai/targets.py), [explanation implementations](../src/filterxai/xai.py), [XAI report](../results/step2/xai_report.md).

### D05 Freeze configuration before explicit test label access

**Timing:** local record `2026-10-25T17:21:06.622851+00:00`.

**Decision:** record hashes of 30 artifacts and the test protocol before the recorded test-label access. Reuse validation settings for test accuracy/explanations; use ADS's 64 geometry coalitions to distinguish model-game agreement from explainer agreement.

**Reason:** validation results had already been used for development. A held-out test stage and a recorded protocol make that boundary inspectable. A simulator-defined Shapley reference gives an exact comparison for the selected target/reference game, not universal truth about every attribution definition or real hardware.

**Evidence:** [freeze record](../results/step3/freeze.json), [freeze writer](../scripts/step3_freeze.py), [data gate](../src/filterxai/data.py), [ADS evaluation](../results/step3/ads_eval.json). The gate checks that the file exists; historical hash agreement is checked separately by the release verifier.

**Current integrity qualification:** 26 of 30 files match, including all nine neural weights. Current differences are `docs/algorithm_plan_v2.md`, `docs/training_xai_protocol_v2.json`, `scripts/step2_xai.py` and `scripts/step2_eval.py`. Plan section 12 records test-path adaptations and exact reproduction of a validation rerun; the evolving plan/protocol also contain later results. Without original snapshots, this release cannot independently reconstruct all historical differences. See [current audit](../results/submission_audit/verification.json).

### D06 Keep failed and ambiguous explanation checks visible

**Timing:** recorded in step 2/3 reports after their respective results; retrospectively consolidated.

**Decision:** retain IG completeness failures near deep S11 dB notches, LIME local-fit diagnostics, background sensitivity and the weak relationship between SHAP–LIME agreement and ADS-reference agreement. Treat deletion/insertion as supporting evidence because its pre-run version cannot be verified. Label the OLS/secondary-feature/bootstrap checks as post-hoc.

**Reason:** a high aggregate score cannot validate one case. Method-definition differences, numerical integration failure and model error are distinct. The project should expose limitations rather than convert every disagreement with Shapley into explainer error.

**Evidence:** [validation XAI](../results/step2/xai_report.md), [test XAI](../results/step3/xai_report_test.md), [ADS reference comparisons](../results/step3/ads_eval.json), [post-hoc controls](../results/step3/controls.md).

### D07 Preregister tuning cases and retain strong baselines

**Timing:** local record `2026-10-25T19:23:08.868157+00:00`; later exhaustive-pair analysis is explicitly exploratory.

**Decision:** select 30 failing single-band test designs with seed 2026; compare explainer-selected pairs with random pairs, sensitivity selectors, ADS Shapley, revert-to-reference, six-parameter surrogate search and ADS coordinate search. Use the recorded search/call budgets and Holm-corrected primary comparisons. Identical subspaces share trajectories, with per-method call accounting.

**Reason:** attribution does not specify a step size, and parameter selection must be tested with the same search/confirmation procedure. The six-parameter baseline prevents overstating explanation benefits. The later 15-pair enumeration tests how much the original random comparison hides.

**Evidence:** [preregistration](../results/step4/prereg.json), [original tuning results](../results/step4/test/results.json), [primary report](../results/step4/step4_report.md), [exploratory report](../results/step4b/step4b_report.md). The current plan hash no longer equals the historical preregistered plan hash because the plan evolved; keep the original recorded hash.

**Outcome affecting interpretation:** SHAP 25/30 versus random 6/30, while all-six search reaches 30/30; fixed L1+L2 reaches 21/30 in the exploratory comparison. This supports localization under a two-parameter restriction, not faster optimization or measured human benefit.

## Stakeholder and full wave decisions

### D08 Frame the stakeholder as a receiver troubleshooting engineer

**Timing:** original interview dated November 5, 2026; exact reframing time is not separately established.

**Decision:** use the available practitioner's receive-side evaluation/troubleshooting role. Present the failed specification, frequency range and violation first; supply case-level method diagnostics; separate diagnostic attribution from geometry-change proposals that need a component designer and validation.

**Reason:** this stakeholder does not directly edit microstrip geometry. Ranked geometry explanations can support a handoff, but a simulator tuning experiment is only a proxy for that decision.

**Evidence:** [original interview paraphrase](stakeholder_interview_notes_en.md), [progress talk notes](progress_talk_5min.md). Numerical examples are not documented field measurements. The interview is one person's account, not a user performance study.

### D09 Replace the planned HFSS checks with openEMS

**Timing:** exact decision date unavailable; the substitution and results are documented in planning material dated November 17, 2026.

**Decision:** use openEMS EC-FDTD for independent full-wave reevaluation when the planned HFSS license was unavailable. Reproduce the diagnostic layout/substrate assumptions, compare the baseline with Momentum, refine the baseline mesh and use m4 for selected cases.

**Reason:** the question is whether circuit-confirmed tuning transfers across modeling levels. An alternative full-wave solver provides a feasible stress test, with Momentum supplying an additional numerical comparison.

**Evidence:** [full-wave report](../results/fullwave_openems/report.md), [solver implementation](../scripts/oems_filter.py), [mesh settings](../scripts/oems_mesh_study.py), [dataset EM diagnostics](../data/filter_supplement_v2/em_diagnostics/).

**Qualification:** numerical-method descriptions aid interpretation but do not isolate a unique physical cause. Baseline mesh refinement does not prove every tuned geometry converged; port, boundary and bandwidth-definition differences remain relevant. Missing two-port power alone is insufficient to establish that all missing power is radiation.

### D10 Fix a reproducible full wave case rule

**Timing:** retained rule says it preceded case simulation; no independent timestamp proves that ordering.

**Decision:** walk the preregistered cases in order and retain the first three ADS-successful SHAP cases with distinct selected parameter pairs. Simulate both original and first ADS-passing geometry; retain failures.

**Reason:** this supplies distinct parameter-pair examples within a limited full-wave budget. It is not random sampling of all designs and cannot estimate a population failure rate.

**Evidence:** [case rule and geometry](../results/fullwave_openems/cases.json), [selection code](../scripts/oems_batch.py), [paired results](../results/fullwave_openems/report.json). All three tuned designs fail the archived full-wave mask; two improve and one worsens.

### D11 Record follow up workflow feedback and AI assistance

**Timing:** follow-up interview November 21, 2026, online voice call; details supplied by the author on November 22.

**Decision:** retain the same practitioner's account of screening, verification, adoption/rejection, responsibility and four potential benefits. Record AI assistance with ADS Python Console batch/parallel simulation and with finding/explaining simulator algorithms and mathematical foundations.

**Reason:** the account clarifies the required evidence and action boundary. Anticipated search-space compression, earlier rejection, communication and judgment calibration are useful design motivations, with independent checks as conditions for use.

**Evidence:** [Chinese follow-up summary](stakeholder_interview_ai_workflow_summary_zh.md), [confirmed AI-use notes](ai_use_reflection_notes.md).

**Qualification:** this is the same person interviewed twice. Benefits and time comparisons are practitioner descriptions, not measured project gains. Slide 8 discussion does not establish that the participant operated the case card. The AI-use notes are preparation material, not the finished 1–2 page reflection.

## Submission preparation

### D12 Preserve experiments and add a separate current audit

**Timing:** November 22, 2026.

**Decision:** keep original source/experiment/report files intact, update the project README, add requirements and this log, and generate a new read-only audit plus complementary code/evidence archives. Keep GitHub publication deferred as requested by the author.

**Reason:** reviewers need a runnable entry point, asset completeness and explicit provenance without retroactively changing historical evidence. Packaging includes every dataset-manifest file, checkpoint and saved response used for recomputation; it excludes generated caches/solver working outputs and intermediate openEMS time series.

**Evidence:** [verifier](../scripts/verify_submission.py), [packager](../scripts/build_submission.py), [runbook](reproducibility.md), [audit](../results/submission_audit/verification.md). Archive file-size/SHA256 manifests establish the release's current byte-level contents.

**Retained issues:** relative versus absolute bandwidth definitions; mixed smooth/hard violation-error statistic; draft TODOs/PDF-source mismatch; four historical hash differences. These are disclosed rather than silently rewriting old metrics or timestamps. Final report/reflection authoring remains a separate step.

### D13 Publish the selected code and complete the final written materials

**Timing:** November 22, 2026, following the author's explicit request to upload with local Git and complete the requirements.

**Decision:** initialize the local repository with the supplied commit email, push selected source/documentation and lightweight results to [the author-provided repository](https://github.com/gux225-spec/DSCI498_Project-for-xAI), and preserve exact archived line endings through `.gitattributes`. Keep large binary evidence separate from Git history. Create independent final-report and AI-reflection projects, leaving the original draft unchanged.

**Reason:** the final submission needs inspectable code, a working repository link, a 6–8 page conference report, a 1–2 page AI-use reflection and an explicit requirements mapping. Completed final materials replace preparation placeholders without altering original experiment evidence. Historical bandwidth definitions and mixed-metric records remain disclosed; the final report uses correctly qualified and case-specific values.

**Evidence:** [requirements matrix](submission_checklist.md), [final report](../report/final/DSCI498_FinalReport.pdf), [AI-use reflection](../report/ai_use_reflection/DSCI498_AI_Use_Reflection.pdf), [compiled artifact checks](../results/submission_audit/final_artifacts.json), and the remote Git history. The report is 8 pages and reflection 2 pages; all rendered pages were inspected. The initial selected-code commit is `bbe8ba8`; subsequent commits contain the completed materials. Binary evidence is provided through a separate submission release, with file-level and archive checksums; the third-party full-text working reference is omitted while citations remain. The live oral defense and course-platform submission remain actions for the author.
