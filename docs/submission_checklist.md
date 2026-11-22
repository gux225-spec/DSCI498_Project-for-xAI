# Final submission requirements and evidence

Updated November 22, 2026. This matrix follows the supplied DSCI 498 final-project brief. It identifies the completed written material and separates it from submission or live presentation actions. The original draft is preserved.

## Report requirements

| Requirement | Final location | Status and scope |
|---|---|---|
| One AI model supporting a real decision | Abstract, Table I and Sections I and III | PR-Net is primary; M0/M1 are controls and B1 is a prediction baseline |
| Data Model Stakeholder Decision framework | Table I, page 1 | Each element is explicit |
| One explanation principle | Table I and Section V | Fixed-reference contrastive explanation |
| Framing question in the supplied template | Introduction, page 1 | Names stakeholder, decision, principle, decision-quality aim and faithfulness constraint |
| Q1 Are explanations true | Section VI, pages 3–4 | Model diagnostics and ADS-reference consistency; different estimands and numerical failures distinguished |
| Q2 Are explanations useful | Section VII, pages 5–6 | Preregistered tuning proxy plus practitioner feedback; human time/decision gains remain unmeasured |
| Q3 Do they hold up under unusual conditions | Section VIII, pages 5–7 | Circuit-to-full-wave stress test, three selected cases and modeling/numerical limits |
| Stakeholder evidence | Section III, page 2; follow-up interpretation in Section VII | Two voice interviews with the same engineer, November 5 and 21; edited summaries, not direct quotations or two independent participants |
| At least three explanation methods | Section V | Exact SHAP, adaptive Integrated Gradients and LIME-delta |
| Report where and why methods disagree | Section VI B, pages 3–4 | Target nonlinearity, method definitions, IG integration, LIME neighborhood and reference effects |
| Faithfulness test recorded before results | Table II, page 3; Section VI D, page 4 | Local freeze/protocol record, validation-development boundary and four current hash differences disclosed; no external timestamp claimed |
| Domain grounding before comparison | Section VI C, page 4 | Exact negative-control invariance, train-domain ranking expectation and fitted modal concepts; precise rankings not treated as universal physics |
| Related work | Section II, page 2 | Attribution, faithfulness evaluation and microwave surrogate/model-discrepancy context |
| Failed or abandoned explanations section | Section IX, pages 6–8 | Agreement heuristic, IG notch failure, reference dependence, step-size misuse and failed transfer retained |
| Report 6–8 pages in conference format | `report/final/DSCI498_FinalReport.pdf` | Compiled IEEE conference format, **8 pages including references**; all pages visually checked |
| Case-level explanation quality | Table IV, page 6 | test_0005 has same-target error, LIME local R2, IG residual/steps and SHAP additivity; diagnostic limits stated |
| Repository link and supporting records | Final report conclusion and repository README | Linked Git repository, runbook, decision log and result-verification script |

## Required files

| Deliverable | File or location | State |
|---|---|---|
| Final report | [8-page PDF](../report/final/DSCI498_FinalReport.pdf) and [TeX source](../report/final/main.tex) | Prepared and compiled; no final-source TODO placeholders |
| Code with README | [Repository](https://github.com/gux225-spec/DSCI498_Project-for-xAI), `src/`, `scripts/`, requirements and root README | Local commits and remote push; binary evidence is distributed separately |
| Decision log | [Decision log](decision_log.md) | Evidence-linked historical decisions plus current submission decisions; retrospective compilation explicitly dated |
| AI-use reflection 1–2 pages | [2-page PDF](../report/ai_use_reflection/DSCI498_AI_Use_Reflection.pdf) and [TeX](../report/ai_use_reflection/main.tex) | Uses confirmed ADS automation and technical-source assistance; explains checking and responsibility; author perspective is explicit |
| Reproducibility evidence | Separate `DSCI498_evidence.zip`, described in README | Full original dataset-manifest coverage, weights, saved predictions/attributions and cached responses |
| Original draft | `report/DSCI498_DraftReport.pdf` and `report/main.tex` | Preserved as historical artifacts; their TODOs are not the final report's outstanding work |
| Oral defense preparation | [Speaking outline and anticipated questions](oral_defense_notes.md) | Two five-minute ideas prepared; actual delivery and speaker allocation remain human actions |

## Remaining actions outside file preparation

- Upload the final report/reflection and provide code/evidence access through the course submission interface. The course page shows a deadline of **December 9, 2026 at 00:00**. This preparation does not submit to the course platform.
- Deliver the oral defense. The brief says 10 minutes and five minutes per member; use the actual team arrangement and confirm any timing ambiguity with the instructor.
- Keep access appropriate for the grader if the GitHub repository is private. No claim is made that the instructor already has access.

## Interpretation limits retained in the final

The project does not claim measured time savings, improved practitioner decision accuracy, fabricated-device validity, a population full-wave failure rate, or a universally certified frequency/bandwidth safety margin. The two interviews motivate use requirements and possible benefits. Relative and absolute bandwidth definitions are separated, and current case-level metrics replace the old population-median example.

Page counts, PDF hashes and final-source checks are in [final artifact checks](../results/submission_audit/final_artifacts.json). Archived calculation verification is in [the audit](../results/submission_audit/verification.md).
