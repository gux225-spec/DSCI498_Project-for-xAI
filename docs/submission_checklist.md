# Final submission checklist (DSCI 498 Final Project brief)

Updated 2026-11-17. Compares `report/main.tex` with the brief clause by clause. ✅ = done in the draft; ⚠️ = partly done or a risk; ❌ = not done yet.

## Report content

| Brief requirement | Where it is in the report | Status |
|---|---|---|
| Objective: one model + whether its explanations can be trusted | Title, abstract, Sec. 1 | ✅ |
| Q1 Are explanations true (do they reflect what the model does) | Sec. Q1: model faithfulness + ADS ground truth | ✅ |
| Q2 Are they useful (do they help a real person decide) | Sec. Q2: pre-registered tuning proxy + interview requirements + case card | ⚠️ No real-person decision test; a follow-up session with the stakeholder using the case card would fill this |
| Q3 Do they hold up (unusual conditions) | Sec. Q3: circuit → full-wave openEMS; M0 weak model | ✅ (n = 3, stated) |
| Framework Data · Model · Stakeholder · Decision | Table I (`tab:frame`) | ✅ |
| Pick one principle | Contrastive (Lipton), Table I + Sec. 4 | ✅ |
| Framing question in the template form | Quote in Sec. 1 | ✅ |
| Stakeholder evidence (interview / practitioner sources) | Sec. 2: one interview + R&S, ADI sources | ✅ (single person, stated) |
| ≥ 3 methods + where and why they disagree | SHAP, IG, LIME; Sec. "Where and why the three methods disagree" | ✅ |
| Faithfulness test written down before seeing results | Table II (`tab:prereg`): freeze 10-25 17:21 UTC, pre-registration 10-25 19:23 UTC | ✅ honest caveats: validation results seen; deletion/insertion not verifiable |
| Domain grounding: state ground truth, check against it | Sec. "Domain grounding" (controls, T0 known answer, physics ranking, poles) + ADS Shapley | ✅ |
| Related work (proposal element) | Sec. "Related Work" | ✅ |
| "Failed or abandoned explanations" section | Sec. 8 | ✅ |
| 6–8 pp, conference format | IEEEtran, **single column** | ⚠️ About 3200 words + 7 tables + 5 figures: roughly 6–7 pp in two-column, about 10–11 pp in single column. Check the length after compiling; if 6–8 pp is counted as single column, cut content (first: mesh-convergence figure, framework table, part of the failed-explanation list) |

## Other deliverables (due Dec 9)

| Item | Status |
|---|---|
| Draft report (Dec 2) | ✅ `report/` (compile and check page count) |
| Author names | ❌ `\todo` in `main.tex` |
| git repo with README | ❌ README.md still describes steps up to step 4 only; add openEMS + report; no git yet |
| Decision log | ❌ `docs/decision_log.md` not written (stakeholder switch, contrastive principle, openEMS instead of HFSS, single-column layout, ...) |
| AI-use reflection (1–2 pp) | ❌ |
| Oral defense (10 min, each member 5 min on one idea) | ❌ team is solo in practice — confirm format with instructor |
| Per-case quality diagnostics in the case card (LIME R², IG residual for test_0005) | ❌ `\todo` in Table "case summary" |
