# Oral defense ideas and questions

Prepared November 22, 2026. This is a ten-minute content outline, arranged as two five-minute ideas. Allocate actual speakers according to their knowledge and the instructor's format; the outline does not assign or imply project contributions. With three named authors, confirm how the brief's ten-minute total and five-minute-per-member wording applies.

## First idea Faithfulness depends on the question

**Minute 0–1:** Introduce the stakeholder: a receive-side troubleshooting engineer chooses checks and designer handoffs, rather than directly editing geometry. State the contrast: this design versus a known-good ADS reference. Show the failed specification and frequency region before attribution.

**Minute 1–2:** Explain the model and target. PR-Net predicts complex S21/S11 through a pole/residue representation. It is primary because modal concepts can be checked; M1 is more accurate and remains an explicit comparison. A smooth mask violation is explained; the hard sampled mask determines pass/fail.

**Minute 2–3:** Explain the reference experiment. Enumerate 512 model coalitions and 64 six-geometry ADS coalitions. Separate the surrogate's own reference-game agreement with ADS from comparisons between attribution definitions. Negative controls have zero simulator influence but may influence a weak learned model.

**Minute 3–4:** Show Figure 1 and the numerical checks. PR-Net SHAP agrees closely with the selected ADS allocation. IG integrates a path; LIME estimates local behavior. Disagreement with Shapley does not automatically mean unfaithfulness. Deep S11 dB notches cause a separate IG integration problem.

**Minute 4–5:** Use Table IV to answer single-case trust. Explain LIME R2=0.846, IG relative residual 0.0821% at 64 steps, same-target response error 0.0749 dB, and the limited meaning of SHAP additivity. None is a calibrated probability that a hardware decision is correct.

## Second idea Faithful attribution does not guarantee a transferable fix

**Minute 5–6:** Show Figure 2. SHAP/IG restricted-pair tuning succeeds in 25/30 cases versus random 6/30. Keep the stronger all-six baseline visible: 30/30. The result supports restricted localization, not faster unrestricted optimization or measured human benefit.

**Minute 6–7:** Explain the solver comparison. ADS uses interconnected distributed component models. openEMS solves a three-dimensional electromagnetic problem using EC-FDTD; Momentum provides another full-wave formulation. Different algorithms provide a cross-check, while setup and numerical errors still matter.

**Minute 7–8:** Show Table V and Figure 5. All three selected tuned cases pass ADS but fail openEMS. Two improve, one worsens. The examples show a transfer limitation; they do not estimate a general failure rate. The baseline mesh study does not certify convergence of every case.

**Minute 8–9:** Explain the reporting correction. The old full-wave summary used absolute -3 dB bandwidth; model metrics used peak-relative -3 dB bandwidth. Both are now labeled. Tuned relative reductions are about 80–87 MHz, compared with absolute reductions of 124–142 MHz. Hard-mask verdicts are unaffected.

**Minute 9–10:** Return to the practitioner: AI supports looking, filtering and evidence gathering; independent verification and the responsible engineer determine action. Mention the two confirmed AI uses: ADS batch/parallel automation and assistance locating/explaining simulator algorithms and mathematics.

## Questions to be ready for

| Question | Supported answer |
|---|---|
| Why PR-Net if M1 is more accurate? | PR-Net passed the validation gate and provides inspectable modal concepts; accuracy superiority is not claimed |
| Is ADS ground truth the real physics? | It is an exact reference game within the circuit simulator and stated target/reference; full-wave/hardware validity is separate |
| Does the comparison unfairly favor SHAP? | It evaluates agreement with a Shapley allocation, so matching definitions are favored; IG/LIME are also assessed with method-specific diagnostics |
| Why only two parameters? | It is the preregistered restriction being tested. All-six optimization is stronger and explicitly retained |
| Did explanations improve the engineer's decisions? | The interviews supply qualitative requirements and anticipated benefits; task performance was not measured |
| Does 15 candidate pairs reduced to one mean 15 times faster? | No. That counts selection alternatives, not measured runtime or complete diagnostic effort |
| Does missing port power prove radiation? | No. Radiation is plausible, but port, boundary and numerical effects were not uniquely isolated |
| Why not claim all meshes converged? | Only baseline refinement was studied systematically; case-specific convergence remains unproven |
| Were all files unchanged after the freeze? | No. Twenty-six historical hashes match, including nine weights; four documented plan/protocol/path files differ and old snapshots were not reconstructed |
| What did AI actually do? | Assisted ADS Python Console batch/parallel work, technical-source retrieval/explanation, and subsequent documentation/recomputation support. The reflection distinguishes these from independent verification |

Do not claim an experiment was performed merely because it was proposed, or assign a teammate a contribution that is not documented. Use the archived results and the final report's limitations when answering follow-up questions.
