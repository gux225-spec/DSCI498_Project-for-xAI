# Stakeholder Notes: RF Receiver Front-End Evaluation and Troubleshooting

Prepared: November 5, 2026

Evidence status: Interview conducted by the project author with one practitioner via internet voice call on November 5, 2026; the author contacted the participant directly. This document is an edited English paraphrase of the author's notes, not a verbatim transcript; the participant is not named. The project was introduced to the participant beforehand, so project-specific terms in the account may reflect that introduction. Technical qualifications and project implications below are editorial analysis, separated from the practitioner perspective. The numerical examples are illustrative, not measurements from a documented incident.

## 1. Role and Working Context

The practitioner works on RF front-end evaluation and troubleshooting on the receive side of a communications system. They treat the filter as a constraint on what reaches the receiver, rather than as a geometry-design task they personally perform. Most of the conversation concerned microwave transceivers and antennas, with the underlying signal-processing algorithms such as mixing at the base of their work; component dimensions came up only rarely.

Typical reported problems include excessive error vector magnitude (EVM), degraded receiver sensitivity, and adjacent-channel interference. Frequency-response checks help narrow the investigation by examining passband flatness, band-edge behavior, and out-of-band rejection. Spectrum measurements provide complementary evidence about interfering signals, harmonics, and spurious components under the relevant operating conditions.

Poor filter performance can contribute to receiver problems, but its effect depends on the filter's position, loss, rejection, and the rest of the chain. Loss ahead of the first LNA can impair receiver noise performance. Insufficient rejection can allow strong blockers to drive nonlinear behavior elsewhere in the receiver. Neither passband ripple alone nor a single response plot establishes the full cause of an EVM or sensitivity failure. [1, 2, 3]

## 2. Evidence and Decision-Making

The account describes the following diagnostic workflow:

1. Use response information to identify a suspicious frequency region or failed specification.
2. Confirm the relevant behavior with a VNA, spectrum/signal analyzer, or other appropriate measurement.
3. Compare the evidence against the applicable specification and expected behavior.
4. Investigate whether the issue originates in design, incoming components, assembly, or another part of the setup or chain.

An explanation is useful when it helps localize a problem or choose the next check. For example, a statement that measured passband ripple is 3 dB against a 1 dB limit identifies both the failure and its magnitude. This is an illustrative example, not a reported measurement.

A geometry ranking such as “L2 is important” is insufficient on its own. The practitioner does not directly adjust microstrip dimensions. Geometry attribution may instead support a discussion with a component designer, provided that it is connected to a clearly defined response metric, frequency region, reference, and unit.

## 3. Conditions for Using Model Explanations

The account identifies two potential uses:

- **Early design review:** comparing how candidate changes affect competing response objectives, such as out-of-band rejection and passband flatness, to help prioritize further investigation.
- **Failure diagnosis:** narrowing the set of hypotheses or components that deserve additional measurement or design-team review.

These are proposed uses expressed in the account. No measured reduction in troubleshooting time, successful field diagnosis, or observed use of the project's explanation interface was reported.

## 4. Three Requirements for a Useful Explanation

### Requirement A: Start with the specification failure

Present the affected frequency range, the failed specification, and the amount of violation before presenting attribution values. Include engineering units and a reference or limit that the user can interpret.

Geometry attribution should answer a defined follow-up question, such as which inputs contribute to a predicted mask violation relative to the chosen baseline. A contribution of “0.7” is incomplete unless its target, unit, sign convention, and reference are stated.

### Requirement B: Provide case-level quality information

An aggregate rank correlation such as 0.99 does not establish whether the explanation for one particular design is reliable. The practitioner wants an indication of when an individual explanation should be treated cautiously.

Candidate diagnostics include LIME's held-out local R², IG completeness residual, explanation stability under the stated settings, and available evidence about local surrogate accuracy. These indicators address different failure modes. A high local R² or small completeness residual is not, by itself, a calibrated uncertainty estimate or proof of physical correctness.

### Requirement C: State the action boundary

Clearly distinguish a diagnostic attribution from a proposed parameter change. A diagnostic explanation identifies contributions to a prediction under a defined comparison; a tuning recommendation proposes a change and requires feasibility and validation checks.

For this stakeholder, an appropriate next action may be to remeasure a frequency region or consult the component designer, rather than modify a geometry parameter.

## 5. Interpretation and Validation Boundaries

The account raises two concerns that should be evaluated separately:

- **Model faithfulness:** does the explanation reflect the surrogate's actual input dependence?
- **Domain plausibility and simulator consistency:** does the result agree with relevant engineering expectations and controlled simulator evidence?

Expectations such as “length changes affect frequency position” and “coupling-related geometry affects response shape or modal Q” are hypotheses to check within the current topology and operating range. They are not a universal ground-truth ranking, and disagreement with intuition does not by itself prove an explanation wrong.

In the project, ADS Shapley is a reference attribution defined by the circuit simulator, selected baseline, target, and coalition construction. Agreement supports simulator consistency under that setup; it does not replace full-wave electromagnetic validation or measurements of a fabricated device.

The current surrogate predicts filter S-parameters and derived targets. It does not directly predict or explain complete-chain EVM, BER, receiver noise figure, or blocker performance.

## 6. Implications for This Project

The following are project-author interpretations of the interview, not additional participant statements:

| Finding from the account | Proposed design response | Evidence still needed |
|---|---|---|
| A frequency region and specification failure are more useful than a bare geometry ranking | Show response/specification context before attribution | Confirm the layout with the practitioner on a concrete case |
| Geometry edits are outside this stakeholder's direct role; their main concerns are transceiver-, antenna-, and algorithm-level (e.g., mixing), and component dimensions rarely arise | Position geometry attribution as design-team communication support; present the filter as one constraint in the receive chain | Identify the actual handoff and the information needed by the designer |
| Aggregate agreement does not answer case-level trust questions | Show method-specific quality diagnostics and their limits | Test whether users understand the diagnostics and make better decisions |
| Diagnostic statements and actionable changes must be distinguished | Label attribution and tuning proposals separately | Validate proposals independently and establish role-appropriate actions |

For a frequency-local claim such as “the deviation is concentrated near 12 GHz,” show the supporting response residual or a frequency-specific target. A single aggregate mask-violation attribution does not itself localize the failure in frequency.

The hard mask-violation target is a nonnegative sum of specification exceedances: zero denotes passing the sampled mask, and positive values denote violation. The smooth training/explanation counterpart is an approximation and need not be exactly zero for a passing response. Neither scalar directly specifies which way to move a geometry parameter.

The proposed controlled-wrapper test can address one model-faithfulness question. If the wrapper explicitly fixes nuisance inputs before calling the model, those inputs have no functional effect by construction. The test should check whether the explanation reports that invariance. This is a proposed test, not a completed result or a validation of the entire receiver chain.

## 7. Slide-Ready Summary

**Stakeholder perspective: downstream RF receiver troubleshooting**

- Decisions begin with the affected band, failed specification, and violation magnitude.
- Geometry attribution is mainly useful for diagnosis and communication with component designers.
- Users need case-level explanation-quality indicators; aggregate agreement is insufficient.
- Diagnostic attribution and executable tuning advice must be clearly separated.

**Project response:** put specification context first, attach method-specific diagnostics, and state what action the evidence supports.

Attribution label for the slide: “Interview with one receive-side RF practitioner (internet voice call, Nov 5, 2026); paraphrased from the author's notes.” “Interview findings” is appropriate; do not present the wording as a direct quote.

## 8. Approximately One-Minute Speaker Note

The practitioner account describes receive-side RF front-end evaluation and troubleshooting. In this role, the filter is a constraint on receiver behavior rather than something the engineer directly redesigns. Three requirements emerged. First, an explanation should start with the affected frequency range, failed specification, and amount of violation. A geometry ranking alone does not identify the next diagnostic step. Second, aggregate agreement with ADS does not tell the engineer whether one particular explanation is reliable, so case-level quality information is needed. Third, diagnostic attribution must be separated from a validated tuning recommendation. These requirements lead me to present response and specification context before SHAP or LIME, add method-specific diagnostics, and make the action boundary explicit. These are requirements from one interview, not evidence that the tool already improves troubleshooting.

## 9. Technical Review of the Supplied Wording

- Replace categorical claims that a distorted passband edge inevitably makes receiver noise figure unmanageable with conditional statements about loss, filtering position, blockers, and the receiver architecture.
- Distinguish swept S-parameter characterization from measurements of actual harmonics or spurious signals. A conventional small-signal S-parameter sweep alone does not identify all interference or nonlinear distortion.
- Treat a good local R², an IG residual, or stability as a specific diagnostic, not as a general confidence score.
- Describe mask violation as exceedance magnitude or pass/fail evidence, not as a geometry-adjustment direction.
- Treat engineering intuition as domain grounding to investigate; keep it separate from a mathematical or simulator-defined reference explanation.
- Do not count the account's references to project slides, PR-Net statistics, ADS Shapley, or the controlled wrapper as independent evidence that a practitioner spontaneously requested those exact methods. They may reflect the project context presented to the respondent.

## Sources Used for Technical Qualifications

1. Analog Devices, **AN-1354: Integrated ZIF, RF to Bits, LTE, Wide Area Receiver Analysis and Test Result**. Receiver-chain example discussing RF filtering, out-of-band energy, and receiver performance. Its specific architecture is not the project's filter topology.
   https://www.analog.com/en/resources/app-notes/an-1354.html

2. Rohde & Schwarz, Paul Denisowski, **Understanding Error Vector Magnitude**. Documents multiple amplitude, phase, I/Q, and configuration-related contributors to EVM.
   https://www.rohde-schwarz.com/us/products/test-and-measurement/analyzers/signal-and-spectrum-analyzers/understanding-error-vector-magnitude_258370.html

3. Analog Devices EngineerZone, **MAX2659 RF-filter placement FAQ**. Explains the importance of minimizing passive loss before an LNA in its GNSS example; not a universal prescription for filter placement.
   https://ez.analog.com/amplifiers/a/documents/c/max2659elt-faq/DO11014/is-it-necessary-to-add-coupling-cap-between-rf-filter-and-rf-input-max2659-can-we-put-a-saw-filter-lc-in-front-of-rf-input

4. Keysight, **Spectrum Analyzer application documentation**. Distinguishes component characterization from signal analysis and describes dedicated spectrum-analysis capabilities on VNA hardware.
   https://helpfiles.keysight.com/csg/pxivna/Applications/Spectrum_Analyzer.htm

5. Project implementation: `src/filterxai/targets.py`, `src/filterxai/xai.py`, and `docs/progress_talk_5min.md`.

External sources support the technical qualifications, not the representativeness of this single interview. Accessed November 5, 2026.
