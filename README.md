# Explaining a microwave filter surrogate

DSCI 498 / ECE 350/450, Explainable AI, Lehigh University.

This project asks whether an RF troubleshooting engineer can trust an AI explanation when choosing what to check next. It compares SHAP, Integrated Gradients and LIME for filter-response models, checks their suggestions in ADS, and tests whether tuned designs still work in openEMS.

> Code and a short [guide](GUIDE.md) are kept here. Data, saved runs and course documents are on [Google Drive](https://drive.google.com/drive/folders/1ZZ6kcEYQmrsJO3eA3q-N06ZXHMaiEowi).

## Repository layout

```text
src/filterxai/        data loading, models, targets, metrics and explanations
scripts/             training, attribution, ADS tuning and openEMS checks
data/filter_supplement_v2/scripts/
                     dataset generation and release loader
ads/baseline.ckt      reference circuit netlist
GUIDE.md             experiment steps and common issues
requirements*.txt    Python dependencies
```

## Data

Download from [Google Drive](https://drive.google.com/drive/folders/1ZZ6kcEYQmrsJO3eA3q-N06ZXHMaiEowi):

| File | Contents |
|---|---|
| `DSCI498_data.zip` | 1,440 ADS filter designs, fixed splits and normalization metadata |
| `DSCI498_results.zip` | Weights, predictions, explanations and saved simulator responses |
| `DSCI498_course_materials.zip` | Report, AI reflection, decision log, interview notes and detailed runbook |

Clone into a folder named `DSCI498_Project`. Extract the ZIPs into its parent folder and merge their `DSCI498_Project/` directories with the clone. Inputs then live in `data/filter_supplement_v2/`, saved runs in `results/`, and course documents in `docs/` and `report/`.

## Setup and run

Python 3.11. To check the saved results without running a simulator:

```powershell
pip install -r requirements-check.txt
python scripts/verify_submission.py
```

For model inference and explanation experiments:

```powershell
pip install -r requirements-xai.txt
python scripts/verify_submission.py --live-models
```

New circuit simulations require ADS 2026 and a license. Full-wave runs require a separate openEMS/CSXCAD installation. See [GUIDE.md](GUIDE.md) for script order and local paths.

## Main limitations

- ADS is the circuit-model reference; agreement with it does not establish hardware accuracy.
- SHAP, IG and LIME answer different attribution questions, so disagreement needs interpretation.
- The full-wave check covers three selected designs. Interview feedback describes possible workflow benefits, not measured time savings.
