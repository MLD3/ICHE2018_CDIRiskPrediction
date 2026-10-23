# ICHE2018_CDIRiskPrediction

Code for training a facility-specific model that predicts a hospitalized patient's daily risk of *Clostridium difficile* infection (CDI) from structured electronic health record (EHR) data.

This repository accompanies:

> Oh J, Makar M, Fusco C, McCaffrey R, Rao K, Ryan EE, Washer L, West LR, Young VB, Guttag J, Hooper DC, Shenoy ES, Wiens J. **A Generalizable, Data-Driven Approach to Predict Daily Risk of *Clostridium difficile* Infection at Two Large Academic Health Centers.** *Infection Control & Hospital Epidemiology* 2018;39(4):425–433. [doi:10.1017/ice.2018.16](https://doi.org/10.1017/ice.2018.16)

## Overview

The paper does not try to build one model that works at every hospital. Instead, it describes a method that any institution can apply to its own EHR data to train its own model. The method was applied at the University of Michigan Hospitals (UM) and Massachusetts General Hospital (MGH):

| | UM | MGH |
|---|---|---|
| Admissions | 191,014 | 65,718 |
| Features | 4,836 | 1,837 |
| Test AUROC (95% CI) | 0.82 (0.80–0.84) | 0.75 (0.73–0.78) |

At a threshold set at the 95th percentile, half of the correctly identified cases were flagged at least 5 days before the sample was collected.

**No patient data is included.** You provide your own extracted features and labels.

## Repository contents

| File | Purpose |
|---|---|
| `reference.py` | **Configuration you fill in.** Sets the output path, random seed, the C grid, and the number of folds. Loads your data and defines how patient-days are subsampled. |
| `cv.py` | Runs stratified k-fold cross-validation for **one** value of C. Folds are split by admission, so no admission appears in both the training and validation folds. Saves `cv_c<index>_auc.pickle`. |
| `model.py` | Combines the CV results, picks the best C, trains the final model on the full training set, scores the training and test data, and smooths the scores over time. Saves `learn_model.pickle`. |

## Method in brief

1. **Unit of data:** each row is one patient-day of one admission. Every feature is binary (see "Preparing your data" below).
2. **Labels:** each day of an admission that ends in a CDI diagnosis is labeled `1`. Every other day is labeled `0`.
3. **Model:** L2-regularized logistic regression with `class_weight='balanced'` (scikit-learn).
4. **Subsampling:** before each fit, a fixed number of days is sampled from each admission, with replacement. This keeps long stays from dominating the training data.
5. **Hyperparameter search:** C is chosen from 5⁻¹⁴ … 5⁰ to maximize the mean CV AUROC. For each admission, its highest daily score is used as its score when computing the AUROC.
6. **Smoothing:** the risk on day *d* is the average of the raw daily scores from day 1 through day *d*.
7. **Admission-level evaluation:** each admission gets the maximum of its smoothed scores from day 2 onward. An admission is flagged as high risk if that score ever crosses the threshold.

## Requirements

- Python 3
- `numpy`, `pandas`, `scipy`, `scikit-learn`

```bash
pip install numpy pandas scipy scikit-learn
```

## Preparing your data

Feature extraction is specific to each site, so it is **not** part of this code. The paper's approach was:

- **Time-invariant features** (fixed at admission): demographics, encounter history (for example, admissions in the last 90 days), and diagnoses and treatments from the most recent previous stay.
- **Time-varying features** (recorded daily): lab results, procedures, medications, vital signs, in-hospital location, and hospital-level and ward-level colonization pressure.
- **Binarization:**
  - Each categorical value becomes its own binary feature (for example, "received drug X today").
  - Continuous values become flags using the EHR's reference ranges (for example, `WBC = high`), standard clinical ranges, or quintiles.
- Concatenate the time-invariant and time-varying features for each patient-day.
- **Cohort and leakage rules used in the paper:**
  - Exclude stays that end before day 3.
  - Exclude patients who test positive within the first 2 calendar days.
  - Exclude repeat positives within 14 days.
  - Stop extracting data before the day of the positive test or discharge.
  - In the training set, also drop the collection day and the day before it for positive cases, so the model can't learn from empiric CDI treatment.
- **Temporal split:** test on the most recent year and train on the years before it.

## Usage

**1. Fill in `reference.py`**

- `returnVars()`: set `pathout` (the output folder, with a trailing `/`) and `randseed`. Optionally change `c_param` and `nfolds`.
- `loadData()`: return the following arrays:

  | Variable | Type / shape | Description |
  |---|---|---|
  | `xtrain`, `xtest` | `scipy.sparse.csr_matrix`, `n × m` | One row per patient-day |
  | `ytrain`, `ytest` | `numpy` array, `n` | Label for each row (1 = CDI admission) |
  | `eid_train`, `eid_test` | `numpy` array, `n` | Admission/encounter ID, unique per admission |
  | `day_train`, `day_test` | `numpy` array, `n` | Day of stay, starting at 1 |

- `subsample()`: set `size`, the number of days drawn per admission (for example, `3`).

**2. Run cross-validation once for each C value** (indices 0–14 with the default grid). The runs are independent, so they can run in parallel:

```bash
for i in $(seq 0 14); do python cv.py cindex=$i; done
```

**3. Train the final model**

```bash
python model.py
```

## Outputs

`cv_c<i>_auc.pickle` (from `cv.py`) holds `[auc_cv, auc_train_ef, auc_test_ef, c, c_param]`:
- the CV AUROCs for this C,
- the training and test AUROCs (max raw daily score per admission),
- the C value used and the full grid.

`learn_model.pickle` (from `model.py`) holds `[auc_cv, c, clf, df, df2, dftr, dftr2]`:
- `auc_cv`: CV AUROC for every C value and fold
- `c`: the selected C value
- `clf`: the fitted `LogisticRegression` model
- `df`, `dftr`: day-level test and training predictions, with columns `encounterID`, `day`, `rawest`, `label`, `rollingscore`
- `df2`, `dftr2`: admission-level results with the maximum `rollingscore` (from day 2 onward) and the label, used for ROC and threshold analysis

## Known issues

These need small fixes before the code will run:

- `model.py` calls `maggie.loadData()`. Change this to `reference.loadData()`.
- `reference.py` uses `np` without importing it. Add `import numpy as np`.
- `model.py` uses `dftr.sort(...)`, which recent versions of pandas no longer have. Change it to `dftr.sort_values(...)`.
- `rollingscore = cumsum / day` assumes days are numbered from 1 with no gaps.
- `cv.py` passes its command-line arguments to `exec()`. Run it only with trusted input.

## Contact

- Jenna Wiens (wiensj@umich.edu)
- Erica S. Shenoy (eshenoy@partners.org)
