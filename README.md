# AISEHack 2.0 — Polymer Property Prediction

Submission for the **Polymer Property Prediction** track of [AISEHack 2.0](https://precog.iiit.ac.in/aisehack) — ANRF's national AI-for-science hackathon, this track contributed by IIT Madras. Predicts two polymer properties (`tg`, `egc`) from SMILES strings.

Data comes from the competition's private Kaggle page (registration-gated) — not included in this repo; see [Data](#data) below.

## Approach

RDKit descriptors → two modeling phases, compared honestly via 5-fold CV rather than assumed:

- **Phase 1 — Ridge regression** on RDKit's default descriptor set (~217 physicochemical descriptors: molecular weight, LogP, ring counts, topological indices, etc). Separate model per target (`tg`, `egc`).
- **Phase 2 — HistGradientBoosting** on the same features. Sidesteps Ridge's sensitivity to feature scale/collinearity entirely.

The better-performing model per target (by CV RMSE) generates the final submission.

## Two real bugs found and fixed

1. **8 `BCUT2D_*` descriptor columns are `NaN` for every single molecule.** Every polymer SMILES here uses a `*` dummy attachment atom, which these particular descriptors can't handle — `fillna(mean())` of an all-`NaN` column is still `NaN`, so these columns are dropped outright rather than imputed.
2. **RDKit's `Ipc` descriptor explodes to ~10⁵⁰–10¹⁰⁰⁺** for certain molecules (a known RDKit quirk). Left in, it wrecks `StandardScaler` + `Ridge` numerically — the scaled design matrix's condition number blows past 1e50 and Ridge silently returns astronomical, meaningless coefficients even with regularization. Dropping this one column fixes it completely.
3. The **test set** needs the identical cleanup before scoring. A common way this exact kind of pipeline breaks silently: clean the train features carefully, forget to do the same to test, and any `inf`/`NaN` descriptor value on a test molecule turns into a `NaN` prediction — which gets rejected or zero-scored.

## Results (5-fold CV RMSE)

| Target | Ridge (full features) | Ridge (top-20 by \|coef\|) | HistGB | Used for submission |
|---|---|---|---|---|
| `tg`  (mean≈140, std≈109) | 47.98 | 58.54 *(worse)* | **40.01** | HistGB |
| `egc` (mean≈4.5, std≈1.56) | 0.643 | 0.958 *(worse)* | **0.551** | HistGB |

Selecting the "top 20 features by absolute Ridge coefficient" made both models *worse*, not better — with this many correlated RDKit descriptors (15+ column pairs at \|corr\| > 0.999), coefficient magnitude doesn't reliably identify the most independently predictive columns; credit gets split arbitrarily across near-duplicates. Kept the full cleaned feature set instead of assuming fewer features generalize better.

## Honest limitations / next steps

- No hyperparameter tuning on HistGB — defaults only. Natural next step if time allows.
- Feature set is RDKit's default 2D descriptors only. Graph-based representations (message-passing GNNs on the polymer graph) or learned embeddings would likely do better but need more time/compute than a first submission allowed.
- `egc`'s exact physical definition wasn't confirmed against the competition's data description page (gated behind registration) — worth checking there.
- CV RMSE is a local estimate; the actual leaderboard metric may differ (e.g. a weighted multi-target score) and the hidden test distribution may not match local CV exactly.

## Data

Not included in this repo (get it from the competition's Kaggle page, which requires AISEHack registration):
- `train.csv` — `smiles`, `target`, `target_type` (`tg` or `egc`)
- `test.csv` — `id`, `smiles`, `target_type` (4,115 rows)
- `sample_submission.csv` — format example only (10 rows; real submission needs one row per `test.csv` id)

To reproduce: drop `train.csv` and `test.csv` alongside `notebook.ipynb` and run top to bottom.

## Repo structure

```
├── README.md
├── notebook.ipynb       # full pipeline, executed end-to-end
├── submission.csv       # final predictions (4,115 rows: id, target)
└── requirements.txt
```
