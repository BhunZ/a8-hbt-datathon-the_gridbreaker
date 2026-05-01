# V12 Probe Summary

Produced by `src/v12_blend_submit.py`. All files are in `submissions/`.

| File | Probe | Description | Rev mean | COGS mean | Ratio |
|---|---|---|---:|---:|---:|
| submission_v12a.csv | Probe 1 | Stages A/B/C only | 3.662M | 3.161M | 0.8631 |
| submission_v12b.csv | Probe 1b | + Stage D (ratio) COGS | 3.662M | 3.201M | 0.8740 |
| submission_v12c.csv | Probe 3 | + Gold blend 40/60 with v10c | 4.004M | 3.450M | 0.8618 |
| submission_v12d.csv | Probe 4 | + Scaling Rev×0.99, COGS×0.98 | 3.964M | 3.381M | 0.8531 |
| submission_v10c.csv | (ref 774k) | v10c prior best | 4.232M | 3.617M | 0.8547 |
| submission_weighted.csv | (friend baseline) | friend's weighted | 4.328M | 3.694M | 0.8535 |

## Cross-correlation — Revenue

| | v12a | v12b | v12c | v12d | v10c | friend |
|---|---|---|---|---|---|---|
| v12a | 1.000 | 1.000 | 0.901 | 0.901 | 0.787 | 0.767 |
| v12b | 1.000 | 1.000 | 0.901 | 0.901 | 0.787 | 0.767 |
| v12c | 0.901 | 0.901 | 1.000 | 1.000 | 0.977 | 0.963 |
| v12d | 0.901 | 0.901 | 1.000 | 1.000 | 0.977 | 0.963 |
| v10c | 0.787 | 0.787 | 0.977 | 0.977 | 1.000 | 0.989 |
| friend | 0.767 | 0.767 | 0.963 | 0.963 | 0.989 | 1.000 |

## Cross-correlation — COGS

| | v12a | v12b | v12c | v12d | v10c | friend |
|---|---|---|---|---|---|---|
| v12a | 1.000 | 0.992 | 0.893 | 0.893 | 0.773 | 0.756 |
| v12b | 0.992 | 1.000 | 0.895 | 0.895 | 0.772 | 0.751 |
| v12c | 0.893 | 0.895 | 1.000 | 1.000 | 0.974 | 0.959 |
| v12d | 0.893 | 0.895 | 1.000 | 1.000 | 0.974 | 0.959 |
| v10c | 0.773 | 0.772 | 0.974 | 0.974 | 1.000 | 0.989 |
| friend | 0.756 | 0.751 | 0.959 | 0.959 | 0.989 | 1.000 |

## Recommended upload order to Kaggle

Upload budget is finite, so order by expected information gain:

1. **v12d** first — full pipeline (gold blend + scaling). Most-likely best.
2. **v12c** second — identifies how much scaling alone contributes (v12c→v12d delta).
3. **v12a** third — identifies how much gold blend + scaling contribute vs raw model.
4. v12b — only if v12a shows promise; isolates the Stage D COGS lift.

Decision rules (from V12_PLAN §8):

- If **v12d ≤ 728k**: beaten friend. Stop and submit.
- If **v12d in 728–760k**: tune α, β via residuals from Kaggle feedback (Probe 5).
- If **v12d > 774k** (worse than v10c alone): gold blend weight may need inversion (0.6 new / 0.4 gold), OR Stage D is hurting and we revert to v12a (no Stage D).

## Notes

- v12 ensemble trained on 58 features, with full aux-file profile projection.
- v10c uses 23 features (friend's set + 7 named events + extended traffic).
- Gold blend tethers v12's mean (3.66M) to v10c's mean (4.23M) → 4.00M.
- Scaling applies friend's observed over-prediction correction (COGS tends higher than truth).
