# V13 Step 1 — Empirical COVID Disruption Audit

Produced by `src/v13_covid_audit.py`. Baseline year: **2019** (last full pre-pandemic year).

## 1. Monthly mean Revenue by year ($M)

| Year | M01 | M02 | M03 | M04 | M05 | M06 | M07 | M08 | M09 | M10 | M11 | M12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | — | — | — | — | — | — | 4.66 | 5.13 | 4.31 | 3.55 | 3.27 | 3.69 |
| 2013 | 2.95 | 3.92 | 4.90 | 6.63 | 6.47 | 6.62 | 5.35 | 3.75 | 4.25 | 3.71 | 3.01 | 2.94 |
| 2014 | 2.96 | 3.87 | 5.92 | 7.75 | 7.88 | 6.94 | 5.46 | 6.30 | 4.02 | 3.90 | 3.11 | 3.35 |
| 2015 | 2.90 | 4.31 | 5.51 | 8.66 | 7.75 | 7.19 | 5.49 | 3.87 | 5.17 | 4.49 | 3.41 | 3.41 |
| 2016 | 3.78 | 4.47 | 5.78 | 8.90 | 8.65 | 8.46 | 5.78 | 7.32 | 4.84 | 4.11 | 3.50 | 3.41 |
| 2017 | 3.07 | 3.93 | 6.08 | 7.65 | 8.48 | 8.91 | 6.10 | 3.78 | 4.50 | 4.25 | 3.37 | 2.70 |
| 2018 | 2.93 | 3.94 | 5.49 | 7.34 | 8.48 | 9.06 | 5.87 | 6.27 | 3.93 | 3.24 | 2.26 | 1.98 |
| 2019 | 2.35 | 3.04 | 3.91 | 4.52 | 4.66 | 4.59 | 3.21 | 2.63 | 2.78 | 2.32 | 1.83 | 1.56 |
| 2020 | 1.54 | 2.15 | 3.61 | 4.67 | 4.28 | 3.69 | 2.93 | 3.94 | 2.63 | 2.20 | 1.52 | 1.41 |
| 2021 | 1.50 | 2.36 | 3.65 | 4.49 | 4.62 | 4.28 | 3.26 | 2.20 | 2.49 | 2.12 | 1.69 | 1.63 |
| 2022 | 1.92 | 2.83 | 4.44 | 4.71 | 4.48 | 4.53 | 3.17 | 3.66 | 2.86 | 2.43 | 1.74 | 1.69 |

## 2. Ratio vs 2019 baseline

| Year | M01 | M02 | M03 | M04 | M05 | M06 | M07 | M08 | M09 | M10 | M11 | M12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | — | — | — | — | — | — | 1.45 | 1.95 | 1.55 | 1.53 | 1.78 | 2.36 |
| 2013 | 1.25 | 1.29 | 1.25 | 1.47 | 1.39 | 1.44 | 1.67 | 1.43 | 1.53 | 1.60 | 1.64 | 1.88 |
| 2014 | 1.26 | 1.27 | 1.51 | 1.71 | 1.69 | 1.51 | 1.70 | 2.40 | 1.45 | 1.68 | 1.70 | 2.14 |
| 2015 | 1.23 | 1.42 | 1.41 | 1.91 | 1.66 | 1.57 | 1.71 | 1.47 | 1.86 | 1.93 | 1.86 | 2.19 |
| 2016 | 1.61 | 1.47 | 1.48 | 1.97 | 1.86 | 1.84 | 1.80 | 2.79 | 1.74 | 1.77 | 1.91 | 2.18 |
| 2017 | 1.31 | 1.29 | 1.55 | 1.69 | 1.82 | 1.94 | 1.90 | 1.44 | 1.62 | 1.83 | 1.84 | 1.73 |
| 2018 | 1.25 | 1.30 | 1.40 | 1.62 | 1.82 | 1.97 | 1.83 | 2.38 | 1.42 | 1.40 | 1.23 | 1.27 |
| 2020 | 0.65 | 0.71 | 0.92 | 1.03 | 0.92 | 0.80 | 0.91 | 1.50 | 0.95 | 0.95 | 0.83 | 0.90 |
| 2021 | 0.64 | 0.78 | 0.93 | 0.99 | 0.99 | 0.93 | 1.02 | 0.84 | 0.90 | 0.91 | 0.92 | 1.04 |
| 2022 | 0.82 | 0.93 | 1.13 | 1.04 | 0.96 | 0.99 | 0.99 | 1.39 | 1.03 | 1.04 | 0.95 | 1.08 |

## 3. Disruption classification

Thresholds: SEVERE < 0.70 · MODERATE 0.70-0.85 · MILD 0.85-0.95 · NORMAL 0.95-1.30 · BOOM > 1.30

| Year | M01 | M02 | M03 | M04 | M05 | M06 | M07 | M08 | M09 | M10 | M11 | M12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | ? | ? | ? | ? | ? | ? | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM |
| 2013 | NORMAL | NORMAL | NORMAL | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM |
| 2014 | NORMAL | NORMAL | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM |
| 2015 | NORMAL | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM |
| 2016 | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM |
| 2017 | BOOM | NORMAL | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM |
| 2018 | NORMAL | NORMAL | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | BOOM | NORMAL | NORMAL |
| 2020 | SEVERE | MODERATE | MILD | NORMAL | MILD | MODERATE | MILD | BOOM | MILD | MILD | MODERATE | MILD |
| 2021 | SEVERE | MODERATE | MILD | NORMAL | NORMAL | MILD | NORMAL | MODERATE | MILD | MILD | MILD | NORMAL |
| 2022 | MODERATE | MILD | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | BOOM | NORMAL | NORMAL | MILD | NORMAL |

## 4. Empirical flag windows derived from the data

| Flag | Start | End | Days |
|---|---|---|---:|
| `is_covid_lockdown` | 2021-01-01 | 2021-01-31 | 31 |
| `is_covid_disruption` | 2020-01-01 | 2020-01-31 | 31 |
| `is_covid_disruption` | 2021-02-01 | 2021-02-28 | 28 |
| `is_covid_disruption` | 2021-03-01 | 2021-03-31 | 31 |
| `is_covid_disruption` | 2021-06-01 | 2021-06-30 | 30 |
| `is_covid_disruption` | 2021-08-01 | 2021-08-31 | 31 |
| `is_covid_recovery` | 2022-08-01 | 2022-08-31 | 31 |
| `is_covid_disruption` | 2021-09-01 | 2021-09-30 | 30 |
| `is_covid_disruption` | 2021-10-01 | 2021-10-31 | 31 |
| `is_covid_disruption` | 2021-11-01 | 2021-11-30 | 30 |

## 5. Public Vietnam COVID timeline (for cross-reference)

| Window | Description |
|---|---|
| 2020-04-01 → 2020-04-22 | Nat'l Directive 16 (3 wk lockdown) |
| 2020-07-28 → 2020-09-05 | Da Nang outbreak |
| 2021-01-28 → 2021-03-18 | Hai Duong / Northern wave |
| 2021-05-27 → 2021-10-01 | DELTA wave (HCMC strict lockdown) |
| 2021-10-01 → 2022-03-31 | Reopening / revenge spending |
| 2022-03-01 → 2022-04-30 | Omicron (mild) |

## 6. How V13 will use this

1. **Features.** Each row of `daily_features_v13.parquet` carries `is_covid_lockdown`, `is_covid_disruption`, `is_covid_recovery`, and `covid_severity ∈ [0, 1]`. All four equal 0 across the 548-day test horizon.
2. **Frozen priors.** `prior_rev_by_month_dow`, `prior_cogs_by_month_dow`, `aov_prior_by_month`, `prior_rev_event_uplift` are computed on the train window **excluding rows where `covid_severity > 0`**. The priors therefore reflect normal-time behaviour, which is what 2023-2024 will be.
3. **Lag bypass.** `rev_same_dow_prev_year` and `rev_yoy_lag_364` skip past any candidate date whose `covid_severity > 0`, bouncing forward another 365 days until they land on a normal date.
4. **Sample weighting.** When fitting a model: `sample_weight = 1 - 0.7 × covid_severity`. Lockdown rows down-weighted to 0.30, disruption rows to 0.65, recovery rows to 0.79.

## 7. Sanity checks

- Train rows flagged lockdown: 31
- Train rows flagged disruption: 242
- Train rows flagged recovery: 31
- Test rows flagged any (must be 0): 0

## 8. Artifacts

- `data/processed/v13_covid_flags.parquet` — per-day flag table (2017-01-01 → 2024-07-01).
- `figures/v13_covid_disruption.png` — Revenue trace + monthly ratio plot.
