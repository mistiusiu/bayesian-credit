# Data layer — the data contract as implemented

Package `bayesian_credit.data` (SPEC §1, §2.6; proposal Appendix F and G.2). This page
documents what the code actually does: the canonical `LoanTable`, how the Kaggle file is
parsed, how outcomes and credit-state paths are derived, and how the synthetic generator
works. Every **[decision]** below can be configured.

| Module | Main API |
|---|---|
| `data/schema.py` (contract) | `RETAINED_VARIABLES` (Table 3), stage column tuples, status sets, `LeakageGuard`, `BorrowerContext` |
| `data/arms.py` (contract) | `LendingPolicy`, `ArmSpace` (`nearest_many`, `eligible`) |
| `data/segmentation.py` (contract) | `Segmenter` (`assign`, `refine`, `coarsen`) |
| `data/loader.py` | `LendingClubLoader`, `canonicalize`, `validate_loan_table`, `sample_bucket`, `subsample` |
| `data/synthetic.py` | `SyntheticLendingClub(seed).generate(...)`, `generate(n, seed, **kw)`, `DEFAULT_SHIFT` |
| `data/outcomes.py` | `derive_outcomes`, `temporal_split`, `summarize`, `months_between`, `add_months` |
| `data/panel.py` | `LoanStatePanel`, `interim_state`, `row_segments`, `paths_to_counts` |
| `data/__init__.py` | `load_loans(source, **kw)` plus re-exports of everything above |

```python
from bayesian_credit.data import load_loans, derive_outcomes, temporal_split, LoanStatePanel
df = load_loans("synthetic", n=100_000, seed=0)                       # or "lending_club"
df = derive_outcomes(df)                                              # adds y_* columns
train, stream = temporal_split(df, "2015-01-01")
panel = LoanStatePanel.from_table(train)                              # 4x4 transition counts
```

---

## 1. Canonical `LoanTable`

One row per loan: `loan_id` followed by the 60 Table 3 variables, in proposal order
(`loader.CANONICAL_COLUMNS`). Nothing else is ever loaded from the Kaggle file. Free text,
URLs, zip/state, `member_id` and `emp_title` are never read (F.4 data minimisation).
`loan_id` (LC `id`) is a row key only. `LeakageGuard` ignores it and it is never a feature.

| Stage | Columns | Who may see them |
|---|---|---|
| `PRE_DECISION` (37) | annual_inc, annual_inc_joint, application_type, dti, dti_joint, emp_length, fico_range_low, fico_range_high, home_ownership, purpose, verification_status, verified_status_joint, delinq_2yrs, delinq_amnt, inq_last_6mths, inq_last_12m, open_acc, total_acc, revol_bal, revol_util, installment, mort_acc, pub_rec, pub_rec_bankruptcies, collections_12_mths_ex_med, num_accts_ever_120_pd, pct_tl_nvr_dlq, num_sats, tot_coll_amt, tot_cur_bal, tot_hi_cred_lim, total_bal_ex_mort, total_bc_limit, bc_util, all_util, grade, sub_grade | the bandit (`BorrowerContext`) |
| `POLICY` (3) | loan_amnt, int_rate, term | the arm |
| `DISBURSEMENT` (2) | funded_amnt, issue_d | time index / exposure |
| `INTERIM` (10) | last_fico_range_low/high, last_pymnt_amnt, last_pymnt_d, out_prncp, out_prncp_inv, num_tl_30dpd, num_tl_90g_dpd_24m, mths_since_last_delinq, next_pymnt_d | MCTM / interim rewards |
| `TERMINAL` (8) | loan_status, recoveries, collection_recovery_fee, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee | posterior updates after resolution |

**Dtypes:**
- `loan_id` and `term` are `int64`.
- `issue_d`, `last_pymnt_d` and `next_pymnt_d` are `datetime64[ns]`, set to the first day of the month.
- The eight categoricals use the pandas‑3 default string dtype, `loader.STRING_DTYPE` = `StringDtype(na_value=nan)`. Missing values are `NaN`, so masks are plain `bool`.
- Every other column is `float64`; missing values stay `NaN`.
- `int_rate`, `revol_util`, `bc_util`, `all_util` and `pct_tl_nvr_dlq` are **fractions** (0.1399, not 13.99).
- `emp_length` is in years: "< 1 year" becomes 0 and "10+ years" becomes 10.

`validate_loan_table(df)` checks columns, dtypes, unique `loan_id`, non‑null required
columns and fraction‑scaled rates. `canonicalize(df)` converts a frame to canonical order
and dtypes; columns that are already canonical are not copied.

## 2. Lending Club loader (`LendingClubLoader`)

**Source.** Kaggle dataset `wordsforthewise/lending-club` (CC0), file
`accepted_2007_to_2018Q4.csv.gz`: 374 MB gzipped, 2,260,701 rows, 151 columns.
`scripts/download_data.sh` fetches it into `data/raw/` with the Kaggle CLI. It uses
`$KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`, without printing the token. Supports
`--force` and `--dry-run`. The default path is `$LENDING_CLUB_CSV` or
`<repo>/data/raw/accepted_2007_to_2018Q4.csv.gz`. A missing file raises
`DataSourceError`, and the message points to the script.

**Parsing.** All parsing happens in one DuckDB `SELECT` over
`read_csv(path, all_varchar=true, header=true, sample_size=20000)`. Every value is
`trim`med and empty strings become NULL.

| Column(s) | Rule |
|---|---|
| `id` → `loan_id` | `TRY_CAST(... AS BIGINT)`; non‑numeric footer rows become NULL and are dropped |
| `verification_status_joint` → `verified_status_joint` | renamed to the Table 3 name |
| `int_rate`, `revol_util` | `replace(x, '%', '')` then `/100`. Accepts both `'13.99'` (this Kaggle export) and `'13.99%'` (older exports). `revol_util` outliers up to 892% are kept (8.92) |
| `bc_util`, `all_util`, `pct_tl_nvr_dlq` | numeric 0–100, divided by 100 |
| `term` | `regexp_extract(term, '(\d+)')` on `' 36 months'` / `'60 months'`, giving 36 / 60 |
| `emp_length` | `'< 1 year'` → 0; else the first integer (`'10+ years'` → 10); `'n/a'` / empty → NULL. (No `least()`: DuckDB's ignores NULLs) |
| dates | `try_strptime(x, '%b-%Y')`, falling back to `'%b-%y'` and ISO, then cast to `TIMESTAMP_NS` |
| `home_ownership` | upper‑cased; `ANY` and `NONE` become `OTHER` (keeps the 1,232 rare rows) |
| other categoricals | trimmed strings |
| other numerics | `TRY_CAST(... AS DOUBLE)` |
| retained column missing from the file | all‑NULL with the canonical dtype; reported by `describe_source()` |

**Cleaning.** A row is dropped if any of `loan_id`, `loan_status`, `issue_d`,
`loan_amnt`, `int_rate` or `term` is NULL. This removes the 33 all‑NULL rows at the end
of the Kaggle file. A row is also dropped if `policy_code` is present and not 1 (it is
always `'1.0'` in this export). The result is **2,260,668** loans.

**Selection.** `load(max_rows, sample_fraction, seed, issue_from, issue_to, columns, order, use_cache)`:
- `issue_from` and `issue_to` are inclusive dates (`"2015-12"` includes loans issued Dec‑2015).
- `sample_fraction`: keeps rows with `sample_bucket(loan_id, seed) < fraction·10000`. The
  bucket is the `lowbias32` integer hash of `(loan_id + seed·2654435769) mod 2³²`. DuckDB
  SQL (UBIGINT arithmetic) and numpy (`loader.sample_bucket`) compute it identically, so
  subsets are reproducible on any machine and DuckDB version.
  `subsample(df, fraction, seed)` applies the same rule to an in‑memory table. The filter
  costs about 0.2 s on the full table.
- `max_rows` applies after the filters, in one of three orders:
  - `order="issue_d"` (default): `ORDER BY issue_d, loan_id`, which gives the **earliest**
    loans. These are the ones most likely resolved by the 2019‑03 snapshot, but the first
    200k loans end in Oct‑2013. With the default `train_end=2015‑01‑01` that leaves no
    stream, so use `order="random"` or `sample_fraction` for a subset that spans the years.
  - `order="random"`: `ORDER BY hash` — a seeded, reproducible random subset.
  - `order="file"`: file order. DuckDB stops reading once `max_rows` rows qualify (about 1 s).
- `columns`: a subset of retained variables (`loan_id` is always included). Any other
  name raises `DataSourceError`, which enforces data minimisation.

**Parquet cache.** `build_cache()` writes the full canonical table once as zstd parquet
(148 MB), sorted by `(issue_d, loan_id)`, to `data/processed/
accepted_2007_to_2018Q4.canonical.v3.parquet`. A JSON sidecar records the source size,
mtime and `LOADER_VERSION`, so the cache goes stale when the CSV changes. With
`cache=True` (default), the first full‑scan query builds the cache and later queries
read it. `order="file"` with `max_rows` always reads the CSV. `to_parquet(df, path)` and
`from_parquet(path)` round‑trip any LoanTable, including `y_*` columns, with dtypes
intact. `describe_source(count_rows=True)` returns existence, size, column presence,
renames, the raw `count(*)` (about 10 s) and the cached row count.

**Timings** (laptop, DuckDB 1.5, full Kaggle file):

| Load | Time |
|---|---|
| CSV, `max_rows=200_000` (issue order) | 12 s |
| CSV, `sample_fraction=0.0885` (≈200k rows) | 12–13 s |
| CSV, `order="file"`, 20k rows | 1 s |
| build the parquet cache | 14 s |
| parquet, 200k rows (any order or sample) | 0.6–1.0 s |
| parquet, all 2.26M rows | 3 s |
| `derive_outcomes` on 2.26M rows | 7 s |
| `LoanStatePanel` on 2.26M rows | 4 s |

## 3. Outcomes (`derive_outcomes(df, arm_space=None, segmenter=None, cfg=DataConfig(), *, snapshot_date=None)`)

Returns a copy of the frame with the `y_*` columns (re)computed. Re‑running it on its own
output gives the same result. The status sets come from `schema.py`. Unknown statuses
count as unresolved and trigger a warning.

| Column | Definition |
|---|---|
| `y_resolved` (bool) | status in `STATUS_RESOLVED` (Fully Paid, Charged Off, Default, and the two "Does not meet the credit policy" variants) |
| `y_default` (float) | 1 for defaulted statuses, 0 for fully paid, NaN if unresolved |
| `y_recovery_rate` | defaults only. `recovery_basis="principal"` (default): `clip((total_rec_prncp + recoveries − collection_recovery_fee)/funded_amnt, 0, 1)`, so `E[L] = L·P(D)·(1−E[R])` is the expected principal loss. `"post_chargeoff"`: `(recoveries − fee)/(funded_amnt − total_rec_prncp)` clipped to [0, 1]; 1.0 when nothing was outstanding |
| `y_realized_revenue` | `total_rec_int + total_rec_late_fee` (revenue to date) |
| `y_realized_loss` | `max(funded_amnt − total_rec_prncp − (recoveries − fee), 0)` for defaults; 0 for fully paid; NaN while unresolved |
| `y_months_on_book` | whole months from `issue_d` to `last_pymnt_d`. If `last_pymnt_d` is missing: to the snapshot for unresolved loans, or the floor of 1 for resolved loans (defaults that never paid). Clipped to [1, term] |
| `y_resolution_date` | fully paid: `last_pymnt_d`. Default: `last_pymnt_d + chargeoff_lag_months` (4), or `issue_d + 4` if no payment was made. **Never later than the snapshot**, because a charge‑off seen at the snapshot happened by then. NaT while unresolved |
| `y_historical_arm` | `arm_space.nearest_many(loan_amnt, int_rate, term)`, evaluated once per distinct triple (identical results); −1 if amount or rate is missing |
| `y_group` | `segmenter.assign(df)`, e.g. `"B|debt_consolidation|RENT"` |

`temporal_split(df, train_end, *, stream_end=None)` returns `(issue_d < train_end, train_end ≤ issue_d [≤ stream_end])`.
`summarize(df)` returns a JSON‑ready dict with `n`, issue range, `status_counts`,
`n_resolved`, `resolved_share`, `n_defaults`, `default_rate` (among resolved),
`mean_recovery`, `default_rate_by_grade`, `default_rate_by_issue_year`, `mean_int_rate`,
`mean_loan_amnt` and `term_mix`. It works with or without the `y_*` columns.

## 4. Credit‑state panel (`LoanStatePanel`, SPEC §2.6 / G.2)

Lending Club is a snapshot, not a panel. Each loan's monthly path over S = {C, D, R, L}
is rebuilt deterministically **[decision]**. Notation: `m_last` = months from issue to
the last payment; `lag` = `MCTMConfig.chargeoff_lag_months` (4).

| loan_status | Path (run‑length segments) |
|---|---|
| Fully Paid, never late (+ policy variant) | `C × k` with k = `clip(m_last, 1, term)`, then `R × 1` under `EXTENDED`. Under `PROPOSAL` the payoff of a never‑delinquent loan is right‑censoring, so there is no C→R count and P[C,R] = 0 |
| Fully Paid with late fees (+ variant) | **repayment following delinquency**, under **both** structures: `C × (k−1)`, then `D × 1` in the payoff month, then `R`. This gives C→C ×(k−2), C→D, D→R; just D→R when k = 1 |
| Charged Off / Default (+ variant) | `C × clip(m_last, 0, term)`, then `D × lag`, then `L`. This gives C→C ×(m_last−1), C→D, D→D ×(lag−1), D→L. A loan that never paid is D from month 1 |
| Current, never late | `C × months on book` (to the last payment, else to the snapshot; clipped to [1, term]); censored in C |
| Current with late fees | **cured delinquency**: one C→C is replaced by C→D + D→C, i.e. a single D month is inserted at the midpoint of the performing run (needs at least 2 C months); censored in C |
| In Grace Period, Late (16‑30 days) | `C × m_last`, then `D × 1`; censored in D |
| Late (31‑120 days) | `C × m_last`, then `D × k`, where k = months since the last payment, clipped to [1, lag] (the cap only affects 18 real loans); censored in D |

**Why late fees decide between D→R and D→C.** Table 6 defines *Recovered* as
"repayment/recovery following delinquency". A loan with `total_rec_late_fee > 0` was
certainly delinquent at some point.
- If it is **Fully Paid**, it is literally repayment following delinquency. Its episode is
  placed immediately before payoff and resolves D→R, so PROPOSAL no longer censors these
  loans.
- If it is **Current** (still on book), the delinquency certainly **cured**, which is D→C.

So D→C is estimated from the population where a cure is unambiguous, and D→R from the
population where repayment after delinquency is unambiguous. Late, grace and charged‑off
loans keep their rules.

- Paths are stored as at most 4 `(state, run_length)` segments per loan, never as a dense
  month grid. Counts come from vectorised `np.bincount`; 2.26M loans take 4 s.
- `counts[i, j]` (indexed by `State`), `n_loans`, `n_loan_months`, `n_censored` (paths
  ending in C or D), `n_skipped` (unknown statuses) and `counts_by_cohort[issue_year]` are
  always computed. `paths` (`loan_id`, `segments`) is only kept with `keep_paths=True`.
- **Invariants**, checked by `validate()` and the tests:
  - rows R and L are zero (absorbing);
  - `counts.sum() == n_loan_months − n_loans`;
  - `PROPOSAL` has zero C→R and C→L;
  - `EXTENDED − PROPOSAL` is non‑zero only at C→R, where it equals the number of
    *never‑late* fully paid loans;
  - `counts[D,R]` equals the number of fully paid loans with late fees (one D→R each).
- **P[D,R] is data‑driven.** On the real file the 24,097 fully paid loans with late fees
  give 24,097 D→R transitions, and the 18,561 current loans with late fees give 18,561 D→C.
  Previously D→R was never observed and came only from the MCTM prior.
- `transition_counts_for(df)` recomputes counts for a holdout subset; counts are additive
  over subsets. `paths_to_counts(segments)` converts one path or many.
- `interim_state(row, month)` returns `(State, months_paid)` at loan‑month `month`
  (1‑based) under the same heuristic.
  - `months_paid` counts the C months so far.
  - For a loan repaid after delinquency, the D payoff month counts once the loan reaches R.
    A loan with 12 payment months and late fees is C for months 1–11, `(D, 11)` in month 12,
    and `(R, 12)` from month 13. Every repaid loan therefore ends with `months_paid` equal
    to its number of payment months.
  - After the path ends: a fully paid loan is `RECOVERED` under both structures (payoff is
    observed), a charge‑off is `LOSS`, and a censored loan carries its last state forward.
  - It is a pure‑Python twin, about 40 µs per call; `panel.interim_states(df, month)` is the
    vectorised version. Scalar and vectorised versions are tested to agree.
- `empirical_matrix()` gives the row‑normalised counts; `to_dict()` gives the JSON for the API.

Reconstructed one‑step matrix (row‑normalised counts = MLE), real data versus synthetic.
The D row is identical under both structures; only the C row differs (C→R).

| row | real 2.26M | synthetic 200k |
|---|---|---|
| C (PROPOSAL): C, D, R, L | 0.9918, 0.0082, 0, 0 | 0.9932, 0.0068, 0, 0 |
| C (EXTENDED): C, D, R, L | 0.9676, 0.0080, 0.0244, 0 | 0.9699, 0.0066, 0.0234, 0 |
| **D: C, D, R, L** | **0.0161, 0.7286, 0.0210, 0.2343** | 0.0182, 0.7269, 0.0246, 0.2304 |

Real D‑row counts: D→C 18,561 · D→D 837,565 · D→R 24,097 · D→L 269,360. Under the old rule
(episode at the midpoint, no D→R) the real row was 0.037 / 0.729 / 0 / 0.234.

## 5. Synthetic generator (`SyntheticLendingClub`)

`SyntheticLendingClub(seed=0).generate(n, start="2012-01", end="2018-12", snapshot="2019-03-31", shift_at=None, shift=None)`
returns a canonical LoanTable. It is fully vectorised: 200k loans take about 1 s. The
aliases are `generate_table`, module‑level `generate(n, seed, **kw)`, and
`load_loans("synthetic", ...)`. CLI:
`python scripts/make_synthetic.py --n 50000 --out data/processed/synthetic_50k.parquet [--shift ['{json}']] [--shift-at 2016-06] [--with-outcomes]`.

1. **Latent creditworthiness** z ~ N(0,1). Each risk‑bearing observable is a noisy view
   `x_j = ρ_j z + √(1−ρ_j²) ε_j`, with ρ = 0.70 (FICO), −0.35 (DTI), 0.30 (log income),
   −0.45 (revolving utilisation), −0.40 (inquiries).
2. **Observables.**
   - FICO = 655 + exp(3.555 + b·x), with b = 0.83 below the median and 0.68 above. It is
     banded to multiples of 5 within [660, 845], and `high = low + 4`.
   - Income is lognormal (median 65k, σ 0.55). DTI is lognormal (median 17, σ 0.45,
     clipped to [0, 45]). Utilisation is logistic, with 0.2% over‑limit outliers.
     Inquiries are Poisson.
   - The other bureau fields follow Poisson, negative‑binomial or lognormal marginals that
     depend on z.
   - Purpose, emp_length, home_ownership and verification follow the LC 2007–2018 mixes.
     Verification depends on the amount.
   - 5% of loans are joint applications with the joint fields filled (NaN otherwise).
   - `inq_last_12m` and `all_util` are NaN before 2015‑12, as in the real file.
3. **Grade and price.**
   - Grading score `g = 0.55·z_eff + 0.835·ε`, cut at fixed normal quantiles so the A…G
     shares are 19/29/29/14/6/2/1 %, with an equal split into sub‑grades.
   - `int_rate` = the linear sub‑grade mean (A1 5.3% → G5 30.9%) + N(0, 0.4%).
4. **Loan.**
   - Amount is lognormal (median 12k, linked to income), rounded to 1000 (45%), 5000
     (15%) or 25 (otherwise), within [1000, 40000].
   - Term is 60 months with a probability that rises with the amount (25% overall); the
     intercept is set by deterministic quadrature.
   - Installment follows the annuity formula, and funded = amount.
   - Issue months are drawn with linearly growing arrivals (the last month gets 5× the first).
5. **Default.** `logit p = −2.33 − 0.9·z_eff + 6(int_rate − 0.13) + 0.35·[60m] +
   0.8·amount/income + purpose/home/verification effects`. The effects are centred on the
   nominal mixes. With c = −2.33 the lifetime PD is ~16%, and ~19% of loans resolved by
   the snapshot are defaults (real file: 20%; see the cohort table below). The first missed payment
   follows a Weibull(k = 1.7) with scale 17 (36‑month) or 24 (60‑month), truncated to the
   term.
6. **Prepayment.** 45% of non‑defaulters repay early, at the **front‑loaded** month
   `6 + floor(Beta(1, 2.5)·(term − 6))` ∈ [6, term−1]. The median is month 13 of a 36‑month
   loan. The rest repay at maturity (`prepay_share`, `prepay_beta_a`, `prepay_beta_b`).
   Front‑loading puts enough early payoffs into recent, mostly unresolved cohorts that
   their resolved default rates follow LC's censoring pattern; with uniform timing they
   were dominated by early charge‑offs (25–32%).
7. **Status at the snapshot.**
   - Defaulters whose last payment is more than `lag` months old are Charged Off (0.5% are
     Default).
   - Last payment 1 month ago: In Grace Period (65%) or Late (16‑30). 2…lag months: Late
     (31‑120).
   - Not yet defaulted: Current. Non‑defaulters are Fully Paid once their payoff month has
     passed.
   - 0.7% of performing loans are temporarily 1–3 months behind (late, later cured).
8. **Cash flows** come from the exact amortisation schedule up to the last paid month; a
   payoff adds the remaining principal.
   - Late fees of 15–45 go to ~2.5% of fully paid, 2% of current, 20% of charged‑off and
     25% of late loans.
   - Recoveries on charge‑offs = Beta(1.1, 10) × outstanding; the collection fee is 16% of
     recoveries.
   - `total_pymnt` = principal + interest + fees + recoveries.
   - Last FICO = FICO + N(0, 20), and 60–120 points lower for bad loans.
   - `num_tl_30dpd` = 1 for late statuses.

**Distribution shift.**
- `shift={"fico_range_low": -30, "dti": 6, "annual_inc": -8000}` (`DEFAULT_SHIFT`, used
  when only `shift_at` is given) applies from `shift_at` (default `"2016-06"`).
- Each delta is added to the generated feature **before** grade, rate and default are
  derived.
- Defaults depend on creditworthiness *as revealed by the features*, so the latent score
  moves by the regression of z on the shifted observables:
  `z_eff = z + Σ a_j Δx_j` (a = 0.52 FICO, −0.15 DTI, 0.12 log income, −0.21 utilisation,
  −0.18 inquiries, per latent sd).
- With the default shift, the shifted cohort's FICO drops 30 points, the grade‑A share
  falls from 19% to 11%, the mean rate rises from 13.1% to 14.9%, and its lifetime
  default share goes from 14.7% to 23.7%. The generator's `last_meta` and `df.attrs` report
  `lifetime_default_share_pre_shift` and `..._post_shift`.
- `grade_mix: "worse" | "better" | float` tilts grading by −0.5, +0.5 or the given number
  of sd. `latent: δ` shifts z itself.
- Any other numeric pre‑decision column can be shifted additively (it carries no risk
  loading). `installment` and `fico_range_high` are derived and refuse shifts.
- Every feature draws from its own named random stream (`default_rng([seed, crc32(name)])`)
  and all calibration constants are fixed. So **the pre‑shift cohort is bit‑identical with
  and without a shift**, a clean counterfactual for Sentinel experiments.

**Calibration against the real file** (synthetic: default settings, 200k loans):

| statistic | real (2.26M) | synthetic |
|---|---|---|
| default rate among resolved | 0.200 | 0.176 |
| default rate by grade A / C / E / G | 0.060 / 0.224 / 0.384 / 0.497 | 0.064 / 0.182 / 0.370 / 0.589 |
| resolved share | 0.596 | 0.575 |
| sub‑grade AUC (resolved) | 0.69 | 0.70 |
| mean recovery, principal basis | 0.366 | 0.341 |
| recoveries / funded (charged off) | 0.075 | 0.071 |
| fully paid with a late fee | 2.2% | 2.3% |
| charged off without any payment | 0.9% | 1.0% |
| months issue → last payment (charged off) | 16.1 | 13.8 |
| Current / Fully Paid / Charged Off / late shares | 0.389 / 0.476 / 0.119 / 0.015 | 0.413 / 0.474 / 0.101 / 0.012 |
| months issue → payoff (fully paid) | 22.1 | 23.4 |
| FICO mean / sd | 699 / 33.0 | 698 / 32.6 |
| int_rate mean; 60‑month share | 0.131; 0.288 | 0.132; 0.253 |
| loan_amnt mean / median | 15,047 / 12,900 | 14,253 / 12,000 |
| income median; DTI median; revol_util mean | 65,000; 17.8; 0.503 | 65,000; 17.0; 0.502 |
| MORTGAGE / RENT / OWN | 0.49 / 0.40 / 0.11 | 0.49 / 0.40 / 0.11 |

**Resolved default rate by issue year** (synthetic: mean of 4 seeds × 300k loans):

| issue year | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|
| real | 0.162 | 0.156 | 0.184 | 0.202 | 0.233 | 0.231 | 0.158 |
| synthetic, front‑loaded prepayment (current) | 0.146 | 0.148 | 0.160 | 0.164 | 0.221 | 0.216 | 0.151 |
| synthetic, uniform prepayment (before) | 0.144 | 0.149 | 0.161 | 0.169 | 0.248 | 0.316 | 0.279 |

- The 2016–2018 cohorts now track LC (RMSE about 1 point).
- Pre‑2015 cohorts are unchanged, because fully resolved cohorts do not depend on payoff
  timing.
- With the front-loaded prepayment alone the overall resolved rate fell from 19.1% to 17.5%
  (recent cohorts contribute more resolved payoffs), so the default intercept was moved from
  −2.45 to **−2.33**, which restores ~19% overall and lands every cohort within ~2 points of
  the real file (2012–18, seed 3, 300k loans: 16.1 / 15.8 / 17.5 / 18.0 / 23.5 / 23.5 / 16.7%
  vs real 16.2 / 15.6 / 18.4 / 20.2 / 23.3 / 23.1 / 15.8%).
- Real prepayment timing in fully resolved cohorts: 63% (36m) / 78% (60m) of fully paid
  loans repay early. Timing on [6, term−1] is only mildly front‑loaded (Beta MoM ≈ 1.16,
  1.21), and 5% repay before month 6.
- A larger share with uniform timing gives a similar early‑payoff mass. For censored
  cohorts this mass is what matters, and Beta(1, 2.5) with the 45% share fits the cohort
  table best: share 0.55–0.65 or a flatter Beta pushes 2016–18 down to 11–20%.

## 6. Real‑data facts (profiled on the Kaggle file)

- 2,260,701 rows × 151 columns; the last 33 rows are all NULL. `policy_code` is always 1.0.
- `issue_d` runs from 2007‑06 to 2018‑12; `last_pymnt_d` max is 2019‑03, hence the
  snapshot of 2019‑03‑31.
- Loans per year: 2012 53k, 2013 135k, 2014 236k, 2015 421k, 2016 434k, 2017 444k, 2018 495k.
- loan_status:

  | Status | Count |
  |---|---|
  | Fully Paid | 1,076,751 |
  | Current | 878,317 |
  | Charged Off | 268,559 |
  | Late (31‑120 days) | 21,467 |
  | In Grace Period | 8,436 |
  | Late (16‑30 days) | 4,349 |
  | Does not meet the credit policy. Status:Fully Paid | 1,988 |
  | Does not meet the credit policy. Status:Charged Off | 761 |
  | Default | 40 |

- Term: 36 months 71%, 60 months 29%. Joint applications: 5.3%. Grades A…G: 433k / 664k /
  650k / 324k / 136k / 42k / 12k.
- Resolved default rate is 19.98% overall. By issue year: 2012 16%, 2013 16%, 2014 18%,
  2015 20%, 2016 23%, 2017 23%, 2018 16% (censoring).
- Charged‑off loans without any `last_pymnt_d`: 2,313. Late (31‑120) loans: 0–4 months
  since the last payment (102 never paid). Current loans: last payment in Mar‑2019
  (826k) or Feb‑2019 (52k).
- About 5% of fully paid loans made their last payment after the scheduled term (capped
  at the term in the panel).
- Late fees: 24,097 fully paid loans carry a late fee (including the policy variant); in
  the panel these are repaid after delinquency, D→R. 18,561 current loans carry one; these
  are cured, D→C.
- Prepayment: in fully resolved cohorts (36m issued 2013–15, 60m issued 2012–13) 62.6% /
  78.3% of fully paid loans repaid at least 2 months early. 5.2% repaid before month 6.
  The early‑payoff month quantiles (10/25/50/75/90%) are 6/12/20/27/32 for 36m and
  8/16/28/43/52 for 60m.
- Among fully paid loans, 2.2% carry a late fee. Among charged‑off loans:
  `recoveries / funded` = 0.075 and the principal‑basis recovery is 0.366.
  `collection_recovery_fee` ≈ 15–18% of recoveries.
- FICO (`fico_range_low`) quantiles: 1% 660, 10% 665, 25% 675, 50% 690, 75% 715, 90% 745,
  99% 805; mean 698.6, sd 33.0.

## 7. Assumptions beyond the SPEC

- **Data types.** The canonical string dtype is pandas‑3 `str` (NaN‑missing).
  `y_group` uses it too, whereas `Segmenter.assign` itself returns `"string"`.
- **Months on book.** `y_months_on_book` for resolved loans with no `last_pymnt_d` is the
  floor (1), not "to the snapshot".
- **Resolution date.** `y_resolution_date` is clipped to the snapshot.
- **Panel heuristic.**
  - The panel's L month is `last payment + lag + 1`, while `y_resolution_date` follows the
    SPEC (`last payment + lag`).
  - The Late (31‑120) D‑run is capped at `lag`.
  - A late fee on a fully paid loan becomes a D payoff month resolving D→R under both
    structures. A late fee on a current loan becomes a cured D month (D→C) at the midpoint,
    needing at least 2 C months.
  - `interim_state` reports R after an observed payoff under both structures, and counts
    the D payoff month in `months_paid` once R is reached.
- **Loader.**
  - `order="issue_d"` is the default for `max_rows`.
  - `sample_fraction` uses `lowbias32` instead of DuckDB's `hash()`, for cross‑version
    reproducibility.
  - The parquet cache is built automatically on the first full‑scan load.
- **Synthetic generator.**
  - It uses the task's recovery model (Beta(1.1, 10) × outstanding) and the profiled
    late‑fee share (~2.5% of fully paid), not the SPEC's Beta(1.2, 6) and 8%.
  - The snapshot default is 2019‑03‑31, not 2018‑12‑31.
  - A loan is charged off once its last payment is more than `lag` months old (late for
    1…lag months), which matches the real late‑status profile.
  - The "Does not meet the credit policy" statuses (2007–2010 only) are never generated.
  - Early payoff timing is front‑loaded (Beta(1, 2.5) on [6, term−1], 45% share) and the
    default intercept is −2.33, giving an overall resolved default rate of ~19% (real 20%).
