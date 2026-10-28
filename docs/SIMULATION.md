# Replay simulation, shadow engine and evaluation report

This page documents the integration layer (SPEC §2.10–§2.12, proposal §3.6, G.3–G.7):
how historical loans are replayed as a stream of borrowers, how delayed feedback reaches
the policies, what the yardstick (oracle, regret) is, the run-directory file protocol the
Django API relies on, and the resume / pause / review semantics.

| Module (`core/bayesian_credit/`) | Main API |
|---|---|
| `simulation/environment.py` | `BayesianEnvironment` – posterior-predictive rewards, online learning from realized outcomes, hindsight oracle |
| `simulation/feedback.py` | `DelayedFeedbackQueue`, `FeedbackItem`, `FeedbackBatch` |
| `simulation/events.py` | event dataclasses, `encode_event` / `decode_event`, `EventSink`, `JSONLSink`, `KafkaSink` |
| `simulation/store.py` | `RunDirectory` (file protocol), `EventStore`, `Checkpoint`, atomic writers |
| `simulation/policies.py` | policy specs (`"thompson"`, `"thompson_se"`, …), `build_policies`, `risk_model_prior_provider` |
| `simulation/runner.py` | `SimulationRunner` (`run`, `resume`, `from_run_dir`, `pause`, `cancel`), `SimulationResult` |
| `simulation/cli.py` | `python -m bayesian_credit.simulation.cli {fit,run,resume,shadow,report}` |
| `shadow/engine.py` | `ShadowLendingEngine`, `ShadowReport`, `PortfolioSummary`, `eligibility_matrix` |
| `evaluation/report.py` | `EvaluationReport` (Table 11) |
| `pipeline.py` | `prepare_dataset`, `fit_models`, `FittedModels`, `evaluate_predictive`, `run_shadow_portfolio`, `run_simulation`, `build_report`, `resilience_check` |
| `scripts/run_pipeline.py` | end-to-end CLI with a readable summary |

## 1. The stream

* **Reference window** – loans with `issue_d < SimulationConfig.start`. It fits the PSI
  Sentinel's reference buckets (features `PSIConfig.features`, pre-decision only).
* **Stream** – `start ≤ issue_d ≤ end`, subsampled to `max_loans` with
  `numpy.random.default_rng(seed).choice(n, max_loans, replace=False)`, then ordered by
  `(issue_d, loan_id)`.
* **Epochs** – the sorted distinct issue months of the stream (`'YYYY-MM'`). An epoch's
  arrivals are the stream loans issued that month.

## 2. One epoch

```
(a) Sentinel     auto-acks due → evaluate the window → trip? (review / wait / monitor)
(b) Feedback     pop due items → value them (feedback_mode) → policy.update
                 environment absorbs every historical outcome observed this month
                 environment.refresh() every 12 epochs
(c) Decisions    contexts (pre-decision only) → every policy selects → log → schedule feedback
    Outputs      decisions/epoch=…, timeseries, sentinel, arms (every k epochs), checkpoint (every k epochs)
```

### (a) Sentinel and breaker

The window is the arrivals of the last `SimulationConfig.psi_window_months` epochs
(including the current one: the applications are in before any decision is made); its
columns go through `LeakageGuard.assert_allowed(..., PRE_DECISION)`. Windows with fewer
than `min_window_size` (50) rows are recorded but do not step the breaker.

**One sentinel, many policies.** Every policy is wrapped in its own `GovernedPolicy`; all
wrappers share one `PSISentinel` (the drift is a property of the data, not of a policy).
`GovernedPolicy` subscribes to the sentinel's breaker transitions and acknowledgements, so a
single `sentinel.evaluate()` per epoch freezes / unfreezes / rolls back every policy at
once. While the breaker is OPEN every policy decides through its fallback
(`PSIConfig.fallback`, default the greedy arm of the champion snapshot) and its updates are
dropped (counted in `n_dropped_frozen`). **[decision]** The historical reference policy uses
the `"historical"` fallback, so it keeps replaying what the lender actually did.

| `breaker_mode` | On a trip |
|---|---|
| `auto_ack` | review file written; acknowledged with `PSIConfig.auto_action` at the start of epoch `trip + auto_ack_epochs` (default 1) → one visible frozen epoch of `fallback=True` decisions |
| `pause` | status `awaiting_review`; the worker polls `reviews/<id>.ack.json` (and `control.json` cancel) every `poll_interval` (1 s) and continues the same epoch once acknowledged (no fallback decisions: nothing is decided while waiting) |
| `off` | `PSISentinel(monitor_only=True)`: readings only, the breaker never trips |

Acknowledgement actions (`ReviewAction`):

* `resume` – unfreeze (the governed policies do it on the sentinel's acknowledgement);
* `rollback` – `GovernedPolicy.handle_acknowledge` restores the champion snapshot taken at
  the trip (everything learned since the incident is discarded);
* `resegment` – `segmenter.refine()` (next dimension of `refinement_order`, e.g.
  `fico_band`); the environment's two-stage model is re-fitted on every loan resolved so
  far under the new keys; every contextual policy is `resegment(new_segmenter,
  env.prior_provider)`-ed (groups re-warm-started lazily; feedback of decisions made under
  the old segmentation updates the pooled level only); `ResegmentationApplied` is logged.

**[decision] Re-basing.** After an acknowledgement the Sentinel's reference is re-fitted on
the window that tripped it (the most recently evaluated window): the human has reviewed
and accepted the new population, so later readings measure drift *from the acknowledged
regime*. Without it a persistent shift (the synthetic 2016-06 shift, or real 2015+ drift
against a 2007–2014 reference) would re-open the breaker on every reading and the review
queue would never drain. A trip during a transition (window half shifted) can re-trip once
the window is fully shifted; the second acknowledgement settles it. Configurable through
`SimulationRunner(rebase_on_ack=...)`.

### (b) Feedback

**Timing.** A loan is *observed* at `obs = max(resolution month, issue month + 1)` (the
`+1` because decisions of epoch `e` are made after epoch `e`'s feedback is released;
`y_resolution_date` is capped at the snapshot). For a decision on loan `i` issued in month
`t`:

* **interim** feedback at months `m = 1 … interim_months` (release `t + m`) while the loan
  is still open: `panel.interim_states(stream, m)` gives the reconstructed credit state and
  installments paid; scheduling stops at the first absorbing state (R / L), at the
  observation month, or after `end`;
* **terminal** feedback at `obs` if `obs ≤ end`. Loans still open at the end of the stream
  never deliver terminal feedback (real censoring).
* the decline arm is not routed through the queue: its reward is identically 0 and its
  prior is pinned at `N(0, min_sigma²)`, so an update would be a no-op.

Interim signals always come from the borrower's **real** repayment path, valued for the
**chosen** arm: `env.interim_reward(g, arm, state, month, months_paid)` =
`reward_fn.interim(policy_arm, state, month, months_paid, risk(g, arm), mctm).mean`, applied
with the policy's `interim_weight` (default 0.5).

Terminal rewards per `feedback_mode`:

| mode | chosen arm == historical arm | otherwise |
|---|---|---|
| `simulated` (default) | posterior-predictive draw | posterior-predictive draw |
| `replay` (Li et al. 2011) | realized `reward_fn.terminal(row)` | skipped (`n_skipped_replay`); interim skipped too |
| `hybrid` | realized | posterior-predictive draw |

A posterior-predictive draw is `env.draw_terminal_reward(g, arm, rng)`: one draw
`p ~ Beta(Stage 1)`, `R ~ Beta(Stage 2)` of `(g, arm)` pushed through
`reward_fn.sample(policy, p, R, mctm)` (its mean is `expected().mean`). Its timing follows the
real loan (a simulated 60-month arm on a loan that resolved after 20 months is rewarded in
month 20). The environment uses the loan's group under the environment's *current*
segmentation. Terminal feedback rows are written to `feedback/epoch=…parquet`; they fill the
decision log's `simulated_reward`.

**The environment learns from reality.** Independently of the policies, every historical
loan of the dataset (reference loans resolving after `start` included) whose observation
month is the current epoch updates the environment's two-stage model with its realized
outcome (`y_historical_arm`, `y_default`, `y_recovery_rate`). **[decision]** The environment's
initial model is fitted *as of the stream start* – on loans observed before `start` – so the
simulation never uses an outcome before it could have been known (the reference fit used for
predictive metrics and the shadow engine uses all training loans resolved by the snapshot,
as the SPEC's protocol prescribes). Its structured leaf prior (policy effect) is fitted on
the same as-of rows, "elapsed term" being judged at the stream start; the hindsight
yardstick fits its own, and a breaker re-segmentation re-fits it with the refined groups.
Child priors are refreshed every 12 epochs (the tilt is re-applied to the new anchors). The
MCTM and the prepayment table (`InterestRealization`, MATH.md §2.8) are the reference
fit's, not updated online. Every expected, sampled and interim reward of `(g, a)` uses
`ρ(grade(g), term(a))`, in the environment and in the yardstick alike.

### (c) Decisions

Contexts are built from the stream's `PRE_DECISION` columns only (`loan_id` is the row key,
`issue_d` the time index) through `BorrowerContext`, whose constructor runs the
`LeakageGuard`; the runner also asserts the guard on the context frame every epoch.
`eligible = ArmSpace.eligible_many(annual_inc, dti, reference_rate)` — the lending
guardrails of docs/MATH.md §2.7: installment ≤ 0.20 · monthly income, amount ≤ 0.50 · annual
income, total DTI (`dti/100` + the arm's payment-to-income) ≤ 0.43 (the QM ceiling), and
rate within ±0.05 of the borrower's reference rate (`FittedModels.reference_rates`: training mean `int_rate` of
the sub-grade → grade → global; no corridor when the fit has no reference rates). The null
arm is always eligible; a borrower left without a lending arm keeps the one closest to the
reference rate with the smallest installment. The masks are a pure function of the stream
and the fit, so a resumed run recomputes them identically (`runner.elig`,
`runner.reference_rate`). **[decision]** The historical policy's eligibility mask always
contains the loan's historical arm (the lender did extend it); the guardrails constrain
counterfactual offers. Policies call `prime_groups()` once per epoch.

## 3. Yardstick: expected reward, oracle, regret

A **hindsight** `BayesianEnvironment` is fitted with the original segmentation, an MCTM on
the full panel and the reference fit's prepayment table. **[decision]** Its risk model uses
the resolved loans (train + stream) whose term had elapsed by the snapshot
(`SimulationConfig.hindsight_sample = "elapsed"`). Using every resolved loan (`"resolved"`,
the original protocol) biases it towards default: only the early defaulters of 2016–2018
have resolved. On the synthetic data the stream's resolved loans default at 26 % against
15 % for elapsed ones, and the historical arms' mean `p` is 0.24 instead of 0.18. Once
rewards are prepayment-aware, that bias made every policy's expected reward negative. For
each decision:

* `expected_reward` = hindsight risk-adjusted expected reward (`mean − λ·std`) of the
  chosen arm for the loan's original group (0 for decline);
* `oracle_reward` = the best such value over the admissible arms (the guardrail-eligible
  arms ∪ the historical arm — the same arms the policies choose from), the same for every
  policy;
* `regret = oracle − expected ≥ 0` (pseudo-regret); `realized_reward` =
  `reward_fn.terminal(row)` if the chosen arm equals the historical arm and the loan has
  resolved, else NaN.

The yardstick never changes during a run (re-segmentation does not touch it), so curves are
comparable across policies and over time.

## 4. Determinism

All randomness comes from generators seeded per epoch:

* policy `p` in epoch `i`: `numpy.random.default_rng([seed, i, p])`;
* environment draws in epoch `i`: `numpy.random.default_rng([seed, i, 7])`.

Checkpoints therefore contain no bit-generator state. Everything else is a pure function of
the dataset, the configuration and the checkpointed state (caches are invalidated on every
state change and the two environment shortcuts are bit-identical to the reference code –
see `test_environment.py`). `test_runner.py` kills runs after 5 and after 6 epochs
(exception in the progress callback), SIGKILLs a subprocess worker mid-run, cancels via
`control.json`, pauses and resumes, and asserts that the decision logs, time series and
Sentinel tables equal an uninterrupted run exactly.

## 5. Run-directory protocol

```
<run_dir>/config.json          {"framework", "simulation", "dataset": {"path", ...}, "fit_dir", "created_at"}
<run_dir>/progress.json        {"status", "epoch_index", "n_epochs", "current_epoch", "pct", "message",
                                "updated_at", "breaker_state", "pid", "started_at", "finished_at", "error"}
<run_dir>/control.json         API → worker {"command": "pause"|"resume"|"cancel", "at"}
<run_dir>/reviews/<id>.json    worker → API ReviewRecord + epoch_index, policies, breaker_mode, segmentation,
                               auto, auto_action, auto_ack_epoch, created_at (+ ack, rebased, segmentation_after)
<run_dir>/reviews/<id>.ack.json API → worker {"action": "resume"|"resegment"|"rollback", "note", "at"}
                               (written by the worker itself for auto acknowledgements, with "auto": true)
<run_dir>/events.jsonl         {"ts", "type", "epoch", "payload"} per line
<run_dir>/decisions/epoch=YYYY-MM.parquet
<run_dir>/feedback/epoch=YYYY-MM.parquet   terminal feedback released in that epoch
<run_dir>/arms/epoch=YYYY-MM.parquet
<run_dir>/timeseries.parquet, sentinel.parquet   rewritten every epoch
<run_dir>/checkpoints/latest.pkl (+ epoch=YYYY-MM.pkl, last 3)
<run_dir>/runtime.json         sessions (pid, wall/CPU seconds, peak RSS), totals, phase_seconds
<run_dir>/report.json          EvaluationReport (written when the run completes)
<run_dir>/dataset.parquet, fit/  only when the runner was given in-memory data / models
```

Every file is written to a temporary file in the same directory, fsync-ed and moved into
place with `os.replace`, so readers never see partial files and SIGKILL never corrupts a
checkpoint.

* **Status values**: `queued` (written by `RunDirectory.create`), `running`, `paused`,
  `awaiting_review`, `completed`, `failed` (with `error`), `cancelled`. `pct` is 0–100.
* **control.json is a latch**: at every epoch boundary the worker reads it; `pause` →
  checkpoint (if not just written), status `paused`, poll every `poll_interval` until the
  command is no longer `pause`; `cancel` → checkpoint, status `cancelled`, exit 0. The API
  should write `resume` before restarting a cancelled or killed worker.
* **SIGTERM** (`kubectl delete pod`, `docker stop`) → the current epoch finishes, then
  checkpoint + `cancelled`. **SIGKILL** leaves the last checkpoint intact; `resume` replays
  from it.
* **Events**: `RunStarted`, `RunResumed`, `EpochStarted`, `SentinelEvaluated`,
  `BreakerTripped`, `ReviewAcknowledged`, `ResegmentationApplied`, `FeedbackReleased`,
  `CheckpointWritten`, `EpochCompleted`, `RunPaused`, `RunCancelled`, `RunCompleted`,
  `RunFailed` (payload keys in `simulation/events.py`). Per-borrower events are not logged
  (the decision parquet holds them). Epochs replayed after a resume appear twice in the log.

Table schemas:

| file | columns |
|---|---|
| decisions | `epoch, policy, loan_id, group, arm, amount, rate, term, declined, expected_reward, oracle_reward, realized_reward, fallback, simulated_reward, historical_match, meta_group_new` |
| timeseries | `epoch, policy, n_decisions, reward, cum_reward, realized, cum_realized, regret, cum_regret, rolling_reward (3 epochs), decline_rate, fallback_rate, mean_amount, mean_rate, n_feedback_interim, n_feedback_terminal, pending_feedback, breaker_state, psi_aggregate, n_groups, epoch_index, sentinel_level, feedback_reward, n_skipped_replay, n_dropped_frozen` |
| arms | `epoch, policy, group, arm, mu, sigma, n, selections` – pooled level `"*"` and the 10 groups with most arrivals, every `posterior_snapshot_every` epochs and at the last epoch |
| sentinel | `epoch, feature, psi, level, aggregate, breaker_state, n_current, skipped` |
| feedback | `epoch, policy, loan_id, decision_epoch, arm, reward, source (simulated / realized / skipped_replay), delivered` |

`reward`/`regret` sum the epoch's decisions; `realized` sums the realized rewards of the
epoch's decisions where known (same convention as `evaluation/policy.py`); `feedback_reward`
sums the rewards delivered to the policy in that epoch (interim × weight + terminal).

**Checkpoint** – a pickled plain dict: `epoch_index, epoch, policies {name: inner state},
governed {name: wrapper state}, sentinel, environment, queue (numpy columns), segmenter,
metrics_state (cumulative sums, time-series rows, Sentinel rows, pending auto-acks,
observation pointer, …), config_hash, created_at`.

## 6. Resume semantics

`SimulationRunner.resume(run_dir)` = `from_run_dir` (config.json → dataset parquet + saved
fit, or a re-fit when `fit_dir` is null) + `run(resume=True)`:

1. load `checkpoints/latest.pkl` (falls back to the newest readable stamped copy); refuse it
   (`CheckpointError`) if its config hash differs from `config.json`;
2. delete decisions / feedback / arms files of later epochs; truncate the time-series and
   Sentinel tables to the checkpoint;
3. restore `PSISentinel.from_dict` first, then the environment, then every
   `GovernedPolicy.from_state(state, arm_space, sentinel=…)` with the restored environment's
   prior provider, then the queue and the metrics state;
4. log `RunResumed` and continue at `epoch_index + 1`. Resuming a completed run only
   rebuilds the report. Without a checkpoint the run restarts from the first epoch.

A cancel while `awaiting_review` happens mid-epoch, so no new checkpoint is written there;
the next resume replays from the last boundary checkpoint (and finds the ack file if it was
written in the meantime).

## 7. Shadow engine

`ShadowLendingEngine(two_stage, mctm, reward_fn, arm_space, segmenter, n_samples=2000,
reference_rates=None, interest_realization=None)` precomputes, per group, all arms'
closed-form mean / std / risk-adjusted score (prepayment-adjusted: column `realization` =
ρ), p(default) mean + HDI, E[R], and Monte-Carlo quantiles / `P(reward > 0)` from
`n_samples` joint draws (one `rng.beta` call per group; generator seeded per group).
`FittedModels.shadow_engine()` passes the fit's reference rates and prepayment table.
`evaluate(context, …, reference_rate=None)` returns a `ShadowReport` (arm table with
`eligible` under every guardrail, recommended arm, historical arm and its expected /
realized reward, uplifts, lendable, and `to_dict()["eligibility"]`: `reference_rate,
n_eligible, n_eligible_lending, historical_admitted, guardrails, rate_corridor_applied`).
`evaluate_portfolio(table, max_loans, seed)` is fully vectorised over the group tables (200k
loans in ~6 s) and returns the per-loan frame (with `reference_rate` and
`n_eligible_lending`) plus a `PortfolioSummary` (totals, by grade, by group, inclusion
table, outcome classes, top-50 loans, `recommendation_mix` — recommended vs historical arms
by term, rate bucket and amount bucket, mean rate / amount, share at the top of the grid,
gap to the reference rate — `eligibility`, `uplift_decomposition` — the expected profit
of the best admissible arm keeping the historical term and amount (repricing only), keeping
the historical term (+ resizing), and unrestricted (+ term choice) — and `by_risk_aversion`:
for λ ∈ {0, 0.5, 1} (`risk_aversions=`), the policy chosen by `mean − λ·std` on the same
tables and eligibility, with its expected profit, uplift, mean amount / rate, 36-month share
and decline share). Outcome classes are exclusive with precedence `declined > rescued > upsized >
downsized > repriced_down > repriced_up > unchanged`; the boolean flags are not exclusive.
`uplift_expected` is the risk-adjusted score difference (= expected-reward difference for the
default λ = 0) and is ≥ 0 because the historical arm is always admissible.

## 8. Evaluation report

`EvaluationReport.from_run_dir(run_dir, fitted=…, shadow_summary=…, timings=…,
resilience=…).build()` returns `dimensions` (one detailed section per Table 11 row),
`table11` (the same as `{key, label, axis, metrics: [{key, label, value, unit, better,
baseline, baseline_label, by_policy, note}]}`), `policy_summary`, `runtime`, `config`,
`config_hashes`, `dataset`, `status` (`final` once completed) and `generated_at`. Predictive
metrics use the **elapsed-term** holdout (resolved loans with `issue_d + term ≤ snapshot`) as
primary and all resolved holdout loans as secondary, plus a per-vintage breakdown: recent
vintages' resolved loans are dominated by early defaults.

## 9. Running it

```bash
python scripts/run_pipeline.py --source synthetic                        # 200k loans, 50k streamed, 6 policies (~75 s)
python scripts/run_pipeline.py --source lending_club --max-rows 300000   # real data (~80 s with the parquet cache)
python -m bayesian_credit.simulation.cli fit --dataset d.parquet --config cfg.json --out fit/
python -m bayesian_credit.simulation.cli run --run-dir runs/x            # after RunDirectory.create(...)
python -m bayesian_credit.simulation.cli resume --run-dir runs/x
python -m bayesian_credit.simulation.cli shadow --fit-dir fit/ --dataset d.parquet --out shadow/
python -m bayesian_credit.simulation.cli report --run-dir runs/x
```

The API prepares a run with `RunDirectory.create(path, framework_cfg, simulation_cfg,
{"path": dataset_parquet}, fit_dir)` and launches `cli run`; `chaos` is a SIGKILL followed
(later) by `cli resume`. A runner constructed on a directory whose `config.json` already
holds the same configuration keeps that file (extra keys such as API ids are preserved) and
only fills in a missing dataset path / fit directory.

**Cost profile** (12-core laptop, 50k loans × 6 policies = 300k decisions, 48 epochs): about
45 s of simulation, of which ~25 s are the bandits' own `update()` calls for ~2.8M interim
and ~120k terminal observations and ~12 s their `select()` calls; the environment's
valuation, queue and I/O take the rest. Peak RSS of the real-data run (~2 GB) is set by the
loader's DuckDB scan of the 2.26M-row cache; the modelling stages stay around 1 GB.

## 10. Known modelling caveat (and what was done about it)

Originally the Stage-1 leaf posterior of an arm never observed for a group equalled the
group's leaf prior (shrinkage to the group mean), and the reward has no price-response or
adverse-selection term, so a rate or amount nobody in the group was ever offered looked
exactly as safe as the group average and the optimisers converged on "largest affordable
loan at the highest rate" (uplift ≈ +1,500 %). Two documented decisions now address it
(SPEC §7): the **structured leaf prior** (MATH.md §2.6) makes unobserved arms riskier with
rate, 60-month term and payment-to-income, and the **lending guardrails** (§2.7) keep offers
affordable and priced within ±5 points of the borrower's sub-grade reference rate.

The reward used to value the *term* and *size* levers generously as well. Two further
decisions address that: a **prepayment-aware expected reward** (SPEC §7.3, MATH.md §2.8),
because fully-paid LC loans realise ≈ 82 % / 72 % of their 36 / 60-month scheduled interest,
and **recovery by term** (SPEC §7.4, MATH.md §2.9), because 60-month defaults recover less
principal. The guardrails now use the standards-based defaults (QM 43 % DTI, 20 % PTI,
50 % amount-to-income). What remains is that revenue grows linearly in the amount while
the fitted payment-to-income effect is moderate, so the shadow engine still recommends
larger loans than the lender issued. `PortfolioSummary.uplift_decomposition` separates
repricing, resizing and term choice; `recommendation_mix`,
`bandit_performance.selection_mix` and `by_risk_aversion` show how the mix moves with the
risk appetite λ.
