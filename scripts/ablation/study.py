#!/usr/bin/env python3
"""Reproduce the DSA4900 variable-ablation study (pilot and full dataset) through the platform API.

The study (docs/REPRODUCE.md): four variable subsets - all variables, no FICO, no FICO and no
Lending Club grade, thin file - each ingested from the Lending Club 2007-2020Q3 export, a warm fit
trained on the six years before 2013-06, and two bandit-lab simulations (warm from the fit, and a
fully cold start with its own replay world) replaying the borrowers issued 2013-06 .. 2020-09 with
all seven policies, hybrid feedback and the PSI Sentinel auto-acknowledging with a capped
resegmentation. Every setting is pinned below; the same API drives the Docker, local-CLI and
Kubernetes installs, so the script is identical for all three (scripts/reproduce_ablation.sh wraps
the mode-specific set-up).

Standard library only, so it runs on any Python >= 3.10 without the project environment:

    python3 scripts/ablation/study.py plan --scale pilot                 # print the exact settings
    python3 scripts/ablation/study.py run --scale pilot --api http://localhost:8000/api/v1
    python3 scripts/ablation/study.py summarize --out data/runs/ablation-pilot
    python3 scripts/ablation/study.py compare RUN_A RUN_B                # check a reproduction

``run`` is resumable: it records every job in ``<out>/state.json`` as soon as it is queued and
picks up from there (a job that was still running is waited for, never duplicated). Jobs run
strictly one at a time. ``--mem-cmd`` is a shell command printing the worker's memory in bytes,
sampled while a job runs (the wrapper sets it per deployment mode).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------------------ data
#: Kaggle ethon0426/lending-club-20072020q1, stored by scripts/download_data.sh.
DATA_FILE = "Loan_status_2007-2020Q3.csv"
DATA_SHA256 = "5878af2a088f8ab5214c9337289fb8b5eb6c6338fd3f417b6cdc18513dc6f35f"
DATA_BYTES = 1_773_470_505
FULL_LOANS = 2_925_492  # rows kept by the loader (one row without an id is dropped)

# --------------------------------------------------------------------------------------- subsets
FICO = ["fico_range_low", "fico_range_high", "last_fico_range_low", "last_fico_range_high"]
GRADE = ["grade", "sub_grade"]
#: Credit-report (bureau) attributes a thin-file applicant does not have (frontend/src/lib/variables.ts).
BUREAU = ["delinq_2yrs", "delinq_amnt", "inq_last_6mths", "inq_last_12m", "open_acc", "total_acc", "revol_bal", "revol_util", "mort_acc", "pub_rec", "pub_rec_bankruptcies", "collections_12_mths_ex_med", "num_accts_ever_120_pd", "pct_tl_nvr_dlq", "num_sats", "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit", "bc_util", "all_util", "num_tl_30dpd", "num_tl_90g_dpd_24m", "mths_since_last_delinq"]
BASE_DIMS = ["purpose_group", "home_ownership", "income_band"]
SUBSETS = [  # lightest segmentation first, so a memory problem shows on the cheapest run
    {"key": "thin_file", "label": "Thin file", "exclude": FICO + GRADE + BUREAU, "dims": BASE_DIMS},
    {"key": "no_fico_grade", "label": "No FICO, no LC grade", "exclude": FICO + GRADE, "dims": BASE_DIMS},
    {"key": "no_fico", "label": "No FICO", "exclude": FICO, "dims": ["grade", *BASE_DIMS]},
    {"key": "all", "label": "All variables", "exclude": [], "dims": ["grade", "fico_band", *BASE_DIMS]},
]
SUBSET_KEYS = [s["key"] for s in SUBSETS]

# ----------------------------------------------------------------------------- model & simulation
POLICIES = ["thompson", "thompson_se", "ucb1", "epsilon_greedy", "greedy", "historical", "random"]
ISSUE_FROM, ISSUE_TO = "2007-06", "2020-09"
TRAIN_END = "2013-06"  # warm fits train on issue_d < 2013-06 (six years); the bandit lab streams the rest
SEED = 42
#: Pre-decision variables recorded for (nearly) every loan both before and after 2013-06. The other
#: pre-decision fields only appear part-way through the file (2012 bureau fields, inq_last_12m /
#: all_util from 2016, joint-application fields), which the PSI Sentinel would read as drift.
PSI_STABLE = ["annual_inc", "application_type", "dti", "emp_length", "fico_range_low", "fico_range_high", "home_ownership", "purpose", "verification_status", "delinq_2yrs", "delinq_amnt", "inq_last_6mths", "open_acc", "total_acc", "revol_bal", "revol_util", "installment", "pub_rec", "pub_rec_bankruptcies", "collections_12_mths_ex_med", "grade", "sub_grade"]
#: Breaker resegmentation order (sub_grade left out: 35 thin levels nested in grade) and cap.
REFINE = ["fico_band", "dti_band", "income_band", "emp_band", "verification_status", "application_type"]
REFINE_SOURCE = {"fico_band": "fico_range_low", "dti_band": "dti", "income_band": "annual_inc",
                 "emp_band": "emp_length", "verification_status": "verification_status",
                 "application_type": "application_type"}
MAX_REFINEMENTS = 2

#: Scale-specific settings. dataset_rows = seeded random sample of the file (None = every loan);
#: sim_loans = borrowers streamed through the bandit lab (None = every loan in the window).
SCALES = {
    "pilot": {"dataset_rows": 500_000, "sim_loans": 50_000, "checkpoint_every": 6, "name": "Pilot"},
    "full": {"dataset_rows": None, "sim_loans": None, "checkpoint_every": 12, "name": "Full"},
}
#: Worker memory each scale was run with (docs/REPRODUCE.md); the heaviest subset sets it.
WORKER_MEMORY_GIB = {"pilot": 6, "full": 9}
POLL_SECONDS = 15
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def segmentation(subset: dict) -> dict:
    order = [d for d in REFINE if REFINE_SOURCE[d] not in subset["exclude"]]
    return {"dimensions": subset["dims"], "refinement_order": order, "max_refinements": MAX_REFINEMENTS}


def psi_features(subset: dict) -> list[str]:
    return [f for f in PSI_STABLE if f not in subset["exclude"]]


def plan(scale: str, sim_loans: int | None = None) -> dict:
    """Every request the study sends, keyed by subset (the exact study definition). ``sim_loans``
    overrides the scale's number of streamed borrowers (recorded in the manifest)."""
    sc = dict(SCALES[scale])
    if sim_loans is not None:
        sc["sim_loans"] = sim_loans
    out: dict = {"scale": scale, "data": {"file": DATA_FILE, "sha256": DATA_SHA256},
                 "sim_loans": sc["sim_loans"], "subsets": {}}
    for s in SUBSETS:
        name = f"{sc['name']}"
        sim_common = {
            "policies": POLICIES, "feedback_mode": "hybrid", "start": TRAIN_END, "end": ISSUE_TO,
            "max_loans": sc["sim_loans"], "seed": SEED, "breaker_mode": "auto_ack",
            "checkpoint_every": sc["checkpoint_every"],
            "psi": {"features": psi_features(s), "auto_action": "resegment"},
        }
        out["subsets"][s["key"]] = {
            "dataset": {"name": f"{name} · {s['label']}", "source": "lending_club", "max_rows": sc["dataset_rows"],
                        "seed": SEED, "issue_from": ISSUE_FROM, "issue_to": ISSUE_TO,
                        "exclude_variables": s["exclude"]},
            "fit": {"name": f"{name} warm · {s['label']}", "train_end": TRAIN_END, "segmentation": segmentation(s)},
            "warm": {**sim_common, "name": f"{name} warm · {s['label']}"},
            "cold": {**sim_common, "name": f"{name} cold · {s['label']}", "cold_start": True,
                     "segmentation": segmentation(s)},
        }
    return out


# ------------------------------------------------------------------------------------------ http
class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        req = urllib.request.Request(
            self.base + path, method=method, data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.status, json.loads(r.read() or b"null")
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read() or b"null")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:  # API restarting
                if attempt == 4:
                    raise SystemExit(f"API unreachable at {self.base}: {e}") from None
                time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    def get(self, path: str):
        code, obj = self.call("GET", path)
        if code != 200:
            raise SystemExit(f"GET {path} -> {code}: {obj}")
        return obj

    def post(self, path: str, body: dict):
        code, obj = self.call("POST", path, body)
        if code != 201:
            raise SystemExit(f"POST {path} -> {code}: {json.dumps(obj)[:2000]}")
        return obj


# ------------------------------------------------------------------------------------------ run
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def memory_mb(cmd: str | None) -> float | None:
    if not cmd:
        return None
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, check=False).stdout.split()
        return int(out[-1]) / 2**20 if out else None
    except (subprocess.SubprocessError, ValueError):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Study:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api = Api(args.api)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "scale": args.scale, "datasets": {}, "fits": {}, "sims": {}}
        if self.state.get("scale") != args.scale:
            raise SystemExit(f"{self.state_path} belongs to a {self.state.get('scale')} study; use another --out")
        self.plan = plan(args.scale, args.sim_loans)
        recorded = self.state.get("manifest", {}).get("settings", {}).get("sim_loans", self.plan["sim_loans"])
        if self.state["sims"] and recorded != self.plan["sim_loans"]:
            raise SystemExit(f"{self.state_path} was run with sim_loans={recorded}; use the same or another --out")
        self.subsets = [s for s in SUBSETS if s["key"] in (args.subsets or SUBSET_KEYS)]

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def wait(self, path: str, done: set[str], label: str) -> tuple[dict, float, float]:
        t0, peak, last = time.time(), 0.0, time.time()
        while True:
            obj = self.api.get(path)
            mem = memory_mb(self.args.mem_cmd)
            peak = max(peak, mem or 0.0)
            st = obj.get("status")
            if st in done:
                secs = time.time() - t0
                log(f"{label}: {st} in {secs / 60:.1f} min" + (f" (worker peak {peak:.0f} MB)" if peak else ""))
                return obj, secs, peak
            if "epochs_done" in obj and time.time() - last > 600:
                last = time.time()
                log(f"{label}: epoch {obj['epochs_done']}/{obj['epochs_total']} after {(last - t0) / 60:.0f} min"
                    + (f", worker {mem:.0f} MB (peak {peak:.0f})" if mem else ""))
            time.sleep(POLL_SECONDS)

    def step(self, kind: str, key: str, path: str, body: dict, done: set[str], ok: str) -> dict:
        """Create (or resume waiting on) one job and record it."""
        rec = self.state[kind].get(key)
        if rec and rec.get("status") == ok:
            return rec
        if rec and rec.get("status") in done and rec.get("status") != ok:
            raise SystemExit(f"{kind[:-1]} {key} #{rec['id']} ended {rec['status']}; inspect it, then delete "
                             f"the entry from {self.state_path} to retry")
        if not rec:
            obj = self.api.post(path, body)
            rec = {"id": obj["id"], "status": obj.get("status")}
            self.state[kind][key] = rec
            self.save()
            log(f"{kind[:-1]} {key} #{rec['id']} queued")
        obj, secs, peak = self.wait(f"{path}{rec['id']}/", done, f"{kind[:-1]} {key}")
        rec.update(status=obj["status"], seconds=round(secs, 1), peak_mb=round(peak) or None)
        self.state[kind][key] = rec
        self.save()
        if obj["status"] != ok:
            raise SystemExit(f"{kind[:-1]} {key} #{rec['id']} ended {obj['status']} - stopping (later runs are "
                             "heavier); see docs/REPRODUCE.md 'Resources'")
        return {**rec, "obj": obj}

    def check_inputs(self) -> None:
        health = self.api.get("/health/")
        log(f"API {self.args.api}: version {health.get('version')}, core {health.get('core_version')}, "
            f"jobs {health.get('jobs', {}).get('backend')}")
        self.state.setdefault("manifest", {}).update(
            started_at=self.state.get("manifest", {}).get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            git_commit=git_commit(), api_health=health, driver_python=platform.python_version(),
            scale=self.args.scale, settings=self.plan,
        )
        if self.args.data_file:
            p = Path(self.args.data_file)
            if not p.is_file():
                raise SystemExit(f"--data-file {p} not found (scripts/download_data.sh)")
            if p.stat().st_size != DATA_BYTES:
                raise SystemExit(f"{p} is {p.stat().st_size} bytes, expected {DATA_BYTES}: a different export")
            if not self.args.skip_checksum:
                log(f"checking the SHA-256 of {p.name} (about 10 s)")
                digest = sha256(p)
                if digest != DATA_SHA256:
                    raise SystemExit(f"{p} SHA-256 {digest} != {DATA_SHA256}: results will not reproduce")
            self.state["manifest"]["data"] = {"file": p.name, "bytes": DATA_BYTES, "sha256": DATA_SHA256,
                                              "verified": not self.args.skip_checksum}
        self.save()

    def run(self) -> None:
        self.check_inputs()
        for s in self.subsets:
            key, p = s["key"], self.plan["subsets"][s["key"]]
            ds = self.step("datasets", key, "/datasets/", p["dataset"], {"ready", "failed"}, "ready")
            if "obj" in ds:
                o = ds["obj"]
                ds_rec = self.state["datasets"][key]
                ds_rec.update(n_loans=o["n_loans"], snapshot_date=o.get("snapshot_date"),
                              excluded=len(o["params"]["exclude_variables"]))
                expected = SCALES[self.args.scale]["dataset_rows"] or FULL_LOANS
                if o["n_loans"] != expected:
                    log(f"WARNING dataset {key}: {o['n_loans']} loans, expected {expected}")
                self.save()
        for s in self.subsets:
            key, p = s["key"], self.plan["subsets"][s["key"]]
            body = {**p["fit"], "dataset": self.state["datasets"][key]["id"]}
            self.step("fits", key, "/fits/", body, {"ready", "failed"}, "ready")
        for s in self.subsets:
            key, p = s["key"], self.plan["subsets"][s["key"]]
            for mode in ("warm", "cold"):
                body = dict(p[mode])
                if mode == "warm":
                    body["fit"] = self.state["fits"][key]["id"]
                else:
                    body["dataset"] = self.state["datasets"][key]["id"]
                self.step("sims", f"{key}:{mode}", "/simulations/", body, TERMINAL, "completed")
        self.state["manifest"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.save()
        log("study complete; writing results")
        summarize(self.api, self.out)


# -------------------------------------------------------------------------------------- results
def summarize(api: Api, out: Path) -> dict:
    """results.json / results.csv / RESULTS.md from the finished jobs recorded in state.json."""
    state = json.loads((out / "state.json").read_text())
    rows, runs = [], {}
    for key, info in state["sims"].items():
        if info.get("status") != "completed":
            continue
        subset, mode = key.split(":")
        sim = api.get(f"/simulations/{info['id']}/")
        ev = api.get(f"/simulations/{info['id']}/events/?type=BreakerTripped,ResegmentationApplied&limit=1000")
        trips = [e for e in ev["results"] if e["type"] == "BreakerTripped"]
        reseg = [e for e in ev["results"] if e["type"] == "ResegmentationApplied"]
        runs[key] = {"sim": info["id"], "seconds": info.get("seconds"), "peak_mb": info.get("peak_mb"),
                     "trips": len(trips), "trip_epochs": [e["epoch"] for e in trips],
                     "final_dims": reseg[-1]["payload"]["dimensions_after"] if reseg else None,
                     "n_epochs": sim["epochs_total"]}
        hist = next(h for h in sim["headline"] if h["policy"] == "historical")["cum_reward"]
        for h in sim["headline"]:
            oracle = h["cum_reward"] + h["cum_regret"]
            rows.append({
                "subset": subset, "mode": mode, "policy": h["policy"], "n_decisions": h["n_decisions"],
                "cum_reward": h["cum_reward"], "cum_regret": h["cum_regret"], "cum_realized": h.get("cum_realized"),
                "decline_rate": h.get("decline_rate"), "avg_regret": h["cum_regret"] / max(h["n_decisions"], 1),
                "oracle_share": h["cum_reward"] / oracle if oracle else None,
                "x_historical": h["cum_reward"] / hist if hist else None,
            })
    fits = {}
    for key, info in state["fits"].items():
        if info.get("status") == "ready":
            f = api.get(f"/fits/{info['id']}/")
            fits[key] = {k: f["metrics"].get(k) for k in ("auc", "brier", "ece", "log_loss", "n_train")}
            fits[key]["dims"] = f["config"]["segmentation"]["dimensions"]
    result = {"scale": state["scale"], "manifest": state.get("manifest", {}), "datasets": state["datasets"],
              "fits": fits, "runs": runs, "policies": rows}
    (out / "results.json").write_text(json.dumps(result, indent=1))
    if rows:
        with (out / "results.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "RESULTS.md").write_text(results_markdown(result))
    log(f"wrote {out / 'results.json'}, results.csv and RESULTS.md")
    return result


def results_markdown(r: dict) -> str:
    label = {s["key"]: s["label"] for s in SUBSETS}
    keys = [k for k in reversed(SUBSET_KEYS) if k in r["fits"] or any(x["subset"] == k for x in r["policies"])]
    cell = {(x["subset"], x["mode"], x["policy"]): x for x in r["policies"]}
    m = r.get("manifest", {})
    lines = [
        f"# {SCALES[r['scale']]['name']} ablation study",
        "",
        (f"Commit `{m.get('git_commit')}` · API {m.get('api_health', {}).get('version')} · "
        f"data `{DATA_FILE}` sha256 `{DATA_SHA256[:12]}…` · started {m.get('started_at')} · "
        f"finished {m.get('finished_at')}. Settings: `settings` in results.json (`study.py plan`)."),
        "",
        "## Warm fits",
        "",
        "| Subset | Segmentation | AUC | Brier | ECE |",
        "|---|---|---|---|---|",
    ]
    for k in keys:
        f = r["fits"].get(k)
        if f:
            lines.append(f"| {label[k]} | {' × '.join(f['dims'])} | {f['auc']:.4f} | {f['brier']:.4f} | {f['ece']:.4f} |")

    def table(metric: str, fmt) -> list[str]:
        out = ["| Policy | " + " | ".join(label[k] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for p in POLICIES:
            vals = []
            for k in keys:
                w, c = cell.get((k, "warm", p)), cell.get((k, "cold", p))
                vals.append(" / ".join(fmt(x[metric]) if x else "–" for x in (w, c)))
            out.append(f"| {p} | " + " | ".join(vals) + " |")
        return out

    lines += ["", "## Share of the hindsight oracle captured (warm / cold)", "",
              "Each subset's world is fitted under its own segmentation, so compare shares, not dollars.", ""]
    lines += table("oracle_share", lambda v: f"{v:.1%}")
    lines += ["", "## Multiple of the historical policy's expected reward (warm / cold)", ""]
    lines += table("x_historical", lambda v: f"{v:.2f}×")
    lines += ["", "## Runs", "", "| Run | Trips | Final dims | Minutes | Worker peak MB |", "|---|---|---|---|---|"]
    for key, run in r["runs"].items():
        lines.append(f"| {key} | {run['trips']} ({', '.join(run['trip_epochs'])}) | "
                     f"{' × '.join(run['final_dims'] or [])} | {(run['seconds'] or 0) / 60:.1f} | {run['peak_mb']} |")
    return "\n".join(lines) + "\n"


def compare(a: Path, b: Path, rtol: float) -> int:
    """Exit 0 when two results.json agree (fit metrics and every policy's totals) within rtol."""
    ra, rb = (json.loads((p / "results.json" if p.is_dir() else p).read_text()) for p in (a, b))
    bad = []

    def close(x, y):
        if x is None or y is None:
            return x == y
        return abs(x - y) <= rtol * max(1.0, abs(x), abs(y))

    for k in sorted(set(ra["fits"]) | set(rb["fits"])):
        for m in ("auc", "brier", "ece"):
            x, y = ra["fits"].get(k, {}).get(m), rb["fits"].get(k, {}).get(m)
            if not close(x, y):
                bad.append(f"fit {k} {m}: {x} vs {y}")
    ia = {(x["subset"], x["mode"], x["policy"]): x for x in ra["policies"]}
    ib = {(x["subset"], x["mode"], x["policy"]): x for x in rb["policies"]}
    for key in sorted(set(ia) | set(ib)):
        for m in ("n_decisions", "cum_reward", "cum_regret", "cum_realized", "decline_rate"):
            x, y = ia.get(key, {}).get(m), ib.get(key, {}).get(m)
            if not close(x, y):
                bad.append(f"{':'.join(key)} {m}: {x} vs {y}")
    for k in sorted(set(ra["runs"]) | set(rb["runs"])):
        x, y = ra["runs"].get(k, {}).get("trip_epochs"), rb["runs"].get(k, {}).get("trip_epochs")
        if x != y:
            bad.append(f"{k} breaker trips: {x} vs {y}")
    for line in bad:
        print("DIFF", line)
    print(f"{len(ia)} policy rows, {len(ra['fits'])} fits compared (rtol {rtol}): "
          + ("identical" if not bad else f"{len(bad)} difference(s)"))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print every request of the study as JSON")
    p.add_argument("--scale", choices=list(SCALES), required=True)
    p.add_argument("--sim-loans", type=int, default=None)
    r = sub.add_parser("run", help="run (or resume) the study")
    r.add_argument("--scale", choices=list(SCALES), required=True)
    r.add_argument("--api", default=os.environ.get("BC_API", "http://localhost:8000/api/v1"))
    r.add_argument("--out", default=None, help="results directory (default data/runs/ablation-<scale>)")
    r.add_argument("--subsets", nargs="*", choices=SUBSET_KEYS, help="run only these subsets")
    r.add_argument("--data-file", default=None, help="the Lending Club CSV, to verify size and SHA-256")
    r.add_argument("--skip-checksum", action="store_true", help="check the size only")
    r.add_argument("--mem-cmd", default=os.environ.get("BC_MEM_CMD"), help="shell command printing worker bytes")
    r.add_argument("--sim-loans", type=int, default=None,
                   help="borrowers streamed per simulation (default: the scale's, see SCALES)")
    s = sub.add_parser("summarize", help="rebuild results.json / .csv / RESULTS.md from state.json")
    s.add_argument("--api", default=os.environ.get("BC_API", "http://localhost:8000/api/v1"))
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare two results.json (exit 1 on a difference)")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--rtol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print(json.dumps(plan(args.scale, args.sim_loans), indent=1))
    elif args.cmd == "run":
        args.out = args.out or str(REPO / "data" / "runs" / f"ablation-{args.scale}")
        Study(args).run()
    elif args.cmd == "summarize":
        summarize(Api(args.api), Path(args.out))
    else:
        return compare(args.a, args.b, args.rtol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
