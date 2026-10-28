"""``python manage.py demo`` — seed a fully populated console in ~1-2 minutes.

Creates a synthetic dataset (20k loans with the default 2016-06 distribution shift), fits the
risk models, runs a shadow portfolio, a small replay simulation (Thompson vs ε-greedy, UCB1,
historical and random; max 3000 loans) and a short "governance drill" whose PSI breaker trips
at the shift and leaves a pending review — all inline (``JOBS_SYNC``).
"""

from __future__ import annotations

import time

from django.conf import settings
from django.core.management.base import BaseCommand
from rest_framework.test import APIClient


class Command(BaseCommand):
    help = "Seed a demo dataset, fit, shadow portfolio and simulation (runs inline)."

    def add_arguments(self, parser):
        parser.add_argument("--n", type=int, default=20_000, help="synthetic loans (default 20000)")
        parser.add_argument("--max-loans", type=int, default=3_000, help="simulation stream size (default 3000)")
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument(
            "--policies", default="thompson,epsilon_greedy,ucb1,historical,random", help="comma-separated policy specs"
        )
        parser.add_argument("--no-portfolio", action="store_true", help="skip the shadow portfolio job")
        parser.add_argument("--no-drill", action="store_true", help="skip the governance-drill simulation")

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        client = APIClient()
        resp = getattr(client, method)(f"/api/v1{path}", body or {}, format="json")
        data = resp.json()
        if resp.status_code >= 400:
            raise SystemExit(f"{method.upper()} {path} -> {resp.status_code}: {data}")
        return data

    def handle(self, *args, **opts):
        previous = settings.JOBS_SYNC, list(settings.ALLOWED_HOSTS)
        settings.JOBS_SYNC = True
        if "*" not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS = [*settings.ALLOWED_HOSTS, "testserver"]
        t0 = time.perf_counter()
        try:
            ds = self._call(
                "post",
                "/datasets/",
                {
                    "name": f"Synthetic demo ({opts['n'] // 1000}k, shift 2016-06)",
                    "source": "synthetic",
                    "n": opts["n"],
                    "seed": opts["seed"],
                    "shift": {"at": "2016-06", "deltas": {}},
                },
            )
            self.stdout.write(
                f"dataset #{ds['id']}: {ds['status']} — {ds['n_loans']} loans ({time.perf_counter() - t0:.1f}s)"
            )
            fit = self._call(
                "post", "/fits/", {"dataset": ds["id"], "name": "Demo fit (train < 2015-01)", "train_end": "2015-01"}
            )
            m = fit.get("metrics") or {}
            self.stdout.write(
                f"fit #{fit['id']}: {fit['status']} — AUC {m.get('auc') or 0:.3f}, Brier {m.get('brier') or 0:.4f}, "
                f"{fit.get('n_groups')} groups ({time.perf_counter() - t0:.1f}s)"
            )
            if not opts["no_portfolio"]:
                job = self._call(
                    "post", f"/fits/{fit['id']}/shadow/portfolio/", {"max_loans": 5000, "seed": opts["seed"]}
                )
                self.stdout.write(
                    f"shadow portfolio job #{job['id']}: {job['status']} ({time.perf_counter() - t0:.1f}s)"
                )
            sim = self._call(
                "post",
                "/simulations/",
                {
                    "fit": fit["id"],
                    "name": "Demo replay: Thompson vs baselines",
                    "policies": [p.strip() for p in opts["policies"].split(",") if p.strip()],
                    "start": "2015-01",
                    "end": "2018-12",
                    "max_loans": opts["max_loans"],
                    "seed": opts["seed"],
                    "breaker_mode": "auto_ack",
                    "checkpoint_every": 6,
                },
            )
            self.stdout.write(
                f"simulation #{sim['id']}: {sim['status']} — {sim['epochs_done']}/{sim['epochs_total']} epochs, "
                f"breaker {sim['breaker_state']}, {sim['pending_reviews']} pending review(s) ({time.perf_counter() - t0:.1f}s)"
            )
            for row in sim.get("headline") or []:
                self.stdout.write(
                    f"  {row['policy']:<16} cum_reward ${row['cum_reward']:>14,.0f}  regret ${row['cum_regret']:>14,.0f}"
                )
            if not opts["no_drill"]:
                # a short run whose breaker trips at the 2016-06 shift and whose review is left pending
                # (auto-acknowledgement due after the end), so the review queue has something to act on
                drill = self._call(
                    "post",
                    "/simulations/",
                    {
                        "fit": fit["id"],
                        "name": "Governance drill: breaker trips, review left pending",
                        "policies": ["thompson", "historical"],
                        "start": "2016-01",
                        "end": "2017-06",
                        "max_loans": 1500,
                        "seed": opts["seed"],
                        "breaker_mode": "auto_ack",
                        "checkpoint_every": 3,
                        "psi": {"auto_ack_epochs": 60, "auto_action": "resegment"},
                    },
                )
                self.stdout.write(
                    f"simulation #{drill['id']}: {drill['status']} — breaker {drill['breaker_state']}, "
                    f"{drill['pending_reviews']} pending review(s) ({time.perf_counter() - t0:.1f}s)"
                )
        finally:
            settings.JOBS_SYNC, settings.ALLOWED_HOSTS = previous
        self.stdout.write(
            self.style.SUCCESS(f"demo ready in {time.perf_counter() - t0:.1f}s — open /api/docs/ or the console")
        )
