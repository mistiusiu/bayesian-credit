"""``python manage.py gc_artifacts [--apply]`` — remove artifact directories no row refers to."""

from __future__ import annotations

import shutil

from django.core.management.base import BaseCommand

from apps.common import artifacts
from apps.common.models import Job
from apps.datasets.models import Dataset
from apps.fits.models import Fit
from apps.simulations.models import Simulation


class Command(BaseCommand):
    help = "List (or with --apply delete) orphaned dataset / fit / simulation directories and job logs."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")

    def handle(self, *args, **opts):
        root = artifacts.root()
        known = {
            "datasets": set(Dataset.objects.values_list("pk", flat=True)),
            "fits": set(Fit.objects.values_list("pk", flat=True)),
            "simulations": set(Simulation.objects.values_list("pk", flat=True)),
        }
        orphans = []
        for kind, ids in known.items():
            base = root / kind
            if not base.is_dir():
                continue
            for d in base.iterdir():
                if d.is_dir() and (not d.name.isdigit() or int(d.name) not in ids):
                    orphans.append(d)
        jobs = set(Job.objects.values_list("pk", flat=True))
        logs = root / "jobs"
        if logs.is_dir():
            orphans += [p for p in logs.glob("*.log") if not p.stem.isdigit() or int(p.stem) not in jobs]
        for path in orphans:
            self.stdout.write(("deleting " if opts["apply"] else "orphan ") + str(path))
            if opts["apply"]:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        self.stdout.write(f"{len(orphans)} orphan(s){'' if opts['apply'] else ' (dry run; pass --apply to delete)'}")
