"""``python manage.py gc_artifacts [--apply]`` — remove artifact directories no row refers to.

The ``ArtifactGC`` Temporal schedule (``TEMPORAL_GC_SCHEDULE``) runs the same clean-up daily.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.common import artifacts


class Command(BaseCommand):
    help = "List (or with --apply delete) orphaned dataset / fit / simulation directories and job logs."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")

    def handle(self, *args, **opts):
        orphans = artifacts.find_orphans()
        for path in orphans:
            self.stdout.write(("deleting " if opts["apply"] else "orphan ") + str(path))
            if opts["apply"]:
                artifacts.remove(path)
        self.stdout.write(f"{len(orphans)} orphan(s){'' if opts['apply'] else ' (dry run; pass --apply to delete)'}")
