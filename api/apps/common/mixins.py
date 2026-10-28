"""Model mixins."""

from __future__ import annotations

from .models import Job

#: Dataset / fit lifecycle (types.ts ``BuildStatus``).
BUILD_STATUSES = ("queued", "running", "ready", "failed", "cancelled")


class BuildStatusMixin:
    """``current_status`` = stored status, corrected by the job's fate (a worker that died
    leaves the stored status at queued/running while the job is failed)."""

    status: str
    job: Job | None

    @property
    def current_status(self) -> str:
        if self.status in ("queued", "running") and self.job is not None:
            if self.job.status == Job.Status.FAILED:
                return "failed"
            if self.job.status == Job.Status.CANCELLED:
                return "cancelled"
            if self.job.status == Job.Status.RUNNING and self.status == "queued":
                return "running"
        return self.status

    @property
    def is_ready(self) -> bool:
        return self.current_status == "ready"
