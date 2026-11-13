"""Temporal orchestration of background jobs.

* :mod:`.workflows` — the deterministic workflow definitions (no Django imports: they run in
  Temporal's workflow sandbox).
* :mod:`.activities` — the activities a ``manage.py temporal_worker`` process executes: the job
  supervisor (spawns and heartbeats ``manage.py run_job <id>``), lost-worker bookkeeping and
  artifact garbage collection.
* :mod:`.client` — the API side: a process-wide client on a background event loop, so sync
  Django views can start workflows, describe them and send them updates.
"""
