#!/bin/sh
# serve      -> migrate + gunicorn API (starts jobs as Temporal workflows; runs none itself)
# worker     -> Temporal worker that executes the jobs:  entrypoint.sh worker [--concurrency N]
# run-job    -> run one job in the foreground (debugging): entrypoint.sh run-job <job_id>
# simulate   -> run a run-dir outside the API:  entrypoint.sh simulate <run_dir>
# resume     -> resume a run-dir outside the API: entrypoint.sh resume <run_dir>
# demo       -> seed a synthetic dataset, fit, small simulation (needs a running worker)
set -e
case "$1" in
  serve)
    python manage.py migrate --noinput
    python manage.py collectstatic --noinput >/dev/null 2>&1 || true
    exec gunicorn config.wsgi:application \
      --bind 0.0.0.0:${PORT:-8000} \
      --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} \
      --timeout ${GUNICORN_TIMEOUT:-600} --access-logfile - --error-logfile -
    ;;
  worker)   shift; exec python manage.py temporal_worker "$@" ;;
  run-job)  shift; exec python manage.py run_job "$@" ;;
  simulate) shift; exec python -m bayesian_credit.simulation.cli run --run-dir "$@" ;;
  resume)   shift; exec python -m bayesian_credit.simulation.cli resume --run-dir "$@" ;;
  demo)     python manage.py migrate --noinput && exec python manage.py demo ;;
  *)        exec "$@" ;;
esac
