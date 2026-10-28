#!/bin/sh
# serve      -> migrate + gunicorn API (also spawns simulation workers as subprocesses)
# worker     -> run one job:        entrypoint.sh worker <job_id>
# simulate   -> run a run-dir:      entrypoint.sh simulate <run_dir>   (used by the k8s Job template)
# resume     -> resume a run-dir:   entrypoint.sh resume <run_dir>
# demo       -> seed a synthetic dataset, fit, small simulation
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
  worker)   shift; exec python manage.py run_job "$@" ;;
  simulate) shift; exec python -m bayesian_credit.simulation.cli run --run-dir "$@" ;;
  resume)   shift; exec python -m bayesian_credit.simulation.cli resume --run-dir "$@" ;;
  demo)     python manage.py migrate --noinput && exec python manage.py demo ;;
  *)        exec "$@" ;;
esac
