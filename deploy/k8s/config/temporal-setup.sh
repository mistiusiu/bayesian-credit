#!/bin/sh
# Prepare Temporal on PostgreSQL. Idempotent; runs in the temporalio/admin-tools image.
#   temporal-setup.sh schema      create the persistence + visibility databases and apply every
#                                 schema version (before the server starts / on upgrades)
#   temporal-setup.sh namespace   wait for the server, then create TEMPORAL_NAMESPACE if missing
set -eu
: "${POSTGRES_SEEDS:=postgres}" "${DB_PORT:=5432}" "${POSTGRES_USER:=temporal}" "${POSTGRES_PWD:=temporal}"
: "${DBNAME:=temporal}" "${VISIBILITY_DBNAME:=temporal_visibility}"
: "${TEMPORAL_ADDRESS:=temporal:7233}" "${TEMPORAL_NAMESPACE:=default}" "${TEMPORAL_RETENTION:=30d}"
SCHEMA=/etc/temporal/schema/postgresql/v12

sql() {
  temporal-sql-tool --plugin postgres12 --ep "$POSTGRES_SEEDS" -p "$DB_PORT" \
    -u "$POSTGRES_USER" --pw "$POSTGRES_PWD" "$@"
}

schema() {
  for pair in "$DBNAME:temporal" "$VISIBILITY_DBNAME:visibility"; do
    db=${pair%%:*}; dir=$SCHEMA/${pair##*:}/versioned
    tries=0
    until out=$(sql --db "$db" create-database 2>&1) || echo "$out" | grep -qi "already exists"; do
      tries=$((tries + 1))
      [ "$tries" -ge 60 ] && { echo "$out"; exit 1; }
      echo "waiting for PostgreSQL at $POSTGRES_SEEDS:$DB_PORT ($tries)"; sleep 2
    done
    sql --db "$db" setup-schema -v 0.0
    sql --db "$db" update-schema -d "$dir"
    echo "database $db: schema up to date"
  done
}

namespace() {
  tries=0
  until temporal operator cluster health --address "$TEMPORAL_ADDRESS" >/dev/null 2>&1; do
    tries=$((tries + 1))
    [ "$tries" -ge 90 ] && { echo "Temporal at $TEMPORAL_ADDRESS never became healthy"; exit 1; }
    sleep 2
  done
  if temporal operator namespace describe -n "$TEMPORAL_NAMESPACE" --address "$TEMPORAL_ADDRESS" >/dev/null 2>&1; then
    echo "namespace $TEMPORAL_NAMESPACE exists"
  else
    temporal operator namespace create -n "$TEMPORAL_NAMESPACE" --retention "$TEMPORAL_RETENTION" \
      --address "$TEMPORAL_ADDRESS"
  fi
}

case "${1:-}" in
  schema) schema ;;
  namespace) namespace ;;
  *) echo "usage: $0 schema|namespace"; exit 2 ;;
esac
