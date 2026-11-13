#!/bin/sh
# First-boot initialisation of the shared PostgreSQL instance (docker-entrypoint-initdb.d):
#   role bc       + database bc        the Django API (CREATEDB so the test runner can make test_bc)
#   role temporal (CREATEDB)           owns temporal + temporal_visibility, created by temporal-setup.sh
# Passwords come from BC_DB_PASSWORD / TEMPORAL_DB_PASSWORD (defaults are for local development).
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<SQL
CREATE ROLE bc LOGIN CREATEDB PASSWORD '${BC_DB_PASSWORD:-bc}';
CREATE DATABASE bc OWNER bc;
CREATE ROLE temporal LOGIN CREATEDB PASSWORD '${TEMPORAL_DB_PASSWORD:-temporal}';
SQL
