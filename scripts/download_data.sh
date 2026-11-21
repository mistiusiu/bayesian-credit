#!/usr/bin/env bash
# Download the Lending Club loan export used by bayesian_credit (SPEC §1.3) into data/raw/
# (or $LC_DATA_DIR):
#   default   Kaggle dataset  ethon0426/lending-club-20072020q1
#             file            Loan_status_2007-2020Q3.csv  (~1.8 GB plain CSV, 2,925,493 rows x 142
#                             columns, issued 2007-06..2020-09, observed 2020-10). Kaggle names it
#                             Loan_status_2007-2020Q3.gzip although it is not compressed.
#   --legacy  Kaggle dataset  wordsforthewise/lending-club  (CC0)
#             file            accepted_2007_to_2018Q4.csv.gz  (~374 MB, 2,260,701 rows x 151 columns)
#
# Credentials (checked in this order, the token is never printed):
#   1. $KAGGLE_API_TOKEN
#   2. ~/.kaggle/access_token        (new-style Kaggle access token; chmod 600)
#   3. ~/.kaggle/kaggle.json or $KAGGLE_USERNAME/$KAGGLE_KEY (legacy API key; needs the kaggle CLI)
# Without the kaggle CLI an access token is used with curl against the Kaggle API.
#
# Usage: scripts/download_data.sh [--legacy] [--force] [--dry-run]
set -euo pipefail

DATASET="ethon0426/lending-club-20072020q1"
REMOTE_FILE="Loan_status_2007-2020Q3.gzip"   # Kaggle's name; the content is a plain CSV
FILE="Loan_status_2007-2020Q3.csv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${LC_DATA_DIR:-$REPO_ROOT/data/raw}"
FORCE=0
DRY_RUN=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    --legacy) DATASET="wordsforthewise/lending-club"; REMOTE_FILE="accepted_2007_to_2018Q4.csv.gz"; FILE="$REMOTE_FILE" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$DEST"
TARGET="$DEST/$FILE"
if [[ -s "$TARGET" && $FORCE -eq 0 ]]; then
  echo "$TARGET already present ($(du -h "$TARGET" | cut -f1)); pass --force to download again."
  exit 0
fi

# kaggle CLI: PATH first, then the repository virtualenv (optional with an access token)
KAGGLE="$(command -v kaggle || true)"
if [[ -z "$KAGGLE" && -x "$REPO_ROOT/.venv/bin/kaggle" ]]; then
  KAGGLE="$REPO_ROOT/.venv/bin/kaggle"
fi

# credentials (never echoed)
if [[ -z "${KAGGLE_API_TOKEN:-}" && -s "$HOME/.kaggle/access_token" ]]; then
  KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "$HOME/.kaggle/access_token")"
  export KAGGLE_API_TOKEN
  CRED_SOURCE="~/.kaggle/access_token"
elif [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  CRED_SOURCE="\$KAGGLE_API_TOKEN"
elif [[ -s "$HOME/.kaggle/kaggle.json" || ( -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ) ]]; then
  CRED_SOURCE="legacy kaggle.json / KAGGLE_KEY"
  if [[ -z "$KAGGLE" ]]; then
    echo "a legacy Kaggle API key needs the kaggle CLI - install it with: pip install kaggle" >&2
    exit 1
  fi
else
  echo "No Kaggle credentials found. Create a token at https://www.kaggle.com/settings and save it" >&2
  echo "to ~/.kaggle/access_token (chmod 600), or export KAGGLE_API_TOKEN." >&2
  exit 1
fi

echo "dataset : $DATASET"
echo "file    : $REMOTE_FILE -> $FILE"
echo "into    : $DEST"
echo "auth    : $CRED_SOURCE"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] would download $REMOTE_FILE with ${KAGGLE:-curl} into a temporary directory"
  exit 0
fi

# download into a fresh, private directory; move into place only when complete
TMP="$(mktemp -d "$DEST/.download.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
if [[ -n "$KAGGLE" ]]; then
  "$KAGGLE" datasets download "$DATASET" -f "$REMOTE_FILE" -p "$TMP" --force
else
  curl -fSL --retry 5 -o "$TMP/$REMOTE_FILE.zip" -H "Authorization: Bearer $KAGGLE_API_TOKEN" \
    "https://www.kaggle.com/api/v1/datasets/download/$DATASET/$REMOTE_FILE"
fi

if [[ -f "$TMP/$REMOTE_FILE.zip" ]] || unzip -tq "$TMP/$REMOTE_FILE" >/dev/null 2>&1; then
  # Kaggle wraps single files in a zip (sometimes under the file's own name)
  ZIP="$TMP/$REMOTE_FILE.zip"
  [[ -f "$ZIP" ]] || { mv "$TMP/$REMOTE_FILE" "$ZIP"; }
  mkdir -p "$TMP/unzipped"
  unzip -q -o "$ZIP" -d "$TMP/unzipped"
  SRC="$(find "$TMP/unzipped" -type f -name "$REMOTE_FILE" -print -quit)"
elif [[ -f "$TMP/$REMOTE_FILE" ]]; then
  SRC="$TMP/$REMOTE_FILE"
else
  SRC=""
fi
if [[ -z "$SRC" || ! -s "$SRC" ]]; then
  echo "download finished but $REMOTE_FILE was not found in the archive:" >&2
  ls -la "$TMP" >&2
  exit 1
fi
if [[ "$FILE" == *.gz ]]; then
  if ! gzip -t "$SRC" 2>/dev/null; then
    echo "downloaded file is not a valid gzip archive" >&2
    exit 1
  fi
elif ! head -c 4096 "$SRC" | grep -q "loan_status"; then
  echo "downloaded file does not look like the Lending Club CSV (no loan_status header)" >&2
  exit 1
fi
mv -f "$SRC" "$TARGET"
echo "saved $TARGET ($(du -h "$TARGET" | cut -f1))"
echo "optional: build the canonical parquet cache once (~10-20 s):"
echo "  python -c 'from bayesian_credit.data import LendingClubLoader; print(LendingClubLoader().build_cache())'"
