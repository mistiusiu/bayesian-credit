#!/usr/bin/env bash
# Download the Lending Club accepted-loans export used by bayesian_credit (SPEC §1.3):
#   Kaggle dataset  wordsforthewise/lending-club  (CC0)
#   file            accepted_2007_to_2018Q4.csv.gz  (~374 MB, 2,260,701 rows x 151 columns)
# into data/raw/ (or $LC_DATA_DIR).
#
# Credentials (checked in this order, the token is never printed):
#   1. $KAGGLE_API_TOKEN
#   2. ~/.kaggle/access_token        (new-style Kaggle access token; chmod 600)
#   3. ~/.kaggle/kaggle.json or $KAGGLE_USERNAME/$KAGGLE_KEY (legacy API key)
#
# Usage: scripts/download_data.sh [--force] [--dry-run]
set -euo pipefail

DATASET="wordsforthewise/lending-club"
FILE="accepted_2007_to_2018Q4.csv.gz"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${LC_DATA_DIR:-$REPO_ROOT/data/raw}"
FORCE=0
DRY_RUN=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
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

# kaggle CLI: PATH first, then the repository virtualenv
KAGGLE="$(command -v kaggle || true)"
if [[ -z "$KAGGLE" && -x "$REPO_ROOT/.venv/bin/kaggle" ]]; then
  KAGGLE="$REPO_ROOT/.venv/bin/kaggle"
fi
if [[ -z "$KAGGLE" ]]; then
  echo "kaggle CLI not found - install it with: pip install kaggle" >&2
  exit 1
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
else
  echo "No Kaggle credentials found. Create a token at https://www.kaggle.com/settings and save it" >&2
  echo "to ~/.kaggle/access_token (chmod 600), or export KAGGLE_API_TOKEN." >&2
  exit 1
fi

echo "dataset : $DATASET"
echo "file    : $FILE"
echo "into    : $DEST"
echo "auth    : $CRED_SOURCE"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] would run: $KAGGLE datasets download $DATASET -f $FILE -p <tmpdir> --force"
  exit 0
fi

# download into a fresh, private directory; move into place only when complete
TMP="$(mktemp -d "$DEST/.download.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
"$KAGGLE" datasets download "$DATASET" -f "$FILE" -p "$TMP" --force

if [[ -f "$TMP/$FILE" ]]; then
  SRC="$TMP/$FILE"
elif [[ -f "$TMP/$FILE.zip" ]]; then   # Kaggle sometimes wraps single files in a zip
  mkdir -p "$TMP/unzipped"
  unzip -q -o "$TMP/$FILE.zip" -d "$TMP/unzipped"
  SRC="$(find "$TMP/unzipped" -type f -name "$FILE" -print -quit)"
else
  SRC=""
fi
if [[ -z "$SRC" || ! -s "$SRC" ]]; then
  echo "download finished but $FILE was not found in the archive:" >&2
  ls -la "$TMP" >&2
  exit 1
fi
if ! gzip -t "$SRC" 2>/dev/null; then
  echo "downloaded file is not a valid gzip archive" >&2
  exit 1
fi
mv -f "$SRC" "$TARGET"
echo "saved $TARGET ($(du -h "$TARGET" | cut -f1))"
echo "optional: build the canonical parquet cache once (~15 s):"
echo "  python -c 'from bayesian_credit.data import LendingClubLoader; print(LendingClubLoader().build_cache())'"
