#!/usr/bin/env bash
set -euo pipefail

# Usage: ./pack_overleaf.sh [output.zip]
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive="${1:-$repo_dir/paper-overleaf.zip}"
if [[ "$archive" != /* ]]; then
    archive="$PWD/$archive"
fi

if ! command -v zip >/dev/null 2>&1; then
    printf 'Error: install zip to package the paper.\n' >&2
    exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf -- "$temp_dir"' EXIT

cd -- "$repo_dir/paper"
zip -qr "$temp_dir/paper-overleaf.zip" \
    ./*.tex ./*.bib sections tables plots style \
    -x '*.csv' '*.aux' '*.bbl' '*.blg' '*.log' '*.out' '*.toc' \
    '*.fls' '*.fdb_latexmk' '*.synctex.gz'

mv -f -- "$temp_dir/paper-overleaf.zip" "$archive"
printf 'Created: %s\n' "$archive"
