#!/usr/bin/env bash
# Symlink the released kit data into the repo (git-ignored) and into kit/ so the kit scripts find ./public.
# Usage: bash claude_output/sh/01_link_data.sh [main_public_dir] [adaptation_public_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
MAIN_PUBLIC="${1:-/home/ganfwe/Downloads/main-participants-60-40/public}"
mkdir -p data
[ -f "$MAIN_PUBLIC/metadata.json" ] || { echo "no metadata.json in $MAIN_PUBLIC"; exit 1; }
ln -sfn "$MAIN_PUBLIC" data/main
ln -sfn "$(pwd)/data/main" kit/public
if [ -n "${2:-}" ]; then ln -sfn "$2" data/adaptation; fi
grep -o '"round": "[a-z]*"' data/main/metadata.json
ls data/main kit/public >/dev/null && echo "data linked: data/main -> $MAIN_PUBLIC"
