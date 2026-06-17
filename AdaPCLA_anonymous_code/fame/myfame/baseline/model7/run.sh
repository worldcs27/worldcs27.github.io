#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper: many folders use run.sh as entrypoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/bash.sh" "${@:-all}"
