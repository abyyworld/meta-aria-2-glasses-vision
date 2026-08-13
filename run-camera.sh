#!/usr/bin/env bash
# Live camera preview — the quickest way to see the detector working.
#
# Run this from Terminal.app (not from an editor), the first time at least:
# macOS will prompt for camera access, and the prompt is attached to whichever
# app owns the process.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m aria_devices.cli camera "$@"
