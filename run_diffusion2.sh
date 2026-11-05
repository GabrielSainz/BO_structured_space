#!/bin/bash
# Compatibility entry point; invoke from the repository root as before.
exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/experiments/run_diffusion2.sh" "$@"
