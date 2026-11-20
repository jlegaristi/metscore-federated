#!/usr/bin/env bash
# Launch the experiment grid detached, so it survives closing the terminal.
#
#   ./scripts/launch.sh                  full grid, 10 repetitions
#   ./scripts/launch.sh --repetitions 3  fast first pass
#   ./scripts/launch.sh --smoke          verify everything runs, ~4 min
#
# Then, from anywhere:
#   tail -f <the log it prints>                        watch progress
#   python scripts/run_experiments.py --status         what has finished
#   kill <the pid it prints>                           stop cleanly
#
# Stopping is safe. Finished runs are kept and skipped on the next launch, so
# an interrupted grid resumes by relaunching with the same arguments.

set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${WORKSPACE}/.venv/bin/python}"
DATA="${METSCORE_DATA_DIR:-${WORKSPACE}/data}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Interpreter not found at ${PYTHON}; set PYTHON to the one to use." >&2
    exit 1
fi

cd "${WORKSPACE}"

if [[ ! -f "${DATA}/discovery.csv" || ! -f "${DATA}/external.csv" ]]; then
    echo "Building the data snapshot first..."
    "${PYTHON}" scripts/prepare_data.py
    echo
fi

mkdir -p results
LOG="results/grid_$(date +%Y%m%d_%H%M%S).log"

# setsid detaches from the terminal session, so a dropped SSH connection does
# not take the grid down with it.
setsid nohup "${PYTHON}" scripts/run_experiments.py "$@" > "${LOG}" 2>&1 &
PID=$!

sleep 2
echo "Grid running.  pid ${PID}"
echo "log            ${WORKSPACE}/${LOG}"
echo
echo "  tail -f ${LOG}"
echo "  ${PYTHON} scripts/run_experiments.py --status"
echo "  kill ${PID}"
echo
head -1 "${LOG}" 2>/dev/null || true
