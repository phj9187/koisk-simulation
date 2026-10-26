#!/bin/sh
set -eu
cd /workspace/koisk-simulation
python3 -m venv /workspace/kiosk-venv
/workspace/kiosk-venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
mkdir -p /tmp/kiosk-cache/fontconfig
export XDG_CACHE_HOME=/tmp/kiosk-cache
export MPLCONFIGDIR=/tmp/kiosk-mpl
export PYTHONPATH=/workspace/koisk-simulation/src
/workspace/kiosk-venv/bin/python -m pip check
/workspace/kiosk-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
