#!/bin/bash
set -euo pipefail

# Vast SSH/Jupyter launch modes replace the image CMD. Put this command in the
# template's On-start Script to launch the same validated entrypoint.
exec python3 /opt/joyai/vast/start.py
