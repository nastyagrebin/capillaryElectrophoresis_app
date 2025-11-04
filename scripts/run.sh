#!/usr/bin/env bash
set -e
conda activate cetools-app
panel serve app/app.py --show --autoreload
