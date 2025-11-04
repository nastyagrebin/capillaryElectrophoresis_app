@echo off
call conda activate cetools-app
panel serve app/app.py --show --autoreload
