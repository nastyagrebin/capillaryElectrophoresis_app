# CEtools Electropherogram Pipeline
**Current Version:** `v2.0.0`

Interactive Bokeh/Panel app for upload → preprocess → align → NMF → alpha diversity → visualize.

## Updating the App (For Git Users)
If you downloaded the app by cloning the repository with Git, you can update to the newest version instantly without re-downloading anything.
1. Open a terminal inside the `capillaryElectrophoresis_app` folder.
2. Run the following command to download the latest updates:
   ```bash
   git pull origin main
   ```
3. Run the app as normal!

## Quick start (fastest path)

### Option A: Mamba/Conda (recommended)
1. Install **Miniforge** from conda-forge. Run the installer and restart your terminal so conda is available. f you already have ```conda``` nstalled, you can skp ths step.
2. Download all files in this repo, keeping the folder structure the same.
3. Create the environment. Open a terminal at the root of this repo and run:
   ```bash
   conda env create -f environment.yml
   conda activate cetools-app

4. Install the local CEtools package (from this repo):
   ```bash
   pip install -e .
   ```
5. Launch the app:
   ```bash
   python app/app.py

### Option B: pip + venv (works if you have a working compiler toolchain)

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
2. Install packages:
   ```bash
   pip install -r requirements.txt
   pip install -e CEtools
3. Run:
   ```bash
   panel serve app/app.py --show

### Self-check (30 seconds)
   ```bash
   python scripts/verify_setup.py
   ```
You should see ✅ lines and version printouts. If Panel/Bokeh versions don’t match, reinstall the environment.

## Troubleshooting

### Command not found: panel
Activate the env: ```mamba activate cetools-app``` (or ```conda activate cetools-app```).

### Browser doesn’t open
Add ```--autoreload``` or manually open the printed URL (usually http://localhost:5006).

### Version mismatch
Recreate the env:
   ```bash
   mamba env remove -n cetools-app
   mamba env create -f environment.yml
   ```

### Excel files won't load
Ensure ```openpyxl``` is installed.

### Tested versions

Python 3.10

Bokeh 3.1.1

Panel 1.0.2

NumPy/Pandas/SciPy/scikit-learn: conda-forge latest as of this file’s date

If you need to deviate from these, do it in a new branch and update ```environment.yml``` after testing.
