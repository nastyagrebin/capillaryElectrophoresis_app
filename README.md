# CEtools Electropherogram Pipeline
**Current Version:** `v2.0.0`

Interactive Bokeh/Panel app for upload → preprocess → align → NMF → alpha diversity → visualize.

## Installation

### Option A: Git Clone + Conda (Recommended for Easy Updates)
This option uses the terminal to clone the repository. It is the best choice because you can instantly download future app updates with a single command.

1. **Install Miniforge** from conda-forge (if you don't already have conda installed). Run the installer and restart your terminal.
2. **Clone the repository** by opening a terminal and running:
   ```bash
   git clone https://github.com/nastyagrebin/capillaryElectrophoresis_app.git
   cd capillaryElectrophoresis_app
   ```
3. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate cetools-app
   ```
4. **Install the local CEtools package:**
   ```bash
   pip install -e .
   ```
5. **Launch the app:**
   ```bash
   python app/app.py
   ```

### Option B: ZIP Download + Conda (Simplest for Beginners)
This option is best if you are uncomfortable with Git and prefer downloading a static folder.

1. **Install Miniforge** from conda-forge (if you don't already have conda installed). Run the installer and restart your terminal.
2. **Download the app files** by clicking the green "Code" button at the top of this GitHub page and selecting "Download ZIP". Extract the folder to your computer.
3. Open a terminal, navigate inside the extracted folder, and **create the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate cetools-app
   ```
4. **Install the local CEtools package:**
   ```bash
   pip install -e .
   ```
5. **Launch the app:**
   ```bash
   python app/app.py
   ```

### Option C: pip + venv (Requires a Working Compiler)
This option works best if you do not want to use conda and have a working C compiler toolchain installed.

1. Download the app (via Git clone or ZIP). Open a terminal inside the app folder.
2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
3. **Install packages:**
   ```bash
   pip install -r requirements.txt
   pip install -e CEtools
   ```
4. **Run the app:**
   ```bash
   panel serve app/app.py --show
   ```

---

## Updating

If you installed the app using **Option A (Git Clone)**, you can update to the newest version instantly without re-downloading anything!

1. Open a terminal inside the `capillaryElectrophoresis_app` folder.
2. Run the following command to download the latest updates:
   ```bash
   git pull origin main
   ```
3. Run the app as normal!

*(Note: If you used Option B, you must download the new ZIP file and replace your old folder).*

---

## Troubleshooting

### Self-check (30 seconds)
You can run a quick self-check to ensure your packages are correct:
```bash
python scripts/verify_setup.py
```
You should see ✅ lines and version printouts. If Panel/Bokeh versions don’t match, reinstall the environment.

### Command not found: panel
Activate the env: `mamba activate cetools-app` (or `conda activate cetools-app`).

### Browser doesn’t open
Add `--autoreload` to your launch command or manually open the printed URL (usually http://localhost:5006).

### Version mismatch
Recreate the environment from scratch:
```bash
mamba env remove -n cetools-app
mamba env create -f environment.yml
```

### Excel files won't load
Ensure `openpyxl` is installed.

### Tested versions
- **Python:** 3.10
- **Bokeh:** 3.1.1
- **Panel:** 1.0.2
- **NumPy/Pandas/SciPy/scikit-learn:** conda-forge latest as of this file’s date

If you need to deviate from these, do it in a new branch and update `environment.yml` after testing.
