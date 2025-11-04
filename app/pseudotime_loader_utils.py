# ============================
# FILE: app/pseudotime_loader_utils.py
# ============================
from __future__ import annotations
from typing import Dict, Optional, Tuple
import io

import numpy as np
import pandas as pd
import panel as pn

OK = "OK:"; WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

pn.extension('tabulator')

def _parse_pseudotimes_wide_csv(name: str, data: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Parse a 'pseudotimes_wide.csv' created by the Alignment tab exporter.

    Expected columns:
      - 'time'  (original time; optional but preferred)
      - for each sample S:
          S+'_pt'  (pseudotime values)
          S        (intensity values)

    Returns:
      pseudotimes_df: DataFrame with columns=samples, values=pseudotimes (float)
      intensities_df: DataFrame with columns=samples, values=intensities (float)
      input_by_sample: dict[sample] -> DataFrame(time, intensity)
    """
    # Read CSV
    name = (name or "").lower()
    if not name.endswith(".csv"):
        raise ValueError("Please provide a CSV file exported as 'pseudotimes_wide.csv'.")
    df = pd.read_csv(io.BytesIO(data))
    if df.empty:
        raise ValueError("CSV is empty.")
    cols = list(df.columns)
    # Discover samples
    pt_cols = [c for c in cols if c.endswith("_pt")]
    if not pt_cols:
        raise ValueError("No '*_pt' columns found; is this a pseudotimes_wide CSV?")
    samples = [c[:-3] for c in pt_cols]  # strip '_pt'
    # Validate intensity columns exist
    missing = [s for s in samples if s not in df.columns]
    if missing:
        raise ValueError(f"Missing intensity columns for: {', '.join(missing)}")
    # Optional time
    has_time = "time" in df.columns
    # Build matrices (align on current row order)
    pseudo = pd.DataFrame({s: df[f"{s}_pt"].astype(float).values for s in samples})
    inten  = pd.DataFrame({s: df[s].astype(float).values for s in samples})
    # Build per-sample DataFrames expected by downstream bits (time,intensity)
    input_by_sample: Dict[str, pd.DataFrame] = {}
    for s in samples:
        if has_time:
            input_by_sample[s] = pd.DataFrame({"time": df["time"].astype(float).values,
                                               "intensity": df[s].astype(float).values})
        else:
            # still provide a monotonically increasing 'time' index if not present
            input_by_sample[s] = pd.DataFrame({"time": np.arange(len(df), dtype=float),
                                               "intensity": df[s].astype(float).values})
    # Keep original index if present
    pseudo.index = df.index
    inten.index  = df.index
    return pseudo, inten, input_by_sample


class PseudotimeLoaderController:
    """
    Lets the user bypass Upload+Preprocess+Alignment by loading a prior 'pseudotimes_wide.csv'.
    Exposes:
      - pseudotimes_df (DataFrame)
      - intensities_df (DataFrame)
      - input_by_sample (dict[sample] -> DataFrame(time,intensity))
      - on_loaded: optional callback receiving (pseudotimes_df, intensities_df, input_by_sample)
    """
    def __init__(self):
        self.pseudotimes_df: Optional[pd.DataFrame] = None
        self.intensities_df: Optional[pd.DataFrame] = None
        self.input_by_sample: Dict[str, pd.DataFrame] = {}
        self.on_loaded = None  # type: Optional[callable]

        self.file = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.load_btn = pn.widgets.Button(name="Load aligned pseudotimes (.csv)", button_type="primary")
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Quick peek widgets
        self.samples_table = pn.widgets.Tabulator(pd.DataFrame(), height=260, selectable=False, show_index=False)
        self.preview_help = pn.pane.Markdown(
            "_Tip: the table shows the first few rows of pseudotime & intensity for the first 5 samples._",
            styles={"color": "#666"}
        )

        self.section = pn.Column(
            pn.pane.Markdown("## Load Previously Aligned Data"),
            pn.pane.Markdown(
                "If you already exported a **pseudotimes_wide.csv** from the Alignment tab, load it here to skip "
                "Upload → Preprocess → Alignment and continue with **NMF**, **Visualization**, and **Diversity**.",
                styles={"color": "#555"}
            ),
            pn.Row(self.file, pn.Spacer(width=12), self.load_btn),
            self.status,
            pn.layout.Divider(),
            self.preview_help,
            self.samples_table,
            sizing_mode="stretch_width",
            visible=True,   # keep visible; user opts in explicitly
        )

        self.load_btn.on_click(self._on_load)

    def _on_load(self, _=None):
        if not self.file.value:
            self.status.object = warn("Choose a CSV file first.")
            return
        try:
            pseudo, inten, by_s = _parse_pseudotimes_wide_csv(self.file.filename or "", bytes(self.file.value))
        except Exception as e:
            self.status.object = warn(f"Failed to parse CSV: {e}")
            self.samples_table.value = pd.DataFrame()
            self.pseudotimes_df = None; self.intensities_df = None; self.input_by_sample = {}
            return

        self.pseudotimes_df = pseudo
        self.intensities_df = inten
        self.input_by_sample = by_s

        # Small preview: head of the first few samples (interleaving *_pt and intensity)
        try:
            first = list(inten.columns)[:5]
            cols = []
            for s in first:
                cols += [f"{s}_pt", s]
            # construct a small preview from the underlying CSV-like values
            prev = pd.DataFrame()
            for s in first:
                prev[f"{s}_pt"] = pseudo[s].values[:10]
                prev[s] = inten[s].values[:10]
            self.samples_table.value = prev
        except Exception:
            self.samples_table.value = pd.DataFrame()

        self.status.object = ok(f"Loaded {inten.shape[1]} sample(s), {inten.shape[0]} points each. You can proceed to NMF.")
        # Notify app
        if callable(self.on_loaded):
            try:
                self.on_loaded(self.pseudotimes_df, self.intensities_df, self.input_by_sample)
            except Exception:
                pass


def build_pseudotime_loader_section():
    ctrl = PseudotimeLoaderController()
    return ctrl.section, ctrl
