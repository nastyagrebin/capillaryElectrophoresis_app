import io
import os
import re
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import panel as pn
import bokeh
import bokeh.plotting as bkp
import bokeh.models
from bokeh.embed import components
from bokeh.resources import CDN, INLINE

from common_plot import TitledPlotPane, apply_export_prefix_to_pane

def _ok(msg: str) -> str:
    return f"<div style='color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 8px 12px; border-radius: 4px; margin: 6px 0;'>✓ {msg}</div>"

def _warn(msg: str) -> str:
    return f"<div style='color: #856404; background-color: #fff3cd; border-color: #ffeeba; padding: 8px 12px; border-radius: 4px; margin: 6px 0;'>⚠️ {msg}</div>"

def extract_bokeh_figure(obj: Any) -> Optional[bokeh.models.LayoutDOM]:
    """Recursively extract a Bokeh figure or LayoutDOM from obj or Panel panes."""
    if obj is None:
        return None
    if isinstance(obj, bokeh.models.LayoutDOM):
        return obj
    if hasattr(obj, "bokeh_pane") and getattr(obj.bokeh_pane, "object", None) is not None:
        return extract_bokeh_figure(obj.bokeh_pane.object)
    if hasattr(obj, "object") and getattr(obj, "object", None) is not None:
        return extract_bokeh_figure(obj.object)
    if hasattr(obj, "objects") and getattr(obj, "objects", None):
        for child in obj.objects:
            fig = extract_bokeh_figure(child)
            if fig is not None:
                return fig
    return None

def find_all_bokeh_figures(obj: Any, max_depth: int = 5) -> List[bokeh.models.LayoutDOM]:
    """Recursively find all Bokeh LayoutDOM instances in a container or pane."""
    figures: List[bokeh.models.LayoutDOM] = []
    if obj is None or max_depth <= 0:
        return figures
    if isinstance(obj, bokeh.models.LayoutDOM):
        figures.append(obj)
        return figures
    if hasattr(obj, "bokeh_pane") and getattr(obj.bokeh_pane, "object", None) is not None:
        figures.extend(find_all_bokeh_figures(obj.bokeh_pane.object, max_depth - 1))
    elif hasattr(obj, "object") and getattr(obj, "object", None) is not None:
        figures.extend(find_all_bokeh_figures(obj.object, max_depth - 1))
    elif hasattr(obj, "objects") and getattr(obj, "objects", None):
        for child in obj.objects:
            figures.extend(find_all_bokeh_figures(child, max_depth - 1))
    return figures

class HTMLReportController:
    """Controller for the HTML Report Export tab."""

    def __init__(self, controllers: Dict[str, Any]):
        self.controllers = controllers
        self._export_prefix = "CE_analysis"
        self._current_html_str: str = ""
        
        # User-added custom items
        self.additional_pipeline_plots: List[Dict[str, Any]] = [] # list of dicts: {id, title, caption, fig, type}
        self.custom_images: List[Dict[str, Any]] = [] # list of dicts: {id, title, caption, data_uri, filename}

        # ---------------- 1. Report Metadata & Config ----------------
        self.report_title_input = pn.widgets.TextInput(
            name="Report Title",
            value=f"{self._export_prefix} Electropherogram Analysis Report",
            sizing_mode="stretch_width"
        )
        self.report_subtitle_input = pn.widgets.TextInput(
            name="Subtitle / Project Description",
            value="",
            placeholder="e.g. Capillary Electrophoresis Quality Control & NMF Decomposition",
            sizing_mode="stretch_width"
        )
        self.author_input = pn.widgets.TextInput(
            name="Author / Analyst Name",
            value="",
            placeholder="e.g. Lab Researcher",
            sizing_mode="stretch_width"
        )
        self.report_date_input = pn.widgets.TextInput(
            name="Report Date",
            value=datetime.now().strftime("%Y-%m-%d"),
            width=160
        )
        self.include_params_chk = pn.widgets.Checkbox(
            name="Include Pipeline Session Parameters",
            value=True,
            width=260
        )
        self.include_toc_chk = pn.widgets.Checkbox(
            name="Include Table of Contents",
            value=True,
            width=200
        )
        self.resource_mode = pn.widgets.RadioButtonGroup(
            name="Resource Mode",
            options=["CDN (Lightweight, ~50 KB)", "Inline (Offline, ~3.5 MB)"],
            value="CDN (Lightweight, ~50 KB)"
        )
        self.filename_input = pn.widgets.TextInput(
            name="HTML File Name",
            value=f"{self._export_prefix}_report.html",
            sizing_mode="stretch_width"
        )

        # ---------------- 2. Standard Plots Selection ----------------
        self.scan_plots_btn = pn.widgets.Button(
            name="🔄 Scan & Refresh Plots",
            button_type="primary",
            width=180
        )
        self.scan_plots_btn.on_click(lambda _: self.refresh_available_plots())

        self.select_all_btn = pn.widgets.Button(
            name="Select All Ready",
            button_type="default",
            width=140
        )
        self.select_all_btn.on_click(lambda _: self._set_all_selected(True))

        self.deselect_all_btn = pn.widgets.Button(
            name="Deselect All",
            button_type="default",
            width=120
        )
        self.deselect_all_btn.on_click(lambda _: self._set_all_selected(False))

        self.plots_summary_md = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.standard_plots_layout = pn.Column(sizing_mode="stretch_width")
        self._standard_plot_widgets: Dict[str, Dict[str, Any]] = {}

        # ---------------- 3. Additional Pipeline Sub-Plots ----------------
        # 3A: Sample Reconstruction
        self.sample_recon_select = pn.widgets.Select(name="Select Sample for Reconstruction", options=[], width=260)
        self.sample_recon_asinh = pn.widgets.Checkbox(name="Use asinh transform", value=False)
        self.sample_recon_title = pn.widgets.TextInput(name="Plot Title", value="Sample Reconstruction", width=260)
        self.sample_recon_caption = pn.widgets.TextInput(name="Optional Caption", value="", width=300)
        self.add_sample_recon_btn = pn.widgets.Button(name="➕ Add Sample Reconstruction", button_type="success", width=220)
        self.add_sample_recon_btn.on_click(self._on_add_sample_reconstruction)

        # 3B: Specific NMF Basis Profile
        self.basis_select = pn.widgets.Select(name="Select NMF Basis", options=[], width=180)
        self.basis_plot_type = pn.widgets.RadioButtonGroup(
            name="Plot Type",
            options=["Continuous Waveform", "Group Loadings (Jitter)"],
            value="Continuous Waveform"
        )
        self.basis_title = pn.widgets.TextInput(name="Plot Title", value="NMF Basis Profile", width=240)
        self.basis_caption = pn.widgets.TextInput(name="Optional Caption", value="", width=300)
        self.add_basis_btn = pn.widgets.Button(name="➕ Add Basis Plot", button_type="success", width=180)
        self.add_basis_btn.on_click(self._on_add_basis_plot)

        self.additional_plots_container = pn.Column(
            pn.pane.Markdown("_No additional sub-plots added yet._"),
            sizing_mode="stretch_width"
        )

        # ---------------- 4. Upload Custom Plot Images ----------------
        self.image_file_input = pn.widgets.FileInput(
            accept=".png,.jpg,.jpeg,.svg,.webp",
            multiple=False,
            sizing_mode="stretch_width"
        )
        self.image_title_input = pn.widgets.TextInput(name="Figure Title", value="External Figure", width=260)
        self.image_caption_input = pn.widgets.TextInput(name="Caption / Notes (Optional)", value="", sizing_mode="stretch_width")
        self.add_image_btn = pn.widgets.Button(name="➕ Add Uploaded Image", button_type="success", width=180)
        self.add_image_btn.on_click(self._on_add_custom_image)

        self.custom_images_container = pn.Column(
            pn.pane.Markdown("_No external images added yet._"),
            sizing_mode="stretch_width"
        )

        # ---------------- 5. Generation, Download & Preview ----------------
        self.build_report_btn = pn.widgets.Button(
            name="🚀 Build & Preview HTML Report",
            button_type="primary",
            sizing_mode="stretch_width",
            height=45
        )
        self.build_report_btn.on_click(self._on_build_report)

        self.download_report_btn = pn.widgets.FileDownload(
            label="⬇️ Download HTML Report",
            filename=self.filename_input.value,
            button_type="success",
            embed=False,
            auto=False,
            callback=self._get_report_bytes,
            disabled=True,
            sizing_mode="stretch_width",
            height=45
        )
        self.filename_input.param.watch(
            lambda e: setattr(self.download_report_btn, "filename", e.new or "report.html"), "value"
        )

        self.report_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.preview_pane = pn.pane.HTML(
            "<div style='padding: 24px; text-align: center; color: #64748b; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px;'><i>Click <b>'Build & Preview HTML Report'</b> to generate your report preview here.</i></div>",
            sizing_mode="stretch_width",
            min_height=500
        )

        # ---------------- Layout Construction ----------------
        left_controls = pn.Column(
            pn.pane.Markdown("### 1. Report Settings"),
            self.report_title_input,
            self.report_subtitle_input,
            pn.Row(self.author_input, self.report_date_input),
            pn.Row(self.include_params_chk, self.include_toc_chk),
            pn.Row(pn.pane.Markdown("**Resource Bundling:**", margin=(5, 5, 0, 0)), self.resource_mode),
            self.filename_input,
            pn.layout.Divider(),

            pn.pane.Markdown("### 2. Standard Pipeline Plots"),
            pn.Row(self.scan_plots_btn, self.select_all_btn, self.deselect_all_btn),
            self.plots_summary_md,
            self.standard_plots_layout,
            pn.layout.Divider(),

            pn.pane.Markdown("### 3. Specify Additional Sub-Plots"),
            pn.Tabs(
                ("Sample Reconstruction", pn.Column(
                    pn.Row(self.sample_recon_select, self.sample_recon_asinh),
                    pn.Row(self.sample_recon_title, self.sample_recon_caption),
                    self.add_sample_recon_btn
                )),
                ("NMF Basis Profile", pn.Column(
                    pn.Row(self.basis_select, self.basis_plot_type),
                    pn.Row(self.basis_title, self.basis_caption),
                    self.add_basis_btn
                )),
            ),
            pn.pane.Markdown("**Current Additional Sub-Plots:**", styles={"margin-top": "10px"}),
            self.additional_plots_container,
            pn.layout.Divider(),

            pn.pane.Markdown("### 4. Upload External Figures / Images"),
            pn.Row(self.image_file_input),
            pn.Row(self.image_title_input, self.image_caption_input),
            self.add_image_btn,
            pn.pane.Markdown("**Current Uploaded Figures:**", styles={"margin-top": "10px"}),
            self.custom_images_container,
            pn.layout.Divider(),

            pn.pane.Markdown("### 5. Export Report"),
            pn.Row(self.build_report_btn, self.download_report_btn),
            self.report_status,
            sizing_mode="stretch_width"
        )

        right_preview = pn.Column(
            pn.pane.Markdown("### 📄 Live HTML Report Preview"),
            self.preview_pane,
            sizing_mode="stretch_width"
        )

        self.section = pn.Column(
            pn.pane.Markdown("## HTML Report Export"),
            pn.pane.Markdown(
                "Generate an interactive, standalone HTML report containing selected pipeline figures, "
                "custom sample reconstructions, specific NMF basis waveforms, uploaded external figures, and exact parameters used during your session."
            ),
            pn.Row(left_controls, right_preview, sizing_mode="stretch_width"),
            sizing_mode="stretch_width"
        )

        # Initialize available plots
        self._init_standard_plot_definitions()
        self.refresh_available_plots()

    @property
    def export_prefix(self) -> str:
        return getattr(self, "_export_prefix", "CE_analysis")

    @export_prefix.setter
    def export_prefix(self, val: str):
        old_val = getattr(self, "_export_prefix", "")
        self._export_prefix = str(val) if val else "CE_analysis"
        if hasattr(self, "report_title_input") and self.report_title_input is not None:
            curr_title = self.report_title_input.value
            if not curr_title or (old_val and curr_title.startswith(old_val)):
                self.report_title_input.value = f"{self._export_prefix} Electropherogram Analysis Report"
        if hasattr(self, "filename_input") and self.filename_input is not None:
            curr_fn = self.filename_input.value
            if not curr_fn or (old_val and curr_fn.startswith(old_val)):
                self.filename_input.value = f"{self._export_prefix}_report.html"

    # ---------------- Standard Plot Registry ----------------
    def _init_standard_plot_definitions(self):
        """Define the catalog of all standard pipeline plots."""
        self.plot_catalog = [
            # Raw & QC
            {
                "id": "raw_electropherograms",
                "category": "Raw Data & QC",
                "default_title": "Raw Electropherograms Preview",
                "getter": lambda: self._get_plot_raw(),
                "default_selected": True
            },
            {
                "id": "snr_plot",
                "category": "Raw Data & QC",
                "default_title": "Signal-to-Noise Ratio (SNR)",
                "getter": lambda: self._get_plot_snr(),
                "default_selected": True
            },
            # Preprocessing
            {
                "id": "preproc_despike",
                "category": "Preprocessing",
                "default_title": "Despiked Traces",
                "getter": lambda: self._get_plot_despike(),
                "default_selected": False
            },
            {
                "id": "preproc_smooth",
                "category": "Preprocessing",
                "default_title": "Smoothed Traces",
                "getter": lambda: self._get_plot_smooth(),
                "default_selected": False
            },
            {
                "id": "preproc_baseline",
                "category": "Preprocessing",
                "default_title": "Baseline Subtracted Traces",
                "getter": lambda: self._get_plot_baseline(),
                "default_selected": False
            },
            {
                "id": "preproc_norm",
                "category": "Preprocessing",
                "default_title": "Normalized Electropherograms",
                "getter": lambda: self._get_plot_norm(),
                "default_selected": True
            },
            # Alignment
            {
                "id": "align_anchors",
                "category": "Alignment",
                "default_title": "Anchor Selection Curves",
                "getter": lambda: self._get_plot_align_anchors(),
                "default_selected": True
            },
            {
                "id": "align_aligned",
                "category": "Alignment",
                "default_title": "Aligned Chromatograms",
                "getter": lambda: self._get_plot_align_aligned(),
                "default_selected": True
            },
            # NMF
            {
                "id": "nmf_recon",
                "category": "NMF Decomposition",
                "default_title": "NMF Overall Sample Reconstruction",
                "getter": lambda: self._get_plot_nmf_recon(),
                "default_selected": True
            },
            {
                "id": "nmf_heatmap",
                "category": "NMF Decomposition",
                "default_title": "NMF Loadings Heatmap",
                "getter": lambda: self._get_plot_nmf_heatmap(),
                "default_selected": True
            },
            # NMF Group Stats
            {
                "id": "nmf_group_recon",
                "category": "NMF Group Statistics",
                "default_title": "Sample Reconstruction by Group / Category",
                "getter": lambda: self._get_plot_nmf_group_recon(),
                "default_selected": True
            },
            {
                "id": "nmf_group_sig",
                "category": "NMF Group Statistics",
                "default_title": "NMF Basis Significance (Effect Size)",
                "getter": lambda: self._get_plot_nmf_group_sig(),
                "default_selected": True
            },
            {
                "id": "nmf_group_pval",
                "category": "NMF Group Statistics",
                "default_title": "NMF Basis Significance (-log10 p-value)",
                "getter": lambda: self._get_plot_nmf_group_pval(),
                "default_selected": True
            },
            {
                "id": "nmf_group_jitter",
                "category": "NMF Group Statistics",
                "default_title": "Inspect Basis Loadings by Group (Jitter)",
                "getter": lambda: self._get_plot_nmf_group_jitter(),
                "default_selected": True
            },
            {
                "id": "nmf_reg_matrix",
                "category": "NMF Group Statistics",
                "default_title": "NMF Basis Regression Matrix",
                "getter": lambda: self._get_plot_nmf_reg_matrix(),
                "default_selected": False
            },
            # Alpha Diversity
            {
                "id": "alpha_diversity",
                "category": "Alpha Diversity",
                "default_title": "Alpha Diversity Metrics",
                "getter": lambda: self._get_plot_diversity(),
                "default_selected": True
            },
            # Visualization
            {
                "id": "viz_pca_scree",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Scree Plot (Explained Variance)",
                "getter": lambda: self._get_plot_viz_scree(),
                "default_selected": True
            },
            {
                "id": "viz_pca_scores",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Scores Scatter Plot",
                "getter": lambda: self._get_plot_viz_pca(),
                "default_selected": True
            },
            {
                "id": "viz_pca_loadings",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Loadings Plot",
                "getter": lambda: self._get_plot_viz_loadings(),
                "default_selected": True
            },
            {
                "id": "viz_mds",
                "category": "Dimensionality Reduction",
                "default_title": "Multidimensional Scaling (MDS)",
                "getter": lambda: self._get_plot_viz_mds(),
                "default_selected": False
            },
            {
                "id": "viz_tsne",
                "category": "Dimensionality Reduction",
                "default_title": "t-SNE Embedding",
                "getter": lambda: self._get_plot_viz_tsne(),
                "default_selected": False
            },
            # Miscellaneous
            {
                "id": "misc_cvc",
                "category": "Miscellaneous Analysis",
                "default_title": "Continuous vs Continuous Correlation",
                "getter": lambda: self._get_plot_misc_cvc(),
                "default_selected": False
            },
            {
                "id": "misc_cat",
                "category": "Miscellaneous Analysis",
                "default_title": "Continuous vs Categorical Comparison",
                "getter": lambda: self._get_plot_misc_cat(),
                "default_selected": False
            },
            {
                "id": "misc_reg",
                "category": "Miscellaneous Analysis",
                "default_title": "Regression Matrix",
                "getter": lambda: self._get_plot_misc_reg(),
                "default_selected": False
            },
            {
                "id": "misc_cat_cov",
                "category": "Miscellaneous Analysis",
                "default_title": "Categorical Covariate Correlation",
                "getter": lambda: self._get_plot_misc_cat_cov(),
                "default_selected": False
            },
            # Time Series
            {
                "id": "ts_markov",
                "category": "Time Series Analysis",
                "default_title": "Markov State Transition Diagram",
                "getter": lambda: self._get_plot_ts_markov(),
                "default_selected": False
            },
            {
                "id": "ts_trajectories",
                "category": "Time Series Analysis",
                "default_title": "Time Series Trajectories",
                "getter": lambda: self._get_plot_ts_trajectories(),
                "default_selected": False
            },
        ]

    # --- Plot Getters ---
    def _get_plot_raw(self):
        state = self.controllers.get("state")
        return getattr(state, "last_fig", None)

    def _get_plot_snr(self):
        ctrl = self.controllers.get("snr")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None))

    def _get_plot_despike(self):
        ctrl = self.controllers.get("despike")
        fig = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        return fig or extract_bokeh_figure(getattr(ctrl, "before_pane", None))

    def _get_plot_smooth(self):
        ctrl = self.controllers.get("smooth")
        fig = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        return fig or extract_bokeh_figure(getattr(ctrl, "before_pane", None))

    def _get_plot_baseline(self):
        ctrl = self.controllers.get("baseline")
        fig = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        return fig or extract_bokeh_figure(getattr(ctrl, "before_pane", None))

    def _get_plot_norm(self):
        ctrl = self.controllers.get("norm")
        fig = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        return fig or extract_bokeh_figure(getattr(ctrl, "before_pane", None))

    def _get_plot_align_anchors(self):
        ctrl = self.controllers.get("alignment")
        return extract_bokeh_figure(getattr(ctrl, "selection_pane", None)) or getattr(ctrl, "fig", None)

    def _get_plot_align_aligned(self):
        ctrl = self.controllers.get("alignment")
        return extract_bokeh_figure(getattr(ctrl, "aligned_pane", None)) or getattr(ctrl, "aligned_fig", None)

    def _get_plot_nmf_recon(self):
        ctrl = self.controllers.get("nmf")
        return extract_bokeh_figure(getattr(ctrl, "recon_pane", None))

    def _get_plot_nmf_heatmap(self):
        ctrl = self.controllers.get("nmf")
        return extract_bokeh_figure(getattr(ctrl, "heatmap_pane", None))

    def _get_plot_nmf_group_recon(self):
        ctrl = self.controllers.get("nmf_meta")
        if not ctrl: return None
        if hasattr(ctrl, "last_recon_fig") and ctrl.last_recon_fig is not None:
            return ctrl.last_recon_fig
        # search dashboard_pane
        figs = find_all_bokeh_figures(getattr(ctrl, "dashboard_pane", None))
        for f in figs:
            t = getattr(getattr(f, "title", None), "text", "").lower()
            if "reconstruction" in t or "average" in t or "group" in t:
                return f
        return figs[0] if len(figs) > 0 else None

    def _get_plot_nmf_group_sig(self):
        ctrl = self.controllers.get("nmf_meta")
        if not ctrl: return None
        if hasattr(ctrl, "last_sig_fig") and ctrl.last_sig_fig is not None:
            return ctrl.last_sig_fig
        figs = find_all_bokeh_figures(getattr(ctrl, "dashboard_pane", None))
        for f in figs:
            t = getattr(getattr(f, "title", None), "text", "").lower()
            if "effect" in t or "statistic" in t or "kruskal" in t:
                return f
        return figs[1] if len(figs) > 1 else None

    def _get_plot_nmf_group_pval(self):
        ctrl = self.controllers.get("nmf_meta")
        if not ctrl: return None
        if hasattr(ctrl, "last_pval_fig") and ctrl.last_pval_fig is not None:
            return ctrl.last_pval_fig
        figs = find_all_bokeh_figures(getattr(ctrl, "dashboard_pane", None))
        for f in figs:
            t = getattr(getattr(f, "title", None), "text", "").lower()
            if "p-value" in t or "pval" in t or "-log10" in t:
                return f
        return figs[2] if len(figs) > 2 else None

    def _get_plot_nmf_group_jitter(self):
        ctrl = self.controllers.get("nmf_meta")
        if not ctrl: return None
        if hasattr(ctrl, "last_jitter_fig") and ctrl.last_jitter_fig is not None:
            return ctrl.last_jitter_fig
        figs = find_all_bokeh_figures(getattr(ctrl, "dashboard_pane", None))
        for f in figs:
            t = getattr(getattr(f, "title", None), "text", "").lower()
            if "loadings by group" in t or "jitter" in t:
                return f
        return figs[3] if len(figs) > 3 else None

    def _get_plot_nmf_reg_matrix(self):
        ctrl = self.controllers.get("nmf_meta")
        if not ctrl: return None
        if hasattr(ctrl, "last_reg_fig") and ctrl.last_reg_fig is not None:
            return ctrl.last_reg_fig
        figs = find_all_bokeh_figures(getattr(ctrl, "basis_reg_container", None))
        return figs[0] if figs else None

    def _get_plot_diversity(self):
        ctrl = self.controllers.get("diversity")
        if not ctrl: return None
        figs = find_all_bokeh_figures(getattr(ctrl, "plots_column", None))
        if figs:
            return figs[0] # Return the first or combined
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None))

    def _get_plot_viz_scree(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "pca_scree_pane", None))

    def _get_plot_viz_pca(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "pca_pane", None))

    def _get_plot_viz_loadings(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "pc_loadings_pane", None))

    def _get_plot_viz_mds(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "mds_pane", None))

    def _get_plot_viz_tsne(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "tsne_pane", None))

    def _get_plot_misc_cvc(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "cvc_plot_pane", None))

    def _get_plot_misc_cat(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "cat_plot_pane", None))

    def _get_plot_misc_reg(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "reg_plot_pane", None))

    def _get_plot_misc_cat_cov(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "cat_cov_plot_pane", None))

    def _get_plot_ts_markov(self):
        ctrl = self.controllers.get("timeseries")
        if not ctrl: return None
        figs = find_all_bokeh_figures(getattr(ctrl, "markov_results_container", None))
        return figs[0] if figs else None

    def _get_plot_ts_trajectories(self):
        ctrl = self.controllers.get("timeseries")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None))

    # ---------------- UI Refresh Logic ----------------
    def refresh_available_plots(self):
        """Scan all controllers to see which plots are ready and update UI."""
        # 1. Update dropdown options for additional plots
        self._update_dropdowns()

        # 2. Check each standard plot availability
        ready_count = 0
        categories: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.plot_catalog:
            p_id = item["id"]
            fig = None
            try:
                fig = item["getter"]()
            except Exception:
                fig = None

            is_ready = fig is not None
            if is_ready:
                ready_count += 1

            cat = item["category"]
            if cat not in categories:
                categories[cat] = []

            # Retain or create widgets
            if p_id not in self._standard_plot_widgets:
                chk = pn.widgets.Checkbox(
                    name=item["default_title"],
                    value=item["default_selected"] and is_ready,
                    disabled=not is_ready,
                    width=280
                )
                title_inp = pn.widgets.TextInput(
                    value=item["default_title"],
                    sizing_mode="stretch_width"
                )
                self._standard_plot_widgets[p_id] = {
                    "checkbox": chk,
                    "title": title_inp,
                    "item": item
                }
            else:
                chk = self._standard_plot_widgets[p_id]["checkbox"]
                chk.disabled = not is_ready
                if not is_ready:
                    chk.value = False
                elif not chk.value and item["default_selected"]:
                    chk.value = True

            categories[cat].append({
                "item": item,
                "is_ready": is_ready,
                "widgets": self._standard_plot_widgets[p_id]
            })

        self.plots_summary_md.object = f"**Status:** Found **{ready_count}** ready plots out of {len(self.plot_catalog)} standard pipeline plots."

        # Build category accordions / rows
        cards = []
        for cat, items in categories.items():
            cat_rows = []
            for it in items:
                w = it["widgets"]
                chk = w["checkbox"]
                title_inp = w["title"]
                is_ready = it["is_ready"]
                
                status_badge = "<span style='color: #16a34a; font-weight: 600;'>[✓ Ready]</span>" if is_ready else "<span style='color: #94a3b8; font-style: italic;'>[Not generated]</span>"
                badge_pane = pn.pane.HTML(status_badge, width=120, margin=(10, 0, 0, 0))
                cat_rows.append(pn.Row(chk, badge_pane, title_inp, sizing_mode="stretch_width"))

            cat_card = pn.Card(
                *cat_rows,
                title=f"{cat} ({sum(1 for it in items if it['is_ready'])} ready)",
                collapsed=False,
                sizing_mode="stretch_width",
                styles={"background": "#ffffff", "border": "1px solid #e2e8f0", "margin-bottom": "8px"}
            )
            cards.append(cat_card)

        self.standard_plots_layout.objects = cards

    def _set_all_selected(self, select: bool):
        for p_id, w in self._standard_plot_widgets.items():
            chk = w["checkbox"]
            if not chk.disabled:
                chk.value = select

    def _update_dropdowns(self):
        """Populate sample list and basis list for additional sub-plots."""
        state = self.controllers.get("state")
        nmf_ctrl = self.controllers.get("nmf")
        samples = []
        if nmf_ctrl and hasattr(nmf_ctrl, "samples") and nmf_ctrl.samples:
            samples = list(nmf_ctrl.samples)
        elif state and hasattr(state, "converted_by_sample") and state.converted_by_sample:
            samples = list(state.converted_by_sample.keys())
        
        self.sample_recon_select.options = samples
        if samples and self.sample_recon_select.value is None:
            self.sample_recon_select.value = samples[0]

        bases = []
        if nmf_ctrl and hasattr(nmf_ctrl, "H_df") and nmf_ctrl.H_df is not None and not nmf_ctrl.H_df.empty:
            bases = [c for c in nmf_ctrl.H_df.columns if str(c).startswith("basis_")]
            if not bases:
                bases = list(nmf_ctrl.H_df.columns)
        elif nmf_ctrl and hasattr(nmf_ctrl, "k_slider"):
            K = int(nmf_ctrl.k_slider.value)
            bases = [f"basis_{i:02d}" for i in range(K)]

        self.basis_select.options = bases
        if bases and self.basis_select.value is None:
            self.basis_select.value = bases[0]

    # ---------------- 3. Additional Sub-Plots Handlers ----------------
    def _on_add_sample_reconstruction(self, _=None):
        sample = self.sample_recon_select.value
        if not sample:
            self.report_status.object = _warn("Please select a sample for reconstruction.")
            return

        nmf_ctrl = self.controllers.get("nmf")
        if not nmf_ctrl or getattr(nmf_ctrl, "H_df", None) is None or getattr(nmf_ctrl, "Phi", None) is None:
            self.report_status.object = _warn("Run NMF in the NMF tab first so continuous basis models are available.")
            return

        try:
            import CEtools as cet
            title = self.sample_recon_title.value or f"Sample Reconstruction: {sample}"
            caption = self.sample_recon_caption.value or ""
            use_asinh = self.sample_recon_asinh.value

            fig = cet.plot_reconstruction_overlays_bokeh(
                str(sample), nmf_ctrl.H_df, nmf_ctrl.Phi,
                pseudotimes_df=nmf_ctrl.pseudotimes_df, norm_df=nmf_ctrl.norm_df,
                rows_are_traces=nmf_ctrl.rows_are_traces,
                title_prefix=title,
                mask_range=(nmf_ctrl.roi_lo.value, nmf_ctrl.roi_hi.value),
                use_asinh=use_asinh
            )
            
            item_id = f"sub_recon_{sample}_{len(self.additional_pipeline_plots)}"
            self.additional_pipeline_plots.append({
                "id": item_id,
                "type": "sample_reconstruction",
                "title": title,
                "caption": caption,
                "figure": fig,
                "details": f"Sample: {sample} (asinh={use_asinh})"
            })
            self._render_additional_plots_list()
            self.report_status.object = _ok(f"Added reconstruction plot for '{sample}' to report.")
        except Exception as e:
            self.report_status.object = _warn(f"Failed to generate reconstruction: {e}")

    def _on_add_basis_plot(self, _=None):
        basis = self.basis_select.value
        if not basis:
            self.report_status.object = _warn("Please select a basis component.")
            return

        nmf_ctrl = self.controllers.get("nmf")
        nmf_meta_ctrl = self.controllers.get("nmf_meta")
        plot_type = self.basis_plot_type.value
        title = self.basis_title.value or f"NMF {basis} Profile"
        caption = self.basis_caption.value or ""

        try:
            if plot_type == "Continuous Waveform":
                if not nmf_ctrl or getattr(nmf_ctrl, "Phi", None) is None:
                    self.report_status.object = _warn("Run NMF first so continuous basis functions are available.")
                    return

                # Evaluate basis continuous curve
                Phi = nmf_ctrl.Phi
                lo = float(nmf_ctrl.roi_lo.value)
                hi = float(nmf_ctrl.roi_hi.value)
                t_grid = np.linspace(lo, hi, 400)
                basis_matrix = Phi(t_grid) # shape (400, K)

                # Find basis index
                b_idx = 0
                if basis in nmf_ctrl.H_df.columns:
                    b_idx = list(nmf_ctrl.H_df.columns).index(basis)
                elif re.search(r'\d+', str(basis)):
                    b_idx = int(re.search(r'\d+', str(basis)).group(0))

                b_idx = min(b_idx, basis_matrix.shape[1] - 1)
                y_curve = basis_matrix[:, b_idx]

                fig = bkp.figure(
                    title=title,
                    x_axis_label="pseudotime",
                    y_axis_label="basis amplitude",
                    sizing_mode="stretch_width",
                    height=320,
                    tools="pan,box_zoom,reset,save,hover"
                )
                fig.line(t_grid, y_curve, line_width=2.5, color="#1e40af", alpha=0.9)
                fig.hover.tooltips = [("pseudotime", "$x{0.000}"), ("amplitude", "$y{0.000}")]

            else: # Group Loadings (Jitter)
                if not nmf_meta_ctrl or getattr(nmf_meta_ctrl, "H_df", None) is None or getattr(nmf_meta_ctrl, "metadata_df", None) is None:
                    self.report_status.object = _warn("NMF Group Stats must be loaded with metadata to generate group jitter plot.")
                    return
                
                # Create jitter figure
                group_col = nmf_meta_ctrl.group_col_select.value
                sample_col = nmf_meta_ctrl.sample_col_select.value
                H_df = nmf_meta_ctrl.H_df.copy()
                meta_df = nmf_meta_ctrl.metadata_df.copy()

                if "Unnamed: 0" not in H_df.columns:
                    H_df["Unnamed: 0"] = H_df.index.astype(str)

                meta_df[sample_col] = meta_df[sample_col].astype(str)
                H_df["Unnamed: 0"] = H_df["Unnamed: 0"].astype(str)
                merged = pd.merge(H_df, meta_df, left_on="Unnamed: 0", right_on=sample_col)

                groups = sorted(merged[group_col].dropna().astype(str).unique())
                x_map = {g: i for i, g in enumerate(groups)}
                d = merged[["Unnamed: 0", group_col, basis]].dropna()
                
                rng = np.random.default_rng(0)
                xs = np.array([x_map[str(g)] for g in d[group_col]], dtype=float) + rng.normal(0, 0.05, size=len(d))
                ys = pd.to_numeric(d[basis], errors="coerce").to_numpy()

                fig = bkp.figure(
                    title=title,
                    x_axis_label=group_col,
                    y_axis_label="NMF Loading",
                    sizing_mode="stretch_width",
                    height=360,
                    tools="pan,box_zoom,reset,save,hover"
                )
                fig.scatter(xs, ys, size=7, color="#2563eb", alpha=0.7)
                fig.xaxis.ticker = list(range(len(groups)))
                fig.xaxis.major_label_overrides = dict(enumerate(groups))
                fig.hover.tooltips = [("Group", "@x"), ("Loading", "$y{0.000}")]

            item_id = f"sub_basis_{basis}_{len(self.additional_pipeline_plots)}"
            self.additional_pipeline_plots.append({
                "id": item_id,
                "type": "basis_plot",
                "title": title,
                "caption": caption,
                "figure": fig,
                "details": f"{basis} ({plot_type})"
            })
            self._render_additional_plots_list()
            self.report_status.object = _ok(f"Added {plot_type} for '{basis}' to report.")
        except Exception as e:
            self.report_status.object = _warn(f"Failed to generate basis plot: {e}")

    def _render_additional_plots_list(self):
        if not self.additional_pipeline_plots:
            self.additional_plots_container.objects = [pn.pane.Markdown("_No additional sub-plots added yet._")]
            return

        rows = []
        for idx, item in enumerate(self.additional_pipeline_plots):
            title_pane = pn.pane.Markdown(f"**{item['title']}** <span style='color: #64748b;'>({item['details']})</span>")
            del_btn = pn.widgets.Button(name="✕ Remove", button_type="danger", width=90)
            def _make_remover(i=idx):
                return lambda _: self._remove_additional_plot(i)
            del_btn.on_click(_make_remover(idx))
            rows.append(pn.Row(title_pane, pn.Spacer(sizing_mode="stretch_width"), del_btn, sizing_mode="stretch_width"))

        self.additional_plots_container.objects = rows

    def _remove_additional_plot(self, index: int):
        if 0 <= index < len(self.additional_pipeline_plots):
            self.additional_pipeline_plots.pop(index)
            self._render_additional_plots_list()

    # ---------------- 4. Custom Uploaded Images Handlers ----------------
    def _on_add_custom_image(self, _=None):
        if not self.image_file_input.value:
            self.report_status.object = _warn("Please select an image file first.")
            return

        filename = self.image_file_input.filename or "image.png"
        raw_bytes = self.image_file_input.value
        ext = os.path.splitext(filename)[1].lower().replace(".", "")
        if ext == "svg":
            mime = "image/svg+xml"
        elif ext in ["jpg", "jpeg"]:
            mime = "image/jpeg"
        elif ext == "webp":
            mime = "image/webp"
        else:
            mime = "image/png"

        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64_str}"

        title = self.image_title_input.value or os.path.splitext(filename)[0]
        caption = self.image_caption_input.value or ""

        item_id = f"custom_img_{len(self.custom_images)}"
        self.custom_images.append({
            "id": item_id,
            "title": title,
            "caption": caption,
            "data_uri": data_uri,
            "filename": filename
        })
        self._render_custom_images_list()
        self.report_status.object = _ok(f"Added custom figure '{title}' to report.")
        # Clear inputs
        self.image_file_input.value = None

    def _render_custom_images_list(self):
        if not self.custom_images:
            self.custom_images_container.objects = [pn.pane.Markdown("_No external images added yet._")]
            return

        rows = []
        for idx, item in enumerate(self.custom_images):
            thumb_html = f"<div style='display:flex; align-items:center; gap:12px;'><img src='{item['data_uri']}' style='height:45px; width:65px; object-fit:cover; border-radius:4px; border:1px solid #cbd5e1;' /> <div><b>{item['title']}</b><br><span style='font-size:0.85em; color:#64748b;'>{item['filename']}</span></div></div>"
            info_pane = pn.pane.HTML(thumb_html, sizing_mode="stretch_width")
            del_btn = pn.widgets.Button(name="✕ Remove", button_type="danger", width=90)
            def _make_remover(i=idx):
                return lambda _: self._remove_custom_image(i)
            del_btn.on_click(_make_remover(idx))
            rows.append(pn.Row(info_pane, pn.Spacer(sizing_mode="stretch_width"), del_btn, sizing_mode="stretch_width"))

        self.custom_images_container.objects = rows

    def _remove_custom_image(self, index: int):
        if 0 <= index < len(self.custom_images):
            self.custom_images.pop(index)
            self._render_custom_images_list()

    # ---------------- 5. Report HTML Assembly & Generation ----------------
    def _on_build_report(self, _=None):
        self.report_status.object = "Generating HTML report..."
        try:
            html_content, plot_count = self._compile_html_document()
            self._current_html_str = html_content
            self.download_report_btn.disabled = False
            
            size_kb = len(html_content.encode("utf-8")) / 1024.0
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{(size_kb/1024):.2f} MB"
            self.report_status.object = _ok(f"HTML Report generated successfully! Includes **{plot_count}** plots/figures ({size_str}).")

            # Update preview pane using isolated base64 iframe
            b64_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
            iframe_tag = f"<iframe src='data:text/html;base64,{b64_html}' style='width: 100%; height: 750px; border: 1px solid #cbd5e1; border-radius: 8px; background: white;'></iframe>"
            self.preview_pane.object = iframe_tag
        except Exception as e:
            self.download_report_btn.disabled = True
            self.report_status.object = _warn(f"Failed to generate report: {e}")

    def _get_report_bytes(self) -> io.BytesIO:
        if not self._current_html_str:
            html_content, _ = self._compile_html_document()
            self._current_html_str = html_content
        return io.BytesIO(self._current_html_str.encode("utf-8"))

    def _compile_html_document(self) -> Tuple[str, int]:
        """Compile selected standard plots, additional subplots, custom images, and parameters into HTML."""
        # 1. Collect standard plots
        selected_standard_plots = []
        for item in self.plot_catalog:
            p_id = item["id"]
            if p_id in self._standard_plot_widgets:
                w = self._standard_plot_widgets[p_id]
                if w["checkbox"].value:
                    fig = item["getter"]()
                    if fig is not None:
                        user_title = w["title"].value or item["default_title"]
                        selected_standard_plots.append({
                            "id": p_id,
                            "category": item["category"],
                            "title": user_title,
                            "figure": fig,
                            "caption": ""
                        })

        # 2. Collect additional subplots
        selected_sub_plots = []
        for sp in self.additional_pipeline_plots:
            fig = sp.get("figure", sp.get("fig"))
            if fig is not None:
                selected_sub_plots.append({
                    "id": sp["id"],
                    "category": "Additional Pipeline Sub-Plots",
                    "title": sp["title"],
                    "figure": fig,
                    "caption": sp.get("caption", "")
                })

        # Check total items
        total_items = len(selected_standard_plots) + len(selected_sub_plots) + len(self.custom_images)
        if total_items == 0:
            raise ValueError("No plots or images selected! Please select at least one standard plot or add an additional figure.")

        # 3. Extract Bokeh figures dictionary for components()
        bokeh_dict = {}
        for p in selected_standard_plots:
            bokeh_dict[p["id"]] = p["figure"]
        for p in selected_sub_plots:
            bokeh_dict[p["id"]] = p["figure"]

        # Run Bokeh components
        script, divs = components(bokeh_dict) if bokeh_dict else ("", {})

        # Resources (CDN vs Inline)
        is_inline = "Inline" in self.resource_mode.value
        res = INLINE if is_inline else CDN
        bokeh_js = res.render_js()
        bokeh_css = res.render_css()

        # 4. Build Session Parameters Table
        params_html = ""
        state = self.controllers.get("state")
        session_log = getattr(state, "session_log", {}) if state else {}
        if self.include_params_chk.value and session_log:
            params_html = "<div class='section-card' id='sec-params'>"
            params_html += "<h2>⚙️ Pipeline Parameters & Session Log</h2>"
            params_html += "<div class='params-grid'>"
            for step_name, step_dict in session_log.items():
                params_html += f"<div class='param-card'><h4>{step_name}</h4><table>"
                if isinstance(step_dict, dict):
                    for k, v in step_dict.items():
                        params_html += f"<tr><td class='param-key'>{k}</td><td class='param-val'>{v}</td></tr>"
                params_html += "</table></div>"
            params_html += "</div></div>"

        # 5. Build Grouped Plot Cards & Table of Contents
        toc_items = []
        body_sections_html = ""

        # Group plots by category
        grouped_plots: Dict[str, List[Dict[str, Any]]] = {}
        for p in selected_standard_plots:
            cat = p["category"]
            grouped_plots.setdefault(cat, []).append(p)
        if selected_sub_plots:
            grouped_plots["Additional Pipeline Sub-Plots"] = selected_sub_plots

        sec_idx = 1
        if self.include_params_chk.value and session_log:
            toc_items.append({"id": "sec-params", "title": "Pipeline Parameters & Session Log"})

        for cat, plots in grouped_plots.items():
            cat_id = f"sec-cat-{sec_idx}"
            sec_idx += 1
            toc_items.append({"id": cat_id, "title": cat})

            body_sections_html += f"<div class='section-card' id='{cat_id}'>"
            body_sections_html += f"<div class='section-header'><h2>{cat}</h2><span class='badge'>{len(plots)} figure{'s' if len(plots)>1 else ''}</span></div>"
            body_sections_html += "<div class='plots-grid'>"

            for p in plots:
                p_div = divs.get(p["id"], "<div style='color:red;'>Plot render unavailable</div>")
                cap_html = f"<p class='plot-caption'>{p['caption']}</p>" if p.get("caption") else ""
                body_sections_html += f"""
                <div class='plot-card'>
                    <div class='plot-card-header'>
                        <h3>{p['title']}</h3>
                    </div>
                    <div class='plot-card-body'>
                        {p_div}
                    </div>
                    {cap_html}
                </div>
                """
            body_sections_html += "</div></div>"

        # 6. Build Custom Uploaded Images Section
        if self.custom_images:
            cat_id = "sec-custom-images"
            toc_items.append({"id": cat_id, "title": "External Figures & Annotations"})
            body_sections_html += f"<div class='section-card' id='{cat_id}'>"
            body_sections_html += f"<div class='section-header'><h2>External Figures & Annotations</h2><span class='badge'>{len(self.custom_images)} image{'s' if len(self.custom_images)>1 else ''}</span></div>"
            body_sections_html += "<div class='plots-grid'>"

            for img in self.custom_images:
                cap_html = f"<p class='plot-caption'>{img['caption']}</p>" if img.get("caption") else ""
                body_sections_html += f"""
                <div class='plot-card'>
                    <div class='plot-card-header'>
                        <h3>{img['title']}</h3>
                        <span class='badge-secondary'>{img['filename']}</span>
                    </div>
                    <div class='plot-card-body' style='text-align: center; padding: 15px;'>
                        <img src='{img['data_uri']}' alt='{img['title']}' style='max-width: 100%; max-height: 650px; height: auto; border-radius: 6px;' />
                    </div>
                    {cap_html}
                </div>
                """
            body_sections_html += "</div></div>"

        # 7. Build Table of Contents HTML
        toc_html = ""
        if self.include_toc_chk.value and toc_items:
            toc_html = "<div class='toc-container'><h3>📋 Table of Contents</h3><div class='toc-links'>"
            for t in toc_items:
                toc_html += f"<a href='#{t['id']}' class='toc-item'>{t['title']}</a>"
            toc_html += "</div></div>"

        # 8. Report Header Info
        title = self.report_title_input.value or "CEtools Analysis Report"
        subtitle = self.report_subtitle_input.value or ""
        author = self.author_input.value or ""
        date_str = self.report_date_input.value or datetime.now().strftime("%Y-%m-%d")

        meta_pills = f"<div class='meta-pills'>"
        meta_pills += f"<span class='pill'><b>Date:</b> {date_str}</span>"
        if author:
            meta_pills += f"<span class='pill'><b>Analyst:</b> {author}</span>"
        meta_pills += f"<span class='pill'><b>Total Figures:</b> {total_items}</span>"
        meta_pills += f"<span class='pill'><b>Engine:</b> CEtools / Bokeh {bokeh.__version__}</span>"
        meta_pills += "</div>"

        subtitle_html = f"<p class='report-subtitle'>{subtitle}</p>" if subtitle else ""

        # 9. Complete HTML Document Assembly
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{bokeh_js}
{bokeh_css}
<style>
  :root {{
    --primary: #1e40af;
    --primary-light: #eff6ff;
    --border: #e2e8f0;
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-page);
    color: var(--text-main);
    margin: 0;
    padding: 30px 15px;
    line-height: 1.5;
  }}
  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}
  .header-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  }}
  .header-card h1 {{
    margin: 0 0 8px 0;
    font-size: 2.1rem;
    color: #1e3a8a;
    font-weight: 700;
  }}
  .report-subtitle {{
    font-size: 1.15rem;
    color: var(--text-muted);
    margin: 0 0 16px 0;
  }}
  .meta-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
  }}
  .pill {{
    background: var(--primary-light);
    color: var(--primary);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.88rem;
    border: 1px solid #bfdbfe;
  }}
  .toc-container {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 28px;
  }}
  .toc-container h3 {{ margin: 0 0 12px 0; font-size: 1.1rem; }}
  .toc-links {{ display: flex; flex-wrap: wrap; gap: 8px 16px; }}
  .toc-item {{
    color: var(--primary);
    text-decoration: none;
    font-size: 0.95rem;
    padding: 4px 8px;
    border-radius: 6px;
    transition: background 0.15s;
  }}
  .toc-item:hover {{ background: var(--primary-light); text-decoration: underline; }}
  .section-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 32px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
  }}
  .section-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 12px;
    margin-bottom: 24px;
  }}
  .section-header h2 {{ margin: 0; font-size: 1.5rem; color: #1e293b; }}
  .badge {{
    background: var(--primary-light);
    color: var(--primary);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
  }}
  .badge-secondary {{
    background: #f1f5f9;
    color: var(--text-muted);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
  }}
  .plots-grid {{
    display: flex;
    flex-direction: column;
    gap: 28px;
  }}
  .plot-card {{
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
  }}
  .plot-card-header {{
    background: #f8fafc;
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .plot-card-header h3 {{
    margin: 0;
    font-size: 1.15rem;
    color: #1e293b;
    font-weight: 600;
  }}
  .plot-card-body {{
    padding: 16px;
    overflow-x: auto;
  }}
  .plot-caption {{
    margin: 0;
    padding: 12px 20px;
    background: #f8fafc;
    border-top: 1px solid var(--border);
    font-size: 0.92rem;
    color: var(--text-muted);
    font-style: italic;
  }}
  .params-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
    gap: 16px;
    margin-top: 16px;
  }}
  .param-card {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }}
  .param-card h4 {{ margin: 0 0 10px 0; color: var(--primary); font-size: 1rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  .param-card table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  .param-key {{ color: var(--text-muted); font-weight: 500; padding: 4px 0; }}
  .param-val {{ text-align: right; font-weight: 600; color: #1e293b; padding: 4px 0; }}
  .footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }}
  @media print {{
    body {{ background: white; padding: 0; }}
    .container {{ max-width: 100%; }}
    .toc-container {{ display: none; }}
    .section-card {{ box-shadow: none; border: none; padding: 0; margin-bottom: 20px; }}
    .plot-card {{ page-break-inside: avoid; break-inside: avoid; margin-bottom: 24px; box-shadow: none; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header-card">
    <h1>{title}</h1>
    {subtitle_html}
    {meta_pills}
  </div>

  {toc_html}
  {params_html}
  {body_sections_html}

  <div class="footer">
    Report compiled on {date_str} using <b>CEtools</b> pipeline.
  </div>
</div>
{script}
</body>
</html>"""
        return full_html, total_items

def build_html_report_section(controllers: Dict[str, Any]) -> Tuple[pn.Column, HTMLReportController]:
    """Factory function to build and return the HTML report export tab layout and controller."""
    ctrl = HTMLReportController(controllers)
    return ctrl.section, ctrl
