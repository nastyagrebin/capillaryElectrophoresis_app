import re

with open("app/viz_utils.py", "r") as f:
    content = f.read()

# 1. Init variables
init_vars = """
        self.diversity_df = None  # NEW: rows=samples, cols=metrics
        
        self.rules = []
        self.sliced_df = None
        
        # ---- Slicing UI ----
        self.slice_col = pn.widgets.Select(name="Variable to Slice By", options=[])
        self.slice_type = pn.widgets.RadioButtonGroup(name="Type", options=["Categorical", "Continuous"], value="Categorical")
        self.slice_val_cat = pn.widgets.MultiChoice(name="Categories to Include", options=[], visible=True)
        self.slice_val_cont_min = pn.widgets.FloatInput(name="Min Value", value=0.0, visible=False)
        self.slice_val_cont_max = pn.widgets.FloatInput(name="Max Value", value=100.0, visible=False)
        self.slice_add_btn = pn.widgets.Button(name="Add Rule", button_type="primary")
        self.slice_clear_btn = pn.widgets.Button(name="Clear Rules", button_type="danger")
        self.slice_apply_btn = pn.widgets.Button(name="Apply Filters", button_type="success")
        self.slice_rules_md = pn.pane.Markdown("**Active Rules:** None", sizing_mode="stretch_width")
"""
content = content.replace("        self.diversity_df: Optional[pd.DataFrame] = None  # NEW: rows=samples, cols=metrics", init_vars)

# 2. Watchers
watchers = """
        self.meta_load_btn.on_click(self._on_load_meta_file)
        self.visualize_btn.on_click(self._on_visualize_enable)
        
        self.slice_col.param.watch(self._update_slicing_ui, 'value')
        self.slice_type.param.watch(self._update_slicing_ui, 'value')
        self.slice_add_btn.on_click(self._add_rule)
        self.slice_clear_btn.on_click(self._clear_rules)
        self.slice_apply_btn.on_click(self._apply_slicing)
"""
content = content.replace("""        self.meta_load_btn.on_click(self._on_load_meta_file)
        self.visualize_btn.on_click(self._on_visualize_enable)""", watchers)

# 3. Layout
layout = """
            self.meta_table,
            pn.Row(self.svg_export, pn.Spacer(width=12), self.visualize_btn),
            pn.layout.Divider(),
            pn.pane.Markdown("### 3. Subset Data (Optional)"),
            pn.Row(self.slice_col, pn.Column("Type:", self.slice_type)),
            pn.Row(self.slice_val_cat, self.slice_val_cont_min, self.slice_val_cont_max),
            pn.Row(self.slice_add_btn, self.slice_clear_btn, self.slice_apply_btn),
            self.slice_rules_md,
            pn.layout.Divider(),
"""
content = content.replace("""            self.meta_table,
            pn.Row(self.svg_export, pn.Spacer(width=12), self.visualize_btn),
            pn.layout.Divider(),""", layout)

# 4. Slicing logic
slicing_logic = """
    # ---------- Slicing ----------
    def _get_master_df(self) -> pd.DataFrame:
        dfs = []
        if self.H_df is not None and not self.H_df.empty:
            dfs.append(self.H_df.copy())
            
        try:
            meta = pd.DataFrame(self.meta_table.value)
            if not meta.empty and "sample" in meta.columns:
                m = meta.set_index("sample")
                m.index = m.index.astype(str)
                dfs.append(m)
        except Exception:
            pass
            
        if self.diversity_df is not None and not self.diversity_df.empty:
            dfs.append(self.diversity_df.copy())
            
        if not dfs: return None
        master = dfs[0]
        for d in dfs[1:]:
            master = master.merge(d, left_index=True, right_index=True, how='outer', suffixes=("", "_dup"))
        return master.loc[:, ~master.columns.duplicated()].copy()

    def _update_slice_options(self):
        mdf = self._get_master_df()
        if mdf is not None:
            cols = list(mdf.columns)
            self.slice_col.options = cols
            if cols and self.slice_col.value not in cols:
                self.slice_col.value = cols[0]
            self._update_slicing_ui()

    def _update_slicing_ui(self, event=None):
        mdf = self._get_master_df()
        if mdf is None or not self.slice_col.value: return
        is_cat = (self.slice_type.value == "Categorical")
        self.slice_val_cat.visible = is_cat
        self.slice_val_cont_min.visible = not is_cat
        self.slice_val_cont_max.visible = not is_cat
        
        col = self.slice_col.value
        if is_cat:
            cats = [str(x) for x in mdf[col].dropna().unique()]
            self.slice_val_cat.options = cats
            self.slice_val_cat.value = []
        else:
            try:
                numeric_series = pd.to_numeric(mdf[col], errors='coerce').dropna()
                self.slice_val_cont_min.value = float(numeric_series.min()) if not numeric_series.empty else 0.0
                self.slice_val_cont_max.value = float(numeric_series.max()) if not numeric_series.empty else 100.0
            except:
                pass

    def _add_rule(self, event):
        if not self.slice_col.value: return
        
        is_cat = (self.slice_type.value == "Categorical")
        if is_cat:
            if not self.slice_val_cat.value: return
            self.rules.append({
                "col": self.slice_col.value,
                "type": "categorical",
                "categories": list(self.slice_val_cat.value)
            })
        else:
            self.rules.append({
                "col": self.slice_col.value,
                "type": "continuous",
                "min": self.slice_val_cont_min.value,
                "max": self.slice_val_cont_max.value
            })
        self._update_rules_display()

    def _clear_rules(self, event):
        self.rules = []
        self.sliced_df = None
        self._update_rules_display()
        
    def _update_rules_display(self):
        if not self.rules:
            self.slice_rules_md.object = "**Active Rules:** None"
            return
            
        md = "**Active Rules (AND Logic):**\n"
        for i, r in enumerate(self.rules):
            if r["type"] == "categorical":
                md += f"- **{r['col']}** is in `{r['categories']}`\n"
            else:
                md += f"- **{r['col']}** between `{r['min']}` and `{r['max']}`\n"
        self.slice_rules_md.object = md

    def _apply_slicing(self, event):
        df = self._get_master_df()
        if df is None: return
        
        for r in self.rules:
            if r["type"] == "categorical":
                df = df[df[r["col"]].astype(str).isin(r["categories"])]
            else:
                df[r["col"]] = pd.to_numeric(df[r["col"]], errors='coerce')
                df = df[(df[r["col"]] >= r["min"]) & (df[r["col"]] <= r["max"])]
                
        self.sliced_df = df.copy()
        self.status.object = f"**Data Sliced:** {len(self.sliced_df)} samples remain."
        self._refresh_all()

    def _get_valid_samples(self):
        if self.sliced_df is not None:
            return list(map(str, self.sliced_df.index))
        return self.samples

    # ---------- External API ----------
"""
content = content.replace("    # ---------- External API ----------", slicing_logic)

# 5. Populate options after load
# In set_input
content = content.replace('        self.pca_pane.object = None; self.mds_pane.object = None; self.tsne_pane.object = None\n', '        self.pca_pane.object = None; self.mds_pane.object = None; self.tsne_pane.object = None\n        self._update_slice_options()\n')
# In _on_load_file
content = content.replace('        self.status.object = ok("Metadata loaded. You can edit cells directly in the table.")\n', '        self.status.object = ok("Metadata loaded. You can edit cells directly in the table.")\n        self._update_slice_options()\n')
# In set_diversity
content = content.replace('        self._populate_div_metric_choices()\n', '        self._populate_div_metric_choices()\n        self._update_slice_options()\n')


# 6. Apply filter to PCA
pca_filter = """
            valid_samples = self._get_valid_samples()
            H_subset = self.H_df.loc[self.H_df.index.isin(valid_samples)].copy()
            if H_subset.empty: return
            
            X = cet.prepare_features(H_subset, row_norm="l1", zscore_cols=True)
            ncom = int(self.pca_n.value)
            pcx  = int(self.pca_x.value)
            pcy  = int(self.pca_y.value)
            ncom_eff = max(ncom, pcx, pcy)

            pca = PCA(n_components=ncom_eff, random_state=0)
            scores = pca.fit_transform(X)
            ix, iy = pcx - 1, pcy - 1
            x = scores[:, ix]; y = scores[:, iy]
            labels = np.array(list(map(str, H_subset.index.astype(str))))
"""
content = content.replace("""
            X = cet.prepare_features(self.H_df, row_norm="l1", zscore_cols=True)
            ncom = int(self.pca_n.value)
            pcx  = int(self.pca_x.value)
            pcy  = int(self.pca_y.value)
            ncom_eff = max(ncom, pcx, pcy)

            pca = PCA(n_components=ncom_eff, random_state=0)
            scores = pca.fit_transform(X)
            ix, iy = pcx - 1, pcy - 1
            x = scores[:, ix]; y = scores[:, iy]
            labels = np.array(list(map(str, self.H_df.index.astype(str))))
""", pca_filter)

# 7. Apply filter to MDS
mds_filter = """
        try:
            import CEtools as cet
            
            valid_samples = self._get_valid_samples()
            H_subset = self.H_df.loc[self.H_df.index.isin(valid_samples)].copy()
            if H_subset.empty: return

            with no_bokeh_show():
                coords, _, mds_fig = cet.embed_with_mds(
                    H_subset,
"""
content = content.replace("""
        try:
            import CEtools as cet
            with no_bokeh_show():
                coords, _, mds_fig = cet.embed_with_mds(
                    self.H_df,
""", mds_filter)

mds_labels = """
                x, y = coords[:, 0], coords[:, 1]
                labels = np.array(list(map(str, H_subset.index.astype(str))))
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
"""
content = content.replace("""
                x, y = coords[:, 0], coords[:, 1]
                labels = np.array(list(map(str, self.H_df.index.astype(str))))
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
""", mds_labels)


# 8. Apply filter to tSNE
tsne_filter = """
        try:
            import CEtools as cet
            
            valid_samples = self._get_valid_samples()
            H_subset = self.H_df.loc[self.H_df.index.isin(valid_samples)].copy()
            if H_subset.empty: return

            with no_bokeh_show():
                coords, _, tsne_fig = cet.embed_with_tsne(
                    H_subset,
"""
content = content.replace("""
        try:
            import CEtools as cet
            with no_bokeh_show():
                coords, _, tsne_fig = cet.embed_with_tsne(
                    self.H_df,
""", tsne_filter)

tsne_labels = """
                x, y = coords[:, 0], coords[:, 1]
                labels = np.array(list(map(str, H_subset.index.astype(str))))
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
"""
content = content.replace("""
                x, y = coords[:, 0], coords[:, 1]
                labels = np.array(list(map(str, self.H_df.index.astype(str))))
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
""", tsne_labels)

with open("app/viz_utils_patched.py", "w") as f:
    f.write(content)
