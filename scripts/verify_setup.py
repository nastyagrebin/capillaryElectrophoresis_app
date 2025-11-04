
## `scripts/verify_setup.py`
```python
import importlib, sys

def check(mod, attr=None, expect=None):
    m = importlib.import_module(mod)
    v = getattr(m, "__version__", "unknown")
    ok = True
    if attr and expect:
        got = getattr(m, attr)
        ok = (str(got) == str(expect))
    print(f"✅ {mod} version: {v}")
    return m

def main():
    bokeh = check("bokeh")
    panel = check("panel")
    pandas = check("pandas")
    numpy = check("numpy")
    scipy = check("scipy")
    sklearn = check("sklearn")
    cet = check("CEtools")
    # Light import smoke tests on CEtools functions you rely on:
    needed = [
        "fit_continuous_basis_loadings_from_dataframes",
        "plot_reconstruction_overlays_bokeh",
        "plot_loadings_heatmap_bokeh",
        "embed_with_mds",
        "embed_with_tsne",
        "prepare_features",
    ]
    missing = [name for name in needed if not hasattr(cet, name)]
    if missing:
        print("❌ CEtools missing:", ", ".join(missing))
        sys.exit(1)
    print("All good. You can run: panel serve app/app.py --show")
if __name__ == "__main__":
    main()
