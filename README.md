# SADAT — Flat GitHub package

All files are side by side. Upload the extracted files to the repository root; do not upload the ZIP as the application. There are no nested folders in this distribution.

## Run / deploy

- Python 3.12 (Python 3.11+ required for tomllib).
- Install: `python -m pip install -r requirements.txt`.
- Run: `streamlit run app.py`.
- Streamlit Cloud entrypoint: `app.py`.

The original `config.toml`, `theme.css`, logo and both model files are unchanged. The app reads all of them from its own directory. The original theme configuration is loaded programmatically using the config API of the pinned Streamlit 1.65.0 release; an initial rerun sends it to the browser. Keep this version pin unless the bootstrap is retested.

Compatibility fixes from the tested package are retained: scikit-learn 1.8.0 matches the saved models, and tracking labels use captions instead of an unsupported st.code argument. No model inference or complaint workflow logic was changed.

The research script and design reference are preserved for reference; they are not executed by the app.

This student-only version creates a local SQLite database at runtime. Hosted storage may reset. It does not implement staff processing or resolution; ratings concern the submission experience.
