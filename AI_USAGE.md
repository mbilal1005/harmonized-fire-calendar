# AI usage

| Step | Tool / model | What it did | Human role |
|---|---|---|---|
| Project spec | Team | Wrote the task spec, method and validation design | Authored |
| Data extraction + inspection | Claude Code (Claude Opus 5.5) | Unzipped FIRMS downloads, inspected columns, date ranges, value distributions | Reviewed output |
| pipeline.py step 1 (load/filter/report) | Claude Code (Claude Opus 5.5) | Wrote loader: sensor detection, type==0 filter, confidence filters, archive/NRT dedup, sanity-check report | Reviewed numbers at STOP |
| Method review | Claude Code (Claude Fable 5.2) | Reviewed the method for validity risks (circular headline metric, weak validation design, megafire-dominated factor, NRT bias, spatial heterogeneity), backed by checks on the overlap data | Team chose which fixes to apply |
| pipeline.py steps 2-6 | Claude Code (Claude Fable 5.2) | Wrote NRT static-source mask, 0.5 deg aggregation, per-tile factor models (pooled, calendar-month, power law), leave-one-year-out and forward validation, model selection, headline numbers | Specified method and fixes; reviewed numbers at STOP |
| app.py (Streamlit app) | Claude Code (Claude Fable 5.2) | Wrote the app: region/custom box selection, metrics, naive vs harmonized vs MODIS chart, calendar heatmap with anomaly outlines, validation panel, downloads, method and limits; tested headlessly | Specified features and wording priorities; reviews in browser |
