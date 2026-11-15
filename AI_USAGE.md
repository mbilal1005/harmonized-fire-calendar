# AI usage

| Step | Tool / model | What it did | Human role |
|---|---|---|---|
| Project spec | Team | Wrote the task spec, method and validation design | Authored |
| Data extraction + inspection | Claude Code (Claude Opus 5.5) | Unzipped FIRMS downloads, inspected columns, date ranges, value distributions | Reviewed output |
| pipeline.py step 1 (load/filter/report) | Claude Code (Claude Opus 5.5) | Wrote loader: sensor detection, type==0 filter, confidence filters, archive/NRT dedup, sanity-check report | Reviewed numbers at STOP |
| Method review | Claude Code (Claude Fable 5.2) | Reviewed the method for validity risks (circular headline metric, weak validation design, megafire-dominated factor, NRT bias, spatial heterogeneity), backed by checks on the overlap data | Team chose which fixes to apply |
| pipeline.py steps 2-6 | Claude Code (Claude Fable 5.2) | Wrote NRT static-source mask, 0.5 deg aggregation, per-tile factor models (pooled, calendar-month, power law), leave-one-year-out and forward validation, model selection, headline numbers | Specified method and fixes; reviewed numbers at STOP |
| app.py (Streamlit app) | Claude Code (Claude Fable 5.2) | Wrote the app: region/custom box selection, metrics, naive vs harmonized vs MODIS chart, calendar heatmap with anomaly outlines, validation panel, downloads, method and limits; tested headlessly | Specified features and wording priorities; reviews in browser |
| README.md | Claude Code (Claude Fable 5.2) | Wrote the README from the pipeline outputs | Reviews wording and numbers |
| App refinements | Claude Code (Claude Fable 5.2) | Inferno calendar, month-to-date comparison, log-scale anomaly rule, fire-event labels, template-based season briefing (no LLM text at runtime) | Specified each change; reviews in browser |
| App restructure: two tabs, map | Claude Code (Claude Fable 5.2) | Split the app into a plain-language planner tab and a method tab; added the folium density map with rectangle drawing | Specified structure and audience; reviews in browser |
| forecast.py (ML forecast test) | Claude Code (Claude Fable 5.2) | Wrote and ran the gradient-boosting forecast test against the seasonal baseline (scikit-learn, trained locally on our aggregates; no LLM involved in predictions). Result: baseline as good or better, so no forecast is shown | Specified features, test design and the decision rule |
| Greece October check | Claude Code (Claude Fable 5.2) | Compared October with July-August detections on fire power, night share, daily counts and spatial spread; added a data-driven small-fire sentence to the planner tab | Asked the question; set the evidence bar |
| "Where it's unusual" map | Claude Code (Claude Fable 5.2) | Fixed the map view bug; added the 0.1 deg display grid, three comparison views, the without-correction toggle and a permutation test for season shifts; checked the result in a headless browser | Specified views and wording; reviews in browser |
