# AI usage

| Step | Tool / model | What it did | Human role |
|---|---|---|---|
| Challenge choice and planning | Claude (claude.ai, Claude Opus 5.5) | Compared challenges, suggested scope, customer framing, data sources and the prompts used in Claude Code | Made the decisions on challenge, customer, regions and features |
| Project spec | Claude (claude.ai) + Bilal | Drafted the task spec, method and validation design (CLAUDE.md) | Edited and approved the spec |
| Data extraction and inspection | Claude Code (Claude Opus 5.5) | Unzipped FIRMS downloads, inspected columns, date ranges and value distributions | Reviewed output, checked known fires as sanity check |
| pipeline.py (harmonization and validation) | Claude Code ([model]) | Filtering, 0.5° grid aggregation, VIIRS-to-MODIS conversion factors per tile and calendar month, leave-one-year-out and forward validation, headline numbers | Set the method and decision rules; chose to exclude 2026 from the fit |
| Method review | Claude Code (Claude Fable 5.2) | Reviewed the method for validity risks (circular headline metric, weak validation design, megafire-dominated factor, NRT bias, spatial heterogeneity), backed by checks on the overlap data | Chose which fixes to apply |
| app.py (Streamlit app) | Claude Code (Claude Fable 5.2) | Wrote the app: region and custom box selection, metrics, naive vs harmonized vs MODIS chart, calendar heatmap with anomaly outlines, validation panel, downloads, method and limits; tested headlessly | Specified features and wording; reviewed in browser |
| README.md | Claude Code (Claude Fable 5.2) | Wrote the README from the pipeline outputs | Reviewed wording and numbers |
| App refinements | Claude Code (Claude Fable 5.2) | Inferno calendar, month-to-date comparison, log-scale anomaly rule, fire-event labels, template-based season briefing (no LLM text at runtime) | Specified each change; reviewed in browser |
| App restructure: two tabs, map | Claude Code (Claude Fable 5.2) | Split the app into a plain-language planner tab and a method tab; added the folium map with rectangle drawing | Specified structure and audience; reviewed in browser |
| forecast.py (ML forecast test) | Claude Code (Claude Fable 5.2) | Wrote and ran a gradient-boosting forecast test against the seasonal baseline (scikit-learn, trained on our aggregates; no LLM in the predictions). Result: baseline as good or better, so no forecast is shown | Specified features, test design and the decision rule |
| Greece October check | Claude Code (Claude Fable 5.2) | Compared October with July–August detections on fire power, night share, daily counts and spatial spread; added a data-driven small-fire sentence to the planner tab | Asked the question; set the evidence bar |
| "Where it's unusual" map | Claude Code (Claude Fable 5.2) | Fixed the map bug; added comparison views, the without-correction toggle and a permutation test for season shifts; checked in a headless browser | Specified views and wording; reviewed in browser |
| Fire profile download | Claude Code (Claude Fable 5.2) | Wrote the one-page HTML profile generator (fixed templates filled from computed numbers; no LLM text at runtime) | Specified contents |
| USA and Europe | Claude Code (Claude Fable 5.2) | Ran the pipeline and validation for USA and Europe as a separate dataset, added dataset switching, greyed out poorly validated tiles, 0.25° display grid | Downloaded the data; set the quality rules; reviewed validation before publishing |
| Map styling and boundaries | Claude Code (Claude Fable 5.2) | Simplified Natural Earth outlines, drew only unusual squares on a faded basemap; checked in a headless browser | Specified the look |
| Season-year selector and rename | Claude Code (Claude Fable 5.2) | Added the season-year selector (chosen year against all other years), renamed the app to FireSeason; checked Greece 2007, California 2020 and USA | Specified the changes |
| Sidebar controls | Claude Code (Claude Fable 5.2) | Moved the season-year selector and years slider into the sidebar, each shown only on its own tab | Asked for the placement |
| Reliability line and fewer caveats | Claude Code (Claude Fable 5.2) | Added a good/fair/low reliability line from the validation error, shortened tab 1 caveats, moved full wording to tab 2 | Specified thresholds and what to move |
| Slides, pitch script and project summary | Claude (claude.ai, Claude Opus 5.5) | Drafted the 7-slide deck, the pitch script and the NASA project summary | Edited text, reordered slides, chose what to present |
