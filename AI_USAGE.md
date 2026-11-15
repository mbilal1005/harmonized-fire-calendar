# AI usage

| Step | Tool / model | What it did | Human role |
|---|---|---|---|
| Project spec | Team | Wrote the task spec, method and validation design | Authored |
| Data extraction + inspection | Claude Code (Claude Opus 5.5) | Unzipped FIRMS downloads, inspected columns, date ranges, value distributions | Reviewed output |
| pipeline.py step 1 (load/filter/report) | Claude Code (Claude Opus 5.5) | Wrote loader: sensor detection, type==0 filter, confidence filters, archive/NRT dedup, sanity-check report | Reviewed numbers at STOP |
