# Block-aware OCR preprocessing

This pipeline treats each invoice as an Excel-style grid with visible and hidden
cell borders. Python first extracts candidate boundaries from PDF vector rules,
rectangles, text-box alignment, and whitespace. A strict vision response may
select those boundary IDs, but it cannot create coordinates:

1. horizontal bands, ordered top to bottom;
2. vertical blocks inside each band, ordered left to right.

It then rejects unknown boundary IDs or overlaps, renders a review overlay, crops every
block, writes exact coordinates and SHA-256 hashes to `manifest.json`, and opens
an animated browser review page. Red lines are horizontal boundaries; blue lines
are vertical boundaries. Pages advance every three seconds by default.

If a model response contains overlapping or invalid geometry, the pipeline makes
one correction request containing the validator error. Both raw attempts are
saved as `page-NNN-layout-attempts.json`; a second failure stops processing.

The model is only a grid-structure selector. Coordinate extraction, cropping,
naming, hashing, validation, and browser rendering are deterministic. Missing
visible borders may be interpreted as hidden gridlines or merged cells, but only
using candidates already supported by document geometry. The source PDF is never
overwritten, and this stage does not perform OCR.

## Run one PDF

```bash
cd /Users/dc/ocr-mini-bench
source .venv/bin/activate
export OPENAI_API_KEY="..."
python blocking/block_pipeline.py \
  --input bench_documents/Logistics/bill_of_lading_alt_1.pdf
```

Use `--no-browser` for unattended jobs or `--delay-seconds 5` to change the
review speed.

## Run the complete benchmark corpus

```bash
python blocking/block_pipeline.py \
  --input-dir bench_documents \
  --output-dir blocking/outputs
```

This makes one vision request per PDF page, so run the single-file command first
to validate the chosen model and expected cost.

## Output contract

```text
blocking/outputs/
  index.html
  batch-summary.json
  <pdf-stem>/
    manifest.json
    page-001-geometry.json
    pages/page-001.png
    overlays/page-001-blocks.png
    blocks/page-001-h01-v01.png
```

Every block record includes normalized coordinates, exact pixel coordinates,
the padded crop coordinates, its parent band, its source page, and a checksum.

## Test geometry without calling a model

```bash
pytest -q blocking/test_block_pipeline.py
```
