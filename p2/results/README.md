# Part II results

This run uses the course's 160-question training panel and matches the four seeded draws
per question in the public benchmark reference files.

- `summary.csv` contains pass@1, pass@2, and pass@4 for all four model variants.
- `analysis.json` contains MATH-500 difficulty results and IFEval pass@4 transitions.
- `sft-training.json` and `opd-training.json` contain the logged training metrics.
- `timing.tsv` records when each stage finished. `elapsed-seconds.txt` records total wall time.
- `figures/` contains the report plots. Run `python3 plot.py .` with Matplotlib to regenerate them.
- `examples/` contains one training trajectory and short cases selected from matched evaluation draws where one adapter passed and the other failed.

Full per-draw results and adapter backups are available in the local archives.
