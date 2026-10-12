# Results

Result files written by the scripts in `validation/`. The file names follow the script numbers; the table in
`validation/README.md` lists which script writes which file.

`real_data_test/` holds the output of script 12:

- `design.json`: the fixed batch design (3 groupings of the 34 scales into 19 batches); it is reused on every run.
- `resp_p<partition>_b<batch>.csv`: the generated respondents for one batch (19 files).
- `cov_p<partition>_b<batch>.csv`: the tool's own coverage report for the same batch: retrieval match, tier and
  expected error for each item (19 files).
- `item_level.csv`, `pair_level.csv`, `scale_pair_level.csv`, `summary_for_paper.csv`: the analysis. Run
  `python validation/12_validate_against_real_data.py --analyze` to recompute them from the saved batches
  without any model call.
