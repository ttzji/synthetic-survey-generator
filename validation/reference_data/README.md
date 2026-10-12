# Reference data (test set only)

These files describe a separate real survey: 398 working adults recruited on Prolific, who answered 153 items from
34 organizational scales. They are used only by `12_validate_against_real_data.py` and `13_test_candidate_fixes.py`
to check the tool's output. **The tool never reads them and never anchors on them.** Only aggregates are
included; the individual responses are not shared.

| File | Contents |
|---|---|
| `items_metadata_FINAL_34.json` | The 34 scales and their items (wording, response format, reverse-worded flag) |
| `prolific_item_value_proportions.json` | For each item, the share of the 398 respondents choosing each answer |
| `prolific_item_correlation_matrix.csv` | Correlations between all 153 items |
| `prolific_scale_correlation_matrix.csv` | Correlations between the 34 scale averages |
| `baseline_run_62items.csv` | A generated dataset (400 synthetic respondents, 62 of the 153 items) with the model's raw agreement ratings, analyzed by script 13 |
| `baseline_run_62items_metadata.json` | Item order, scale and reverse-worded flag for that run |
