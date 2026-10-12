# Validation studies

Everything that tests the tool lives here. Run scripts from the repository root, for example
`python validation/05_validate_hybrid_coverage.py`. "Model calls" marks scripts that make real API calls
(small cost, noted in each script's docstring). Scripts 01 to 11 test design choices on GSS data only. Scripts 12 and 13 are the only ones that use a
different survey (a real sample of working adults, as a test set), and neither feeds anything back into the tool.

| Script | What it tests | Model calls | Output |
|---|---|---|---|
| `01_validate_threshold.py` | Each GSS variable treated as a novel item: calibration error against match similarity | no | `results/validation_results.csv` |
| `02_validate_combined_rule.py` | The combined rule (similarity, real n, years fielded) against similarity alone | no | `results/threshold_grid_search.csv` |
| `03_confidence_score_model.py` | A continuous confidence score from regression (R-squared = .044; not deployed) | no | printed results |
| `04_validate_ensemble.py` | Blending the top matches against the single best match (mean error 0.109 against 0.122) | no | `results/ensemble_validation_results.csv` |
| `05_validate_hybrid_coverage.py` | The hybrid rule and its coverage: 33.2% of 6,064 held-out variables, against 23.7% for the single best match alone | no | `data/hybrid_validation_results.csv` (**required by `generate_dataset.py`**) |
| `06_validate_generic_shape.py` | The generic Likert fallback shape against a uniform assumption (error 0.149 against 0.212) | no | printed results |
| `07_test_batch_size_reliability.py` | Response-count mismatch by batch size, 5 to 30 items | yes | `results/batch_size_reliability_results.csv` |
| `08_test_batch_size_followup.py` | The same at 40 to 50 items (batch 10 reached 33% mismatch; 8 had none) | yes | `results/batch_size_reliability_followup_results.csv` |
| `09_validate_propensity_std_cutoff.py` | Spread cutoff of 0.06 from 60 items | yes | `results/propensity_std_validation_results.csv` |
| `10_validate_vs_naive_baseline_100.py`, `_300.py`, `_500.py`, `_1000.py` | Calibrated pipeline against raw model output on 40 held-out GSS items, at that many synthetic respondents | yes | `results/naive_baseline_comparison_results_N<N>.csv` |
| `11_validate_min_real_n.py` | Is 200 a sensible minimum number of real respondents for an anchor? Measures the sampling noise of an anchor at 50 to 500 respondents and expresses it as a share of the benefit of calibration (about 17% at 200 respondents, 23% at 100, 35% at 50; no sharp elbow). Needs the raw GSS `.sav` | no | `results/min_real_n_validation_results.csv` |
| `12_validate_against_real_data.py` | Ground truth: 153 items from 34 organizational scales against 398 real working adults, each gap read against a noise floor | yes (to generate; analysis is free) | `results/real_data_test/` |
| `13_test_candidate_fixes.py` | Exploratory: would simple fixes close the gaps that 12 found (rating-based binning, flipping the output, removing the general agreement tendency)? In-sample on 62 items, so a first look and not a validation | no | `results/candidate_fixes_results.csv` |

Folders:

- `reference_data/` holds aggregates of the real survey used as the test set in 12 and 13, plus the 62-item
  generated run that 13 analyzes. See its README.
- `results/` holds the result files above. Scripts 01 to 06 and 13 are free to rerun and reproduce them; so is 11 if
  you have the raw GSS file. Scripts 07 to 10 and 12 call the model, so rerunning them costs API calls and their saved
  results are included instead (12 can be re-analyzed for free from its saved batches with `--analyze`).
