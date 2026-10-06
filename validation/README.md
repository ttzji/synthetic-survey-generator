# Validation studies

Each script tests one design choice. Run scripts from the repository root: most write to a relative
`data/` path. "Model calls" marks scripts that make real API calls (small cost, noted in each
script's docstring).

| Script | What it tests | Model calls | Output |
|---|---|---|---|
| `01_validate_threshold.py` | Each GSS variable treated as a novel item: calibration error against match similarity | no | `data/validation_results.csv` |
| `02_validate_combined_rule.py` | The combined rule (similarity, real n, years fielded) against similarity alone | no | `data/threshold_grid_search.csv` |
| `03_confidence_score_model.py` | A continuous confidence score from regression (R-squared = .047; not deployed) | no | printed results |
| `04_validate_ensemble.py` | Blending the top matches against the single best match | no | `data/ensemble_validation_results.csv` |
| `05_validate_hybrid_coverage.py` | The hybrid rule and its coverage (33.2% of 6,064 held-out variables) | no | `data/hybrid_validation_results.csv` (required by `generate_dataset.py`) |
| `06_validate_generic_shape.py` | The generic Likert fallback shape against a uniform assumption | no | printed results |
| `07_test_batch_size_reliability.py` | Response-count mismatch by batch size, 5 to 30 items | yes | `data/batch_size_reliability_results.csv` |
| `08_test_batch_size_followup.py` | The same at 40 to 50 items (batch 10 reached 33% mismatch; 8 had none) | yes | `data/batch_size_reliability_followup_results.csv` |
| `09_validate_propensity_std_cutoff.py` | Spread cutoff of 0.06 from 60 items | yes | `data/propensity_std_validation_results.csv` |
| `10_validate_vs_naive_baseline_100.py`, `_300.py`, `_500.py`, `_1000.py` | Calibrated pipeline against raw model output on 40 held-out GSS items, at that many synthetic respondents | yes | `data/naive_baseline_comparison_results_N<N>.csv` |
| `13_validate_item_correlation_inflation.py` | Generated against real item correlations (1,788 pairs); basis of the "divide by about 1.1" note | no | reads `data/item_correlation_comparison.csv` |

Ground-truth tests against real organizational survey data sit at the repository root: see
`full_validation.py` and the section "Validation against real organizational data" in the main README.
