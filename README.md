# Synthetic Survey Data Generator

Generates synthetic survey responses for researcher-supplied scale items, calibrated against real
U.S. General Social Survey (GSS) answer distributions where a strong enough match exists, with a
confidence label and an empirical error rate for every item.

**Intended use: pretesting** item wording, response formats, analysis scripts and floor/ceiling
effects. It is not a substitute for human data in inference.
**Scope:** the GSS is a U.S. survey, so all calibration reflects U.S. adults.
**License:** [PolyForm Noncommercial 1.0.0](LICENSE). Free for research, teaching and other
noncommercial use. For commercial use, contact Jack Zhang (jleaf1983@gmail.com).

## What the validation shows

| Finding | Evidence | Where |
|---|---|---|
| Calibration beats raw model output on GSS-covered items | Distribution error fell 43.2%, 40.8%, 39.6% and 38.2% in four runs (N = 100 to 1,000 synthetic respondents; all p <= .004) | `validation/10_validate_vs_naive_baseline_*.py`, `data/naive_baseline_comparison_results_N*.csv` |
| A confident real anchor exists for a third of novel items | 33.2% of the 6,064 held-out GSS variables pass the production gate | `validation/05_validate_hybrid_coverage.py`, `data/hybrid_validation_results.csv` |
| Generated correlations run too high | About +0.07 between scales and +0.16 within scales, against 398 real working adults. Dividing off-diagonal correlations by about 1.1 cut average error about 10% | `validation/13_validate_item_correlation_inflation.py`, `full_validation.py` |
| **Against real organizational data the tool does not reproduce item distributions** | 153 items, 34 scales: generated means were 1.70 points below real on a 7-point scale (noise floor 0.08); all items with no anchor received one identical distribution; confidence tiers did not predict accuracy | `full_validation.py`, `validation_output/` |

In short: use the tool to pretest, and read its confidence labels, but do not treat generated item
means, SDs or scale reliabilities as estimates for organizational scales. Every validation script
and the decision it supports is listed in [`validation/README.md`](validation/README.md).

## How it works

1. **Demographic resampling** draws synthetic respondents from real GSS respondents' joint
   demographic profiles (age, sex, race, marital status, employment, religion), either
   representatively (weighted by real survey weight) or filtered/targeted to user-specified
   proportions.
2. **Item retrieval** embeds each user-supplied item and finds the nearest matching real GSS
   variable(s) among 6,064 indexed variables, using a hybrid strategy: a single best match at high
   similarity (>= .75), a quality-filtered blend of several matches at moderate similarity (.50 to
   .75), or no real anchor below that.
3. **Generation** prompts an LLM (gpt-4.1-mini) for each synthetic persona's endorsement propensity
   for every item, 8 personas per call.
4. **Calibration** maps each propensity's percentile rank within the generated batch onto the real
   (or, with no strong match, a generic Likert) response distribution, so the marginal distribution
   matches that anchor by construction.

Every item is labeled with a confidence tier (`high`, `moderate` or `none`) and an empirical expected
error rate from the held-out validation in `validation/`.

## Repository structure

```
generate_dataset.py    Core pipeline (retrieval, calibration, generation)
app.py                 Streamlit web interface
requirements.txt
LICENSE                PolyForm Noncommercial 1.0.0
data/                  Derived GSS data and validation results (the raw .sav is NOT included)
pipeline/              One-time scripts that build data/ from the raw GSS .sav file
validation/            Studies supporting each design choice (see validation/README.md)
full_validation.py     Ground-truth test against real organizational survey data
validation_output/     design.json (fixed batches) and the saved results of that test
validation_common.py   Helper for the random-draw scripts below
test_3scale_draws.py, test_30item_draws.py
                       Random-draw replications of the correlation findings
test_bunderson_pathanalysis.py
                       Comparison with Bunderson & Thompson (2009)
.github/               Keeps the hosted demo awake (workflow + script)
```

Run every script from the repository root.

## Quick start (use the tool)

The derived data files are already in `data/`, so you do not need the raw GSS file.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
streamlit run app.py
```

The web interface asks for an access code. Locally, create `.streamlit/secrets.toml` (never commit
it; `.gitignore` excludes it):

```toml
OPENAI_API_KEY = "your-key-here"
ACCESS_CODE = "your-chosen-code"
```

On Streamlit Community Cloud, set both under the app's Settings -> Secrets. To call the pipeline
from code, import `generate_dataset` and call `generate_dataset(items, n, seed, ...)`, which returns
`(dataset, coverage_report)`.

## Rebuilding `data/` from the raw GSS file (optional)

1. Download the GSS cumulative data file (`.sav`) from NORC (https://gss.norc.org/get-the-data) and
   place it at `data/gss7224_r3a.sav` (or edit `SAV_PATH` in the pipeline scripts). It is not
   redistributed here.
2. Run the scripts in `pipeline/` in numbered order. Each writes into `data/`:

   | Script | Produces |
   |---|---|
   | `01_check_sav_file.py` | (diagnostic only) |
   | `02_build_full_catalog.py` | `data/gss_variable_catalog.csv` |
   | `03_build_weighted_stats.py` | `data/gss_master_variable_stats.json` |
   | `04_build_embeddings.py` | `data/gss_variable_embeddings.npy`, `data/gss_variable_index.csv` |
   | `05_extract_demographics_pool.py` | `data/gss_demographics_pool.csv`, `data/gss_demographics_value_labels.json` |
   | `06_build_generic_shape.py` | `data/generic_likert_shape.npy` |

3. Run `validation/05_validate_hybrid_coverage.py` to produce `data/hybrid_validation_results.csv`,
   which `generate_dataset.py` reads for its expected-error reporting.

**Files `generate_dataset.py` and `app.py` need in `data/`:** `gss_variable_embeddings.npy`,
`gss_variable_index.csv`, `gss_master_variable_stats.json`, `gss_demographics_pool.csv`,
`gss_demographics_value_labels.json`, `generic_likert_shape.npy`, `hybrid_validation_results.csv`.

## Validation against real organizational data

`full_validation.py` generates all 153 items from 34 organizational scales, in 19 batches that keep
each scale whole, and compares the output with 398 real working adults (Prolific) on mean, SD,
skewness, tail shares, distribution distance, and item- and scale-level correlations. Every gap is
read against a noise floor: the gap expected between two independent surveys of 398 people. The
real-data inputs are derived aggregates only (no individual responses):
`data/prolific_item_value_proportions.json`, `data/prolific_item_correlation_matrix.csv`,
`data/prolific_scale_correlation_matrix.csv` and `data/items_metadata_FINAL_34.json`.

```bash
python full_validation.py             # generate (resumable; 19 batches x about 50 model calls), then analyze
python full_validation.py --analyze   # re-analyze the saved output, no model calls
python full_validation.py --design    # show the batches and call count, call nothing
```

Results are written to `validation_output/` (`summary_for_paper.csv`, `item_level.csv`,
`pair_level.csv`, `scale_pair_level.csv`; `design.json` fixes the batches). Run it only with the
GSS-only `generate_dataset.py`: anchoring on the test sample would make the test circular.

## Understanding the output

Every generated dataset ships with a **coverage report** (per-item retrieval match, confidence tier,
expected error) and, in the web interface, a **data dictionary** for every column, including the
value labels for demographic codes (for example what `sex=1` means). Demographic labels come from a
hardcoded, verified `DEMOGRAPHIC_CODEBOOK` in `generate_dataset.py`, not from
`gss_demographics_value_labels.json`: GSS attaches about 12 standardized missing-value codes to
nearly every variable, and code that read labels directly from that file was tripped up by that
noise and a key-format mismatch. Call `build_data_dictionary()` to get it programmatically.

## Known limitations

- **Coverage.** A real anchor exists for about a third of novel items. Organizational constructs
  usually fall back to a generic Likert shape that is identical for every unanchored item.
- **Distributions.** On organizational items, generated answers pile up at the low end while real
  answers lean toward agreement (see the table above). Flipping the output when the model judges an
  item to be mostly agreed with roughly halved the distance on 62 items in a post-hoc check; this is
  in-sample and not part of the tool.
- **Correlations.** Generated correlations run too high, most within scales, so scale reliability
  (Cronbach's alpha) will be overstated; alpha itself has not been tested. Individual generated
  correlations are imprecise (mean absolute error about 0.23, about four times the noise floor).
- **Confidence tiers** were validated within the GSS. On organizational items they did not predict
  accuracy.
- **Population and model.** One real comparison sample (working adults), one language model
  (a proprietary API), and U.S.-only anchors. Accuracy for demographic subgroups has not been tested.
- Demographic resampling treats user-specified gender and race targets as independent of each other.
