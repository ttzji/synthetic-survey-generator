# Synthetic Survey Data Generator

Generates synthetic survey responses for researcher-supplied scale items, calibrated against real
U.S. General Social Survey (GSS) answer distributions where a strong enough match exists, with a
confidence label and an empirical error rate for every item.

**Intended use: pretesting** organizational research ideas. It is not a substitute for human data in inference.
**Scope:** the GSS is a U.S. survey, so all calibration reflects U.S. adults.
**License:** [PolyForm Noncommercial 1.0.0](LICENSE). Free for research, teaching and other
noncommercial use. For commercial use, contact Jack Zhang (jleaf1983@gmail.com).

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

Every item is labeled with a confidence label (`high`, `moderate` or `low`) and an empirical expected
error rate from the held-out validation in `validation/`.

## Repository structure

```
generate_dataset.py     Core pipeline (retrieval, calibration, generation)
app.py                  Streamlit web interface
requirements.txt        Libraries for everything in this repository
CITATION.cff            How to cite this software
LICENSE                 PolyForm Noncommercial 1.0.0
data/                   What the tool reads: derived GSS files only (the raw .sav is NOT included)
pipeline/               One-time scripts that build data/ from the raw GSS .sav file
validation/             Everything that tests the tool (see validation/README.md)
  01-10                 Design-choice studies, built and tested on GSS data only
  11                    Minimum real sample size for an anchor
  12                    Ground-truth test against a real organizational survey
  13                    Scale reliability (Cronbach's alpha), generated against real
  reference_data/       The real survey's aggregates, used only as a test set (never as an anchor)
  results/              Result files of the scripts above; results/real_data_test/ holds script 12's
.github/                Scheduled workflow that pings the hosted demo so it stays awake (not needed to use the tool)
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
   which `generate_dataset.py` reads for its expected-error reporting. (Scripts 01 to 04 and 06 in
   `validation/` are free and offline too; they write to `validation/results/`.)

**Files `generate_dataset.py` and `app.py` need in `data/`:** `gss_variable_embeddings.npy`,
`gss_variable_index.csv`, `gss_master_variable_stats.json`, `gss_demographics_pool.csv`,
`gss_demographics_value_labels.json`, `generic_likert_shape.npy`, `hybrid_validation_results.csv`.

## Validation against real organizational data

The tool is built only on the GSS. `validation/12_validate_against_real_data.py` tests it against a separate
real survey of 398 working adults (Prolific): it generates all 153 items from 34 organizational scales, in 19
batches that keep each scale whole, and compares the output with the real answers on mean, SD, skewness, tail
shares, distribution distance, and item- and scale-level correlations. Every gap is read against a noise
floor: the gap expected between two independent surveys of 398 people. The real survey is never an anchor:
`generate_dataset.py` has no code that reads it, and anchoring on the test sample would make the test
circular. Its derived aggregates (no individual responses) are in `validation/reference_data/`.

```bash
python validation/12_validate_against_real_data.py             # generate (resumable; 19 batches x about 50 model calls), then analyze
python validation/12_validate_against_real_data.py --analyze   # re-analyze the saved output, no model calls
python validation/12_validate_against_real_data.py --design    # show the batches and call count, call nothing
```

Results are written to `validation/results/real_data_test/` (`summary_for_paper.csv`, `item_level.csv`,
`pair_level.csv`, `scale_pair_level.csv`; `design.json` fixes the batches).

`validation/13_compare_alpha.py` compares Cronbach's alpha for the 34 scales: generated alpha averaged .83
against a real .73 and was higher for 29 scales, so synthetic data overstate scale reliability. It needs only
the saved output of script 12 and makes no model calls.

## Understanding the output

Every generated dataset ships with a **coverage report** (per-item retrieval match, confidence label,
expected error) and, in the web interface, a **data dictionary** for every column, including the
value labels for demographic codes (for example what `sex=1` means). Demographic labels come from a
verified `DEMOGRAPHIC_CODEBOOK` in `generate_dataset.py` and not from
`gss_demographics_value_labels.json`, because the GSS attaches about 12 standardized missing-value codes to
nearly every variable. Call `build_data_dictionary()` to get the dictionary programmatically.
