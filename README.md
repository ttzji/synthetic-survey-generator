# Synthetic Survey Data Generator

Generates synthetic survey response data for arbitrary researcher-supplied
scale items, calibrated against real U.S. General Social Survey (GSS)
response distributions where a sufficiently strong match exists, with
transparent per-item confidence labeling and empirically-derived error
estimates.

**Scope:** GSS is a U.S.-only survey; all calibration reflects the U.S.
adult population.

## How it works

1. **Demographic resampling** draws synthetic respondents from real GSS
   respondents' joint demographic profiles (age, sex, race, marital
   status, employment, religion), either representatively (weighted by
   real survey weight) or filtered/targeted to user-specified proportions.
2. **Item retrieval** embeds each user-supplied item and finds the
   nearest matching real GSS variable(s) via semantic similarity, using a
   hybrid strategy: a single best match at high similarity, a
   quality-filtered blend of several matches at moderate similarity, or
   no real anchor at low similarity.
3. **Generation** prompts an LLM to produce a raw endorsement propensity
   for each item, jointly across all items for each synthetic persona in
   one call.
4. **Calibration** maps each propensity's percentile rank within the
   generated batch onto the real (or population-level, if no strong match
   was found) response distribution's shape, so the synthetic data's
   marginal distribution matches real survey data by construction.

Every generated item is labeled with a confidence tier (`high`,
`moderate`, or `none`) and an empirically-derived expected error rate
from the accompanying validation studies -- see `validation/`.

## Repository structure

```
generate_dataset.py   Core pipeline (retrieval, calibration, generation)
app.py                 Streamlit web interface
requirements.txt
data/                   Derived data files (see below) -- NOT included in
                         this repo; you generate them yourself via pipeline/
pipeline/               One-time setup scripts: build the files in data/
                         from a raw GSS .sav file
validation/             Empirical validation studies supporting the
                         pipeline's design choices (thresholds, batch
                         size, calibration methods)
examples/               Example end-to-end generation scripts
```

## Setup

1. **Create a `data/` folder** at the repo root (the pipeline scripts also
   create it automatically if it doesn't exist).

2. **Obtain the GSS cumulative data file** (`.sav` format) directly from
   NORC: https://gss.norc.org/get-the-data, and place it at
   `data/gss7224_r3a.sav` (or update `SAV_PATH` in the pipeline scripts to
   match your filename). This file is not included in this repository --
   too large for GitHub, and should not be redistributed.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key** as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
   For the web interface (`app.py`), also set an access code so the app
   can be shared as "link + code" without needing individual viewer
   emails. Locally, create `.streamlit/secrets.toml` (never commit this
   file -- already excluded by `.gitignore`):
   ```toml
   OPENAI_API_KEY = "your-key-here"
   ACCESS_CODE = "your-chosen-code"
   ```
   On Streamlit Community Cloud, set both under the deployed app's
   Settings -> Secrets instead.

5. **Run the pipeline scripts in `pipeline/`, in numbered order.** Each
   writes its output into `data/`:

   | Script | Produces |
   |---|---|
   | `01_check_sav_file.py` | (diagnostic only, no output file) |
   | `02_build_full_catalog.py` | `data/gss_variable_catalog.csv` |
   | `03_build_weighted_stats.py` | `data/gss_master_variable_stats.json` |
   | `04_build_embeddings.py` | `data/gss_variable_embeddings.npy`, `data/gss_variable_index.csv` |
   | `05_extract_demographics_pool.py` | `data/gss_demographics_pool.csv`, `data/gss_demographics_value_labels.json` |
   | `06_build_generic_shape.py` | `data/generic_likert_shape.npy` |

6. **Run `validation/05_validate_hybrid_coverage.py`**, which produces
   `data/hybrid_validation_results.csv` -- also required by
   `generate_dataset.py` for its expected-error reporting. The other
   scripts in `validation/` are optional (they reproduce the empirical
   findings behind the pipeline's design choices) and also write their
   output CSVs into `data/`. Most are free (reuse existing embeddings, no
   API calls); a few make small numbers of real API calls at trivial cost
   (noted in each script's docstring).

7. **Generate data** either via the example scripts in `examples/`, or
   the web interface:
   ```bash
   streamlit run app.py
   ```

### Complete list of files that end up in `data/`

**Required for `generate_dataset.py` / `app.py` to run:**
- `gss_variable_embeddings.npy`
- `gss_variable_index.csv`
- `gss_master_variable_stats.json`
- `gss_demographics_pool.csv`
- `gss_demographics_value_labels.json`
- `generic_likert_shape.npy`
- `hybrid_validation_results.csv`

**Intermediate pipeline artifact (not read at generation time, but needed to run the pipeline scripts in order):**
- `gss_variable_catalog.csv`

**Not included in this repo (obtain yourself; excluded by `.gitignore`):**
- `gss7224_r3a.sav` (or your equivalent GSS cumulative file)

**Optional, produced by running the validation studies (supports the paper's empirical claims, not required for the tool to run):**
- `validation_results.csv`
- `ensemble_validation_results.csv`
- `threshold_grid_search.csv`
- `propensity_std_validation_results.csv`
- `batch_size_reliability_results.csv`
- `batch_size_reliability_followup_results.csv`



## Understanding the output

Every generated dataset ships alongside a **coverage report** (per-item retrieval
match, confidence tier, expected error) and, in the web interface, a
**data dictionary** explaining every column -- including the actual value
labels for demographic codes (e.g. what `sex=1` or `race=2` means),
generated directly from the same file the pipeline uses internally, so it
can never drift out of sync with the real data. In the web app, this
appears as an expandable "What do these columns mean?" panel next to the
results, plus its own downloadable CSV so it travels with the dataset.
Programmatically, call `build_data_dictionary()` in
`generate_dataset.py` (no arguments needed -- demographic value labels
come from a hardcoded, verified `DEMOGRAPHIC_CODEBOOK`, not from
`gss_demographics_value_labels.json`. That file's raw content is actually
correct, but GSS attaches ~12 standardized missing-value codes to nearly
every variable, and code that displayed or looked up labels directly from
it was tripped up by that noise plus a key-format mismatch. Using a
verified hardcoded table sidesteps both issues cleanly).

## Known limitations

- Real-data calibration is only as broad as GSS's own topic coverage;
  occupation- or domain-specific constructs with no GSS analogue (see
  `examples/02_generate_zoo_scales.py` for a demonstration) fall back to
  population-level calibration, clearly labeled as such.
- Demographic resampling treats user-specified gender and race targets as
  independent of each other (a disclosed simplifying assumption).
- All validation is proxy-based (held-out real GSS items standing in for
  novel items); ground-truth validation against newly-collected human
  response data is planned future work.
