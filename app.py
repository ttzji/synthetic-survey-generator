"""
Streamlit front-end for the synthetic survey data generation tool.

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pushing this repository (including
the data/ folder -- see README.md for its full contents) to GitHub, then
connecting it at share.streamlit.io. Set OPENAI_API_KEY (and optionally
ACCESS_CODE, for link+code sharing without individual viewer emails) under
the app's "Secrets" in the Streamlit Cloud dashboard -- never hardcode
either here.
"""

import importlib.util
import streamlit as st

# --- Load the existing, validated pipeline ---
spec = importlib.util.spec_from_file_location("gen", "generate_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

st.set_page_config(page_title="Synthetic Survey Data Generator", layout="wide")

# --- Access code gate ---
# Share the app link + this code together (e.g. in your paper or with
# reviewers) rather than needing to know each viewer's email in advance.
# Set ACCESS_CODE under this app's Secrets in the Streamlit Cloud dashboard.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Synthetic Survey Data Generator")
    entered_code = st.text_input("Enter access code to continue", type="password")
    if st.button("Submit"):
        correct_code = st.secrets.get("ACCESS_CODE")
        if not correct_code:
            st.error("Access code not configured for this app. Contact the app owner.")
        elif entered_code == correct_code:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()

st.title("Synthetic Survey Data Generator")
st.caption(
    "Generates a synthetic dataset calibrated against real U.S. General Social "
    "Survey (GSS) response data where a good match exists, with transparent "
    "confidence labeling for every item. Scope: U.S. adult population only."
)

# =============================================================================
# INPUT FORM
# =============================================================================

st.header("1. Survey items")
st.write("Enter one item per line. Mark each item's type below.")

likert_text = st.text_area(
    "Likert-scale items (one per line)",
    height=150,
    placeholder="I feel my contributions are valued at this organization\nMy supervisor gives me clear feedback on my performance",
)
binary_text = st.text_area(
    "Binary (yes/no) items (one per line) -- optional",
    height=80,
    placeholder="I am currently a member of a labor union",
)

likert_scale = st.radio("Likert response scale", [5, 7], horizontal=True, index=1)

st.header("2. Sample")

col1, col2 = st.columns(2)
with col1:
    n = st.number_input("Sample size (N)", min_value=10, max_value=1000, value=100, step=10)
    st.caption("Larger N takes proportionally longer to generate.")
with col2:
    seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)

mode = st.radio(
    "Sample type",
    ["Representative (no choices -- full U.S. adult population)", "Custom"],
)
mode_key = "representative" if mode.startswith("Representative") else "custom"

filters, age_min, age_max, female_pct, white_pct = None, None, None, None, None
if mode_key == "custom":
    st.subheader("Custom sample options")
    filter_options = st.multiselect(
        "Restrict to respondents who are:",
        ["employed", "married", "religious"],
    )
    filters = filter_options if filter_options else None

    age_col1, age_col2 = st.columns(2)
    with age_col1:
        use_age = st.checkbox("Restrict age range")
        if use_age:
            age_min = st.number_input("Min age", min_value=18, max_value=89, value=25)
    with age_col2:
        if use_age:
            age_max = st.number_input("Max age", min_value=18, max_value=89, value=65)

    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        use_female = st.checkbox("Set target % female")
        if use_female:
            female_pct = st.slider("% Female", 0, 100, 50) / 100
    with demo_col2:
        use_white = st.checkbox("Set target % white")
        if use_white:
            white_pct = st.slider("% White", 0, 100, 60) / 100

st.divider()

# =============================================================================
# GENERATE
# =============================================================================

if st.button("Generate Dataset", type="primary"):
    likert_items = [t.strip() for t in likert_text.split("\n") if t.strip()]
    binary_items = [t.strip() for t in binary_text.split("\n") if t.strip()]
    total_items = len(likert_items) + len(binary_items)

    if not likert_items and not binary_items:
        st.error("Please enter at least one item.")
    elif total_items > 50:
        st.error(
            f"You entered {total_items} items, which exceeds the maximum of 50. "
            f"This limit reflects the range validated for response reliability "
            f"(see documentation) -- item counts beyond this have not been "
            f"tested. Please reduce your item list, or split it into multiple "
            f"separate generation requests."
        )
    else:
        items = (
            [{"text": t, "type": "likert"} for t in likert_items]
            + [{"text": t, "type": "binary"} for t in binary_items]
        )

        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("Retrieving item matches against real GSS data...")

        def update_progress(completed, total):
            progress_bar.progress(completed / total)
            status_text.text(f"Generating synthetic respondents... {completed}/{total}")

        try:
            df, coverage = gen.generate_dataset(
                items=items, n=int(n), seed=int(seed),
                mode=mode_key, filters=filters,
                age_min=age_min, age_max=age_max,
                female_pct=female_pct, white_pct=white_pct,
                likert_scale=int(likert_scale),
                confirm_large_request=True,  # UI already enforces max via number_input bounds
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0)
            status_text.text("Done.")
            st.session_state["result_df"] = df
            st.session_state["result_coverage"] = coverage
            st.success(f"Generated {len(df)} synthetic respondents.")
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Generation failed: {e}")

# =============================================================================
# RESULTS
# =============================================================================

if "result_df" in st.session_state:
    st.header("3. Results")

    st.subheader("Item confidence summary")
    coverage = st.session_state["result_coverage"]
    tier_counts = coverage["confidence_tier"].value_counts()
    st.write(
        f"**{tier_counts.get('high', 0)}** item(s) at high confidence (strong real-data match) · "
        f"**{tier_counts.get('moderate', 0)}** at moderate confidence (blended real-data match) · "
        f"**{tier_counts.get('none', 0)}** using population-level calibration (no strong real-data match found)"
    )
    st.dataframe(coverage[[
        "item", "method", "matched_variable", "similarity_top1",
        "confidence_tier", "expected_error", "expected_error_n_cases",
    ]], use_container_width=True)

    st.subheader("Generated dataset preview")
    st.dataframe(st.session_state["result_df"].head(20), use_container_width=True)

    with st.expander("What do these columns mean?"):
        data_dict = gen.build_data_dictionary()
        st.dataframe(data_dict, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download full dataset (CSV)",
            st.session_state["result_df"].to_csv(index=False),
            file_name="synthetic_dataset.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download coverage / confidence report (CSV)",
            coverage.to_csv(index=False),
            file_name="coverage_report.csv",
            mime="text/csv",
        )
    with col3:
        st.download_button(
            "Download data dictionary (CSV)",
            data_dict.to_csv(index=False),
            file_name="data_dictionary.csv",
            mime="text/csv",
        )
    st.caption("Generated data is held only in this browser session and is not "
               "stored on any server. Download it before closing this tab if "
               "you want to keep it.")
