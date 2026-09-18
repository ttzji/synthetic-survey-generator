"""
Empirically determine the MIN_PROPENSITY_STD cutoff, replacing the
originally-guessed 0.08 with a value grounded in the actual observed
distribution of std values the model produces across diverse items.

Method: generate raw propensities (the same way production does -- joint,
multi-item calls) for 4 diverse sets of ~15 items each, N=100 dummy
personas per set. Collect every item's raw_propensity_std, then look at
the natural distribution across all 60 items to pick a defensible
low-percentile cutoff instead of a guess.

NOTE: makes real API calls (small but non-zero cost -- see prior estimate: ~$0.05).
"""

import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("gen", "generate_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

N_PEOPLE = 100
BATCH_SIZE = 8

ITEM_SETS = {
    "work_attitudes": [
        "I feel valued by my organization", "My manager communicates clearly",
        "I often feel stressed at work", "I am proud of where I work",
        "I have opportunities to grow in my role", "My work is meaningful to me",
        "I feel supported by my team", "I am satisfied with my compensation",
        "I have good work-life balance", "My contributions are recognized",
        "I am motivated to do my best work", "I feel included at work",
        "I have the resources I need to do my job", "I feel respected by my colleagues",
        "I receive helpful feedback",
    ],
    "trust_and_social": [
        "Most people can be trusted", "I feel connected to my community",
        "I believe people are generally honest", "I trust my neighbors",
        "I feel a sense of belonging where I live", "People in my life support me",
        "I have close friends I can rely on", "I feel comfortable asking others for help",
        "I believe most people mean well", "I feel socially connected",
        "I trust institutions to act in my interest", "I feel a strong sense of community",
        "I am generally optimistic about people", "I feel accepted by those around me",
        "I believe cooperation is usually possible",
    ],
    "personal_wellbeing": [
        "I feel optimistic about my future", "I am generally a happy person",
        "I handle stress well", "I feel confident in my abilities",
        "I am satisfied with my life overall", "I feel a sense of purpose",
        "I am comfortable with who I am", "I feel emotionally stable",
        "I recover quickly from setbacks", "I feel in control of my life",
        "I am generally calm under pressure", "I feel good about my personal growth",
        "I am able to relax when I need to", "I feel physically healthy",
        "I am content with my daily life",
    ],
    "general_attitudes": [
        "I value spending time with family", "I am open to trying new experiences",
        "I believe hard work leads to success", "I enjoy learning new things",
        "I am adaptable to change", "I value honesty in relationships",
        "I take pride in doing things well", "I am curious about the world",
        "I value fairness in how people are treated", "I enjoy helping others",
        "I am patient with people who disagree with me", "I value personal responsibility",
        "I am comfortable making decisions", "I value keeping my commitments",
        "I enjoy solving problems",
    ],
}

def make_dummy_personas(n, seed):
    rng = np.random.default_rng(seed)
    genders = ["Male", "Female"]
    races = ["White", "Black", "Other"]
    maritals = ["Married", "Never married", "Divorced"]
    employments = ["Working full time", "Working part time"]
    religions = ["Protestant", "Catholic", "None"]
    return [
        {
            "age": int(rng.integers(22, 65)),
            "gender": rng.choice(genders),
            "race": rng.choice(races),
            "marital_status": rng.choice(maritals),
            "employment": rng.choice(employments),
            "religion": rng.choice(religions),
        }
        for _ in range(n)
    ]

def run_validation():
    results = []
    for set_name, item_texts in ITEM_SETS.items():
        print(f"\n=== Set: {set_name} ({len(item_texts)} items) ===")
        items_info = [{"item_text": t} for t in item_texts]
        n_items = len(items_info)

        personas = make_dummy_personas(N_PEOPLE, seed=hash(set_name) % 1000)
        all_propensities = []
        for start in range(0, N_PEOPLE, BATCH_SIZE):
            batch = personas[start:start + BATCH_SIZE]
            result = gen.call_llm_batch(batch, items_info)
            if len(result) != len(batch):
                print(f"  (batch mismatch: got {len(result)}, expected {len(batch)} -- skipping this batch)")
                continue
            all_propensities.extend(result)
        print(f"  Collected {len(all_propensities)} / {N_PEOPLE} responses")

        for i, item_text in enumerate(item_texts):
            col_key = f"item_{i+1}"
            raw = np.array([p[col_key] for p in all_propensities])
            std = raw.std()
            mean = raw.mean()
            results.append({
                "set": set_name, "item": item_text,
                "mean_propensity": mean, "std_propensity": std,
                "n_responses": len(raw),
            })
            print(f"    {item_text[:45]:45s} std={std:.4f} mean={mean:.3f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_validation()
    df.to_csv("data/propensity_std_validation_results.csv", index=False)
    print("\nSaved -> propensity_std_validation_results.csv\n")

    print("=== Distribution of std across all items ===")
    print(df["std_propensity"].describe())
    print()
    for pct in [5, 10, 15, 20, 25, 50]:
        val = np.percentile(df["std_propensity"], pct)
        print(f"  {pct}th percentile: {val:.4f}")

    print("\nRecommended cutoff (10th percentile of observed distribution):")
    print(f"  MIN_PROPENSITY_STD = {np.percentile(df['std_propensity'], 10):.4f}")
