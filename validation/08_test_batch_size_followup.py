"""
Follow-up to 07_test_batch_size_reliability.py: confirm batch_size=10
stays reliable at higher item counts (40, 50 -- near the soft cap), and
check a narrow band around 10 in case slightly higher still works at
these item counts specifically. Uses more repeats (6 instead of 4) for
more statistical confidence, given the previous test showed some noisy/
inconsistent results at borderline sizes.

NOTE: makes real API calls (small but non-zero cost -- see estimate below).
"""

import importlib.util
import itertools
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("gen", "generate_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

ITEM_COUNTS_TO_TEST = [40, 50]
BATCH_SIZES_TO_TEST = [8, 10, 12, 15, 20]
REPEATS_PER_COMBO = 6

DEMO_ITEM_POOL = [
    "I feel valued by my organization", "My manager communicates clearly",
    "I often feel stressed at work", "I am proud of where I work",
    "I trust my coworkers", "I have opportunities to grow in my role",
    "My work is meaningful to me", "I feel supported by my team",
    "I am satisfied with my compensation", "I would recommend this workplace",
    "I have good work-life balance", "I feel safe raising concerns at work",
    "My contributions are recognized", "I understand my role's expectations",
    "I am motivated to do my best work", "I feel included at work",
    "My workplace treats people fairly", "I have the resources I need to do my job",
    "I look forward to coming to work", "I feel respected by my colleagues",
    "My job allows me to use my strengths", "I receive helpful feedback",
    "I am confident in my job security", "My workplace values diversity",
    "I feel a sense of accomplishment", "I can be myself at work",
    "My manager supports my development", "I am proud of my team's work",
    "I feel connected to my coworkers", "I believe in my organization's mission",
    "I have clear career growth opportunities", "I feel my work is challenging",
    "I am given autonomy in my role", "I feel my ideas are heard",
    "I have a good relationship with my manager", "I feel loyal to my organization",
    "My workload is manageable", "I trust leadership's decisions",
    "I feel psychologically safe at work", "I am proud to tell others where I work",
    "I feel my skills are well utilized", "I have access to necessary training",
    "I feel my pay is fair for my role", "My team collaborates well",
    "I understand how my role fits the bigger picture", "I feel encouraged to innovate",
    "My workplace supports my wellbeing", "I would stay at this job long-term",
    "I feel my manager cares about me as a person", "I am recognized for good work",
]

def make_dummy_personas(n):
    rng = np.random.default_rng(0)
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

def run_test():
    results = []
    combos = list(itertools.product(ITEM_COUNTS_TO_TEST, BATCH_SIZES_TO_TEST))
    print(f"Testing {len(combos)} combinations x {REPEATS_PER_COMBO} repeats "
          f"= {len(combos) * REPEATS_PER_COMBO} total API calls\n")

    for n_items, batch_size in combos:
        items_info = [{"item_text": t} for t in DEMO_ITEM_POOL[:n_items]]
        mismatches = 0
        diffs = []
        for trial in range(REPEATS_PER_COMBO):
            personas = make_dummy_personas(batch_size)
            try:
                result = gen.call_llm_batch(personas, items_info)
                actual_count = len(result)
            except Exception as e:
                print(f"  ERROR on items={n_items}, batch={batch_size}, trial={trial}: {e}")
                actual_count = -1
            diff = actual_count - batch_size
            diffs.append(diff)
            if diff != 0:
                mismatches += 1
            print(f"  items={n_items:3d} batch={batch_size:3d} trial={trial+1}: "
                  f"requested={batch_size}, got={actual_count} "
                  f"{'MISMATCH' if diff != 0 else 'ok'}")

        results.append({
            "n_items": n_items,
            "batch_size": batch_size,
            "product": n_items * batch_size,
            "n_trials": REPEATS_PER_COMBO,
            "n_mismatches": mismatches,
            "mismatch_rate": mismatches / REPEATS_PER_COMBO,
            "avg_diff": np.mean(diffs),
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_test()
    df.to_csv("data/batch_size_reliability_followup_results.csv", index=False)
    print("\nSaved -> batch_size_reliability_followup_results.csv\n")
    print(df.to_string(index=False))
