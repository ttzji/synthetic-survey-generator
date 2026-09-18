"""
Generate a representative U.S. adult sample (N=300) with responses to the
Calling, Work Meaning, and Moral Duty subscales.
"""

from importlib import import_module
import importlib.util

spec = importlib.util.spec_from_file_location("gen", "generate_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

items = [
    # Calling
    {"text": "The work I do feels like my calling in life", "type": "likert"},
    {"text": "It sometimes feels like I was destined to do the work I do", "type": "likert"},
    {"text": "The work I do feels like my niche in life", "type": "likert"},
    {"text": "I am definitely the sort of person who fits in my line of work", "type": "likert"},
    {"text": "My passion for the work I do goes back to my childhood", "type": "likert"},
    {"text": "I was meant to do the work I do", "type": "likert"},
    # Work Meaning
    {"text": "The work that I do is important", "type": "likert"},
    {"text": "I have a meaningful job", "type": "likert"},
    {"text": "The work that I do makes the world a better place", "type": "likert"},
    {"text": "What I do at work makes a difference in the world", "type": "likert"},
    {"text": "The work that I do is meaningful", "type": "likert"},
    # Moral Duty
    {"text": "I have a moral obligation to give customers the best possible support", "type": "likert"},
    {"text": "If I did not give customers the best possible support, I would feel like I was breaking a solemn oath", "type": "likert"},
    {"text": "I consider it my sacred duty to do all I can for our customers", "type": "likert"},
    {"text": "Supporting our customers is like a sacred trust to me", "type": "likert"},
]

df, coverage = gen.generate_dataset(
    items=items,
    n=500,
    seed=42,
    mode="representative",   # no demographic choices -- full U.S. adult representative sample
    likert_scale=7,
)

df.to_csv("calling_workmeaning_moralduty_dataset.csv", index=False)
coverage.to_csv("calling_workmeaning_moralduty_coverage.csv", index=False)
print("\nSaved calling_workmeaning_moralduty_dataset.csv and _coverage.csv")
print(f"\nConfidence tier breakdown:")
print(coverage["confidence_tier"].value_counts())
