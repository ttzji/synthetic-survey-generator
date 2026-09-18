"""
Generate a representative U.S. adult sample (N=500), filtered to employed
respondents, with responses to seven scales: Neoclassical Calling,
Occupational Identification, Moral Duty, Work Meaningfulness, Occupational
Importance, Willingness to Sacrifice, and Perceived Organizational Duty.

Item sources (cite appropriately in any resulting work):
  - Neoclassical Calling, Moral Duty, Occupational Importance,
    Willingness to Sacrifice, Perceived Organizational Duty: developed for
    the original study based on field data (+ AAZK input for some).
  - Occupational Identification: Mael & Ashforth (1992).
  - Work Meaningfulness: Spreitzer (1995); Wrzesniewski et al. (1997);
    Pratt & Ashforth (2003).
"""

import importlib.util

spec = importlib.util.spec_from_file_location("gen", "generate_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

items = [
    # Neoclassical Calling (1-7, very strongly disagree to very strongly agree)
    {"text": "Working with animals feels like my calling in life", "type": "likert"},
    {"text": "It sometimes feels like I was destined to work with animals", "type": "likert"},
    {"text": "Working with animals feels like my niche in life", "type": "likert"},
    {"text": "I am definitely an animal person", "type": "likert"},
    {"text": "My passion for animals goes back to my childhood", "type": "likert"},
    {"text": "I was meant to work with animals", "type": "likert"},

    # Occupational Identification (Mael & Ashforth, 1992)
    {"text": "When someone criticizes the animal keeping profession, it feels like a personal insult", "type": "likert"},
    {"text": "I am very interested in what others think of the animal keeping profession", "type": "likert"},
    {"text": "When I talk about the animal keeping profession, I usually say 'we' rather than 'they'", "type": "likert"},
    {"text": "The animal keeping profession's successes are my successes", "type": "likert"},
    {"text": "When someone praises the animal keeping profession, it feels like a personal compliment", "type": "likert"},

    # Moral Duty (1-7, not at all to a very great extent)
    {"text": "I have a moral obligation to give my animals the best possible care", "type": "likert"},
    {"text": "If I did not give my animals the best possible care, I would feel like I was breaking a solemn oath", "type": "likert"},
    {"text": "I consider it my sacred duty to do all I can for my animals", "type": "likert"},
    {"text": "Caring for my animals is like a sacred trust to me", "type": "likert"},

    # Work Meaningfulness (Spreitzer 1995; Wrzesniewski et al. 1997; Pratt & Ashforth 2003)
    {"text": "The work that I do is important", "type": "likert"},
    {"text": "I have a meaningful job", "type": "likert"},
    {"text": "The work that I do makes the world a better place", "type": "likert"},
    {"text": "What I do at work makes a difference in the world", "type": "likert"},
    {"text": "The work that I do is meaningful", "type": "likert"},

    # Occupational Importance (developed for original study, with AAZK input)
    {"text": "Zoos that breed endangered species play a critical role in the larger animal conservation effort", "type": "likert"},
    {"text": "Keeping animals in zoos is justified because zoos are working to prevent species extinction", "type": "likert"},
    {"text": "Captivity and captive breeding may be the only hope for many endangered species", "type": "likert"},
    {"text": "Educating the public by showing them captive animals may be the only way to change attitudes about animal conservation", "type": "likert"},
    {"text": "Keeping animals in zoos is justified because zoos educate the public about animal issues", "type": "likert"},
    {"text": "Efforts to promote animal conservation would be a lot harder if zoos weren't around to educate the public about animals", "type": "likert"},

    # Willingness to Sacrifice (1-7, not at all to a very great extent)
    # Shared stem merged into each item so retrieval has full intended meaning:
    {"text": "How willing would you be to give up your free non-work time, without pay, to care for a sick animal", "type": "likert"},
    {"text": "How willing would you be to give up your free non-work time, without pay, to provide enrichment activities for an animal", "type": "likert"},
    {"text": "How willing would you be to give up your free non-work time, without pay, to serve on a committee to improve animal care at your facility", "type": "likert"},

    # Perceived Organizational Duty (1-7, not at all to a very great extent)
    {"text": "I believe that this facility is morally obligated to give its animals the best possible care", "type": "likert"},
    {"text": "If this facility does not give its animals the best possible care, it would be like it is breaking a solemn oath", "type": "likert"},
    {"text": "I believe that this facility has a sacred duty to do all it can for its animals", "type": "likert"},
    {"text": "I believe that caring for animals is like a sacred trust for this facility", "type": "likert"},
]

print(f"Total items: {len(items)}")

df, coverage = gen.generate_dataset(
    items=items,
    n=500,
    seed=42,
    mode="custom",
    filters=["employed"],
    likert_scale=7,
)

df.to_csv("zoo_scales_dataset.csv", index=False)
coverage.to_csv("zoo_scales_coverage.csv", index=False)
print("\nSaved zoo_scales_dataset.csv and zoo_scales_coverage.csv")
print(f"\nConfidence tier breakdown:")
print(coverage["confidence_tier"].value_counts())
