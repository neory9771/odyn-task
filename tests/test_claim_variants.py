"""Claim paraphrases keep meaning: signature parsing, style-card ownership, code checks, planted changes."""

import random

from odyn_sft.dataset_generation.claim_variants import paraphrase as cp
from odyn_sft.dataset_generation.claim_variants.planted import perturb
from odyn_sft.dataset_generation.claim_variants.signature import signature

VARIABLES = [
    {"id": "age", "label": "Age"},
    {"id": "toyota", "label": "Toyota"},
    {"id": "honda", "label": "Honda"},
]
PROGRAM = """g = group(var('age'), {'Under 45': ['18-24', '25-34'], '45+': ['45-54', '55-64']})
with subclaim('c1', "Respondents in Under 45 record selecting 'Toyota' more often than respondents in 45+."):
    e1 = proportion(var('toyota'), by=g)
    compare(e1, 'Under 45', '45+', expect='>')
"""
CLAIM = "Respondents in Under 45 record selecting 'Toyota' more often than respondents in 45+."


def test_signature_reads_direction_and_groups_from_the_program():
    sig = signature(PROGRAM, VARIABLES, CLAIM)
    (part,) = sig["parts"]
    assert part["comparison"] == {"type": "difference", "direction": "more often than"}
    assert set(part["groups"]) == {"Under 45", "45+"}
    assert not sig["causal_clause"]


def test_test_split_only_uses_test_style_cards():
    for i in range(50):
        assert cp.card_for(f"r{i}", "test") in cp.TEST_CARDS
        assert cp.card_for(f"r{i}", "train") in cp.TRAIN_CARDS


def test_code_checks_reject_new_numbers_and_copies():
    sig = signature(PROGRAM, VARIABLES, CLAIM)
    assert any("number" in e for e in cp.code_checks(CLAIM.replace("more often", "30% more often"), CLAIM, sig))
    assert any("identical" in e for e in cp.code_checks(CLAIM.upper(), CLAIM, sig))
    assert not cp.code_checks("Do younger people (under 45) pick Toyota more than the 45+ crowd?", CLAIM, sig)


def test_planted_changes_alter_exactly_one_meaning_element():
    sig = signature(PROGRAM, VARIABLES, CLAIM)
    rng = random.Random(0)
    assert "less often" in perturb("flip_direction", CLAIM, sig, VARIABLES, rng)
    swapped = perturb("swap_groups", CLAIM, sig, VARIABLES, rng)
    assert swapped.index("45+") < swapped.index("Under 45")
    assert "because" in perturb("add_causal", CLAIM, sig, VARIABLES, rng)
    assert "'Honda'" in perturb("swap_measure", CLAIM, sig, VARIABLES, rng) or "'Age'" in perturb(
        "swap_measure", CLAIM, sig, VARIABLES, random.Random(1))
