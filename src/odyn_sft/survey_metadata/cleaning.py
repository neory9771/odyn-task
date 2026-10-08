"""Normalize stored labels consistently during preparation and execution."""

RENAMES = {"Pasives": "Passives", "Fariness": "Fairness", "Saftety": "Safety"}


def clean(value):
    return RENAMES.get(str(value).replace("’", "'"), str(value).replace("’", "'"))
