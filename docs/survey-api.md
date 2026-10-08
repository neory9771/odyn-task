# Python Survey API

The API turns recorded survey answers into structured, citeable evidence.
It computes observed estimates and comparisons and records measurement or
identification limits. Its results do not establish population representativeness
or causation. Statistics delegate to NumPy, SciPy and statsmodels.

Implementation: [`survey_api/`](../src/odyn_sft/survey_api/).
Concept definitions: [glossary](glossary.md).

## A complete example

This artificial survey compares awareness in two age groups. All IDs and values
below belong to the example; use the supplied catalogue for a real study.

```python
import pandas as pd
from odyn_sft.survey_api import SurveyAPI

answers = pd.DataFrame({
    "age": ["18-34"] * 40 + ["55+"] * 40,
    "awareness": ["Yes"] * 30 + ["No"] * 10 + ["Yes"] * 12 + ["No"] * 28,
})
metadata = {
    "hash": "demo-awareness-v1",
    "study": {"label": "Artificial awareness survey", "weighted": False},
    "variables": [
        {"id": "age", "label": "Age band", "block_id": "demographics",
         "answer_type": "single", "values": ["18-34", "55+"],
         "ordered_values": ["18-34", "55+"], "flags": [],
         "denominator_rule": "observed_item_excluding_refusal"},
        {"id": "awareness", "label": "Brand awareness", "block_id": "brand",
         "answer_type": "single", "values": ["Yes", "No"],
         "ordered_values": None, "flags": [],
         "denominator_rule": "observed_item_excluding_refusal"},
    ],
}

api = SurveyAPI(answers, metadata)
groups = api.group("age", {"Younger": "18-34", "Older": "55+"})
with api.subclaim("c1", "Younger respondents report greater brand awareness"):
    estimate = api.proportion("awareness", "Yes", by=groups)
    api.compare(estimate, "Younger", "Older", expect=">")
with api.subclaim("c2", "Advertising caused the awareness difference"):
    api.not_measured("causal effect", "This survey does not identify advertising effects")
evidence = api.pack()
assert [row["value"] for row in estimate["estimates"]] == [0.75, 0.30]
```

One session holds one study's filters, estimates and evidence. Create a new
session for each program. Read `pack()` after all analyses so the multiple-test
adjustment includes every executed test. Metadata `hash` should identify the
frozen metadata; the example uses a descriptive string for convenience.

## Executing generated Python

`run_program` binds API functions directly, so generated code calls
`group(...)`, not `api.group(...)`. The program needs no imports for those names.

```python
from odyn_sft.survey_api import run_program

program = """
g = group('age', {'Younger': '18-34', 'Older': '55+'})
with subclaim('c1', 'Younger respondents report greater awareness'):
    e = proportion('awareness', 'Yes', by=g)
    compare(e, 'Younger', 'Older', expect='>')
"""
evidence = run_program(program, answers, metadata)
```

This runs ordinary Python in a fresh worker process. Default limits are
30 seconds wall time, 20 CPU seconds, 2,048 MB memory, 1 MB program source and
2 MB printed output. Override them with `limits=ProcessLimits(...)`.
An exception fails the whole program; partial evidence is not returned.
Printed text is discarded from the evidence channel.

The environment provides `np`/`numpy`, `pd`/`pandas`, `scipy`, `stats`,
`statsmodels`, `sm`, and copies named `data` and `metadata`. Numerical calculations
outside API calls do not automatically become recorded evidence.
`run_reference(program, answers, metadata)` executes trusted reference code in
process; `execute_program(program, api)` uses an existing trusted session.

Direct `SurveyAPI` methods use exact string IDs. The program runtime also
resolves integer variable and block aliases from the frozen catalogue. These
are separate ID spaces: `rank_options` requires a block ID. Labels are descriptive
and may repeat. Do not infer an ID from a label or reuse aliases from another study.

## Study inputs

Load respondent data with `pd.read_parquet(...)` and metadata with `json.load(...)`.
The wide DataFrame has one row per respondent, unique index/column names, and a
column for every metadata variable ID. Metadata contains:

- `hash`: string identifying the frozen metadata.
- `study`: disclosed study/design information.
- `variables`: cards with `id`, `label`, `block_id`, `answer_type`, `values`,
  `ordered_values`, `flags` and a nonempty `denominator_rule` description.

Answer types are `single`, `select_all`, `ordered_scale` and `rank`.
`ordered_values` is a sequence of stored categories, or `None`; ordered scales
require a sequence. Checkbox columns contain booleans or missing values.
The full runtime metadata schema differs from the compressed model-facing
catalogue. Its denominator descriptions do not define arbitrary new base rules.
See [`types.py`](../src/odyn_sft/survey_api/types.py) and
[`data.py`](../src/odyn_sft/survey_api/data.py) for validation details.

## Functions

For direct use, prefix these calls with `api.`. In generated programs the
runtime binds the following 21 functions. Use `by=`, `among=`, `expect=`,
`order=` and `top=` as keywords to avoid positional ambiguity.

| Function | Purpose and limits |
|---|---|
| `var(v)` | Validate an exact variable ID; using that ID directly is also valid. |
| `find_variables(query, k=10, section=None)` | Return locally matched IDs. Discovery does not resolve ambiguous meanings. |
| `values(v)` | Return stored categories, ordered categories first. |
| `describe(v)` | Return a JSON string containing the card and observed counts. |
| `eligible(v)` | Return a Filter for the observed item/block base, not verified routing. |
| `where(v, value_or_list)` | Return a Filter matching recorded categories within that base. |
| `group(v, mapping=None)` | Map labels to one variable's values or lists of values. Without a mapping, labels are `str(value)`. |
| `group({label: Filter})` | The alternative form of the same function; define disjoint groups using filters. |
| `subclaim(id, text)` | Context manager associating evidence with one proposition. Authoring convention: one block per proposition, without nesting. |
| `proportion(v, value=None, by=None, among=None)` | Return an Estimate for one stored value; select-all defaults to selection `True`. A list of values is not supported. |
| `distribution(v, by=None, among=None)` | Return `{stored_value: Estimate}`; excludes the exact refusal and does not support select-all items. |
| `top_box(v, k=2, by=None, among=None)` | Share in the last k documented ordered values. Not for rank items; check that this end matches the claim. |
| `mean_score(v, by=None, among=None)` | Mean position 1…K for an ordered scale; assumes equal spacing. Not for rank items. |
| `compare(est, a, b, expect=None)` | Compare two labels from one Estimate, or label a against numeric threshold b. Effect is a minus b. |
| `trend(est, order=None, expect=None)` | Trend across 3+ distinct group labels. Declare `order=[...]`; without it, estimate group order is used. Assumes equal group steps. |
| `association(x, y, among=None, expect=None)` | Relationship between two variable IDs, not Estimates, in their joint observed base. Nominal variables allow only `any` or `None`. |
| `adjusted_compare(est, a, b, controls, expect=None)` | Stratified comparison of proportions/top-box estimates, on the odds-ratio scale. Controls are an ID or nonempty list; outcome and rank items cannot be controls. |
| `rank_options(block, by=None, among=None, top=None)` | Rank checkbox option shares within groups; takes a select-all block ID and returns `{label: [ranking rows]}`. |
| `segment_sizes(groups, among=None)` | Count group membership before applying an outcome-specific base. |
| `not_measured(concept, reason)` | Record a missing measurement or identification limitation. |
| `proxy(concept, [IDs], rationale)` | Declare an indirect measure; estimate the supplied variables separately. |
| `caveat(text)` | Record a qualification of at most 200 characters. |

`pack()` is a direct session method, not a generated-program callable.
The runtime collects the final pack automatically.

## Grouping and comparison patterns

To compare two groups, create **one estimate containing both groups**, then
compare its labels. Passing a second Estimate to `compare` is invalid.

```python
# Calls below are inside run_program; IDs/values refer to the example above.
g = group('age', {'Younger': '18-34', 'Older': '55+'})
e = proportion('awareness', 'Yes', by=g)
compare(e, 'Younger', 'Older', expect='>')

# Majority within one subgroup: the threshold is a fraction, not 50.
young = proportion('awareness', 'Yes', among=where('age', '18-34'))
compare(young, 'All observed respondents', 0.5, expect='>')

# distribution returns a mapping: select its Estimate before comparing.
d = distribution('awareness', by=g)
compare(d['Yes'], 'Younger', 'Older', expect='>')
```

Without `by=`, the group label is `All observed respondents`, even when `among=`
restricts the population. With a mapping, group labels are exactly its keys.
Probability thresholds must be strictly between 0 and 1; mean-score thresholds
must be strictly between 1 and K.

| Function | Allowed `expect` |
|---|---|
| `compare`, `adjusted_compare` | `>`, `<`, `!=`, `None` |
| `trend` | `increasing`, `decreasing`, `any`, `None` |
| `association` | `positive`, `negative`, `any`, `None`; nominal variables only `any`/`None` |

Run the comparisons implied by the claim: extra tests increase the Holm family.

## Bases, missingness and uncertainty

- Single/rank items use observed answers excluding the exact `Prefer not to say`.
  Combined labels such as `Don't know / Prefer not to say` remain ordinary
  categories; ordered statistics exclude them if they are outside the sequence.
- Select-all items use respondents with any recorded option in that block.
  Inside this base, a blank option counts as unselected. An entirely blank block
  is outside the base; it does not establish that the respondent selected none.
- Top-box and mean-score bases additionally require a documented ordered answer.
  Estimates intersect the outcome base, group membership and `among` filter.
- Filters combine with `&`, `|`, `~`; parenthesise expressions such as `(f & g) | h`.
  Both AND and OR use the intersection of the filters' observed universes.
  OR therefore excludes a respondent outside either input universe. NOT selects
  the complement inside its own universe, not all other respondents.
- Bases below 10 suppress estimates; below 30 receive `small_n` flags.
  Retain flags, bases and unavailable results in interpretation.
- Adjusted comparisons drop strata below 10 or missing either comparison group.
  Results apply to retained strata, with exclusions recorded. Adjustment is not
  causal identification.

All estimates are unweighted. Intervals/tests assume the procedures' sampling
conditions; they do not correct unknown recruitment, clustering or measurement
bias. Proportions use Wilson intervals; contrasts use the appropriate SciPy or
statsmodels procedure, including sparse-table fallbacks and resampling where
implemented. Exact methods and assumptions are recorded in the evidence and
[`statistics.py`](../src/odyn_sft/survey_api/statistics.py).

## Returned evidence and failures

An EvidencePack contains `api_version`, `metadata_hash`, `study`, `results`,
`test_count`, `label_rule` and `label_holm_rule`. Results include subclaim records,
estimates, contrasts, rankings and notes, each with an ID such as `c1.r2`.
Estimate rows contain `group`, `n`, `numerator`, `value`, `ci`, `flags` and
`denominator_rule`; mean-score rows have no binomial numerator.
Cite these result IDs in SFT2, and retain units and the relevant base.

Contrast `label` describes unadjusted evidence relative to `expect`:
`consistent`, `inconsistent`, `no_clear_difference`, `not_tested` or `unavailable`.
`pack()` adds `p_holm` and `label_holm` across all executed nonsuppressed tests.
Unadjusted and adjusted labels may differ. Neither label is a verdict on the
truth of the whole claim; `no_clear_difference` does not prove equality.

`AnalysisError` carries a stable `code`, message and optional source location;
`as_dict()` exposes them. Common failures include unknown IDs/values, overlapping
groups, mismatched function/item types, invalid expectations, refusal selection,
wrong estimate/group labels and excessively long caveats. A valid analysis can
also return an `unavailable` contrast when its inference cannot be computed.
Use [inference](inference.md) and [judging](judging.md) for pipeline handling.
