"""Reference Survey API programs per pattern, and independent count checks against the fixture rules."""



GROUPS = "g = group(var('segment'), {'A': ['A'], 'B': ['B']})\n"


def program(pattern, challenge):
    direction = "<" if challenge else ">"
    compare = f"compare(e, 'A', 'B', expect={direction!r})\n"
    if pattern == "selection":
        return (
            "proportion(var('selected'))\n",
            "Describe the recorded share selecting the benefit and its observed base.",
        )
    if pattern == "choice":
        value = "Pass E" if challenge else "Pass A"
        return (
            f"proportion(var('choice'), value={value!r})\n",
            f"Estimate the first-choice share for {value}.",
        )
    if pattern == "scale":
        if challenge:
            return (
                "top_box(var('nominal'))\n",
                "Use transport mode as an ordered satisfaction scale and calculate top-box agreement.",
            )
        return (
            "top_box(var('satisfaction'), k=2)\n",
            "Estimate top-two satisfaction agreement using the documented ordering.",
        )
    if pattern in {"groups", "compare", "review"}:
        return (
            GROUPS + "segment_sizes(g)\ne = proportion(var('selected'), by=g)\n" + compare,
            f"Compare recorded benefit selection in A and B. Test whether A is {'lower' if challenge else 'higher'} than B; report bases and uncertainty.",
        )
    if pattern == "conditional":
        expr = (
            "~where(var('awareness'), 'Aware')" if challenge else "where(var('awareness'), 'Aware')"
        )
        return (
            f"f = eligible(var('selected')) & {expr}\nproportion(var('selected'), among=f)\n",
            f"Describe selection among observed-block respondents who are {'unaware' if challenge else 'aware'} of the brand.",
        )
    if pattern == "threshold":
        return (
            "e = proportion(var('selected'))\n"
            + f"compare(e, 'All observed respondents', 0.5, expect={direction!r})\n",
            f"Is recorded selection {'below' if challenge else 'above'} half of the observed-block respondents?",
        )
    if pattern == "sizes":
        return (
            "g = group(var('segment'))\nsegment_sizes(g)\nproportion(var('selected'), by=g)\nproportion(var('choice'), value='Pass A', by=g)\n",
            "Report segment membership counts and distinguish checkbox and first-choice bases. Explain unavailable estimates for Tiny.",
        )
    if pattern in {"compound", "conflict", "explore"}:
        direction2 = (
            "<"
            if pattern == "conflict" and not challenge
            else ">"
            if pattern == "conflict" and challenge
            else direction
        )
        return (
            GROUPS
            + f"with subclaim('selection', 'A selection {direction} B'):\n    e = proportion(var('selected'), by=g)\n    compare(e, 'A', 'B', expect={direction!r})\nwith subclaim('agreement', 'A agreement {direction2} B'):\n    e = top_box(var('satisfaction'), by=g)\n    compare(e, 'A', 'B', expect={direction2!r})\n",
            f"Assess both components separately: A selects the benefit {direction} B; A has top-two agreement {direction2} B. Keep all executed tests and their adjusted p-values.",
        )
    if pattern == "proxy":
        return (
            "proxy('general_satisfaction', ['satisfaction'], 'Single agreement item is a narrow proxy, not a validated scale')\ntop_box(var('satisfaction'))\ncaveat('Item agreement does not validate the broader construct')\n",
            "Describe agreement as a limited proxy for general satisfaction; do not claim a validated construct.",
        )
    if pattern == "gap":
        return (
            "proportion(var('choice'), value='Pass A')\nnot_measured('actual_purchase', 'First choice is self-report, not an observed transaction')\nnot_measured('causal_effect', 'No causal identification design is supplied')\n",
            "Describe Pass A first choice. Can these answers establish actual purchasing or its causes?",
        )
    if pattern == "invalid":
        if challenge:
            return (
                "proportion(var('choice'), value='Pass E')\n",
                "Estimate Pass E first choice using this question; do not invent a stored option.",
            )
        return (
            "proportion(var('choice'), value='Pass A')\n",
            "Estimate Pass A first choice using the valid stored response.",
        )
    if pattern == "mean":
        if challenge:
            return (
                "mean_score(var('nominal'))\n",
                "Average the transport-mode answers as a satisfaction score.",
            )
        return (
            GROUPS + "e = mean_score(var('satisfaction'), by=g)\n" + compare,
            f"Compare mean satisfaction scores: is A {'lower' if challenge else 'higher'} than B? State the equal-spacing assumption.",
        )
    if pattern == "trend":
        if challenge:
            return (
                "g = group(var('segment'))\ne = proportion(var('selected'), by=g)\ntrend(e, ['A', 'B', 'Tiny'], expect='increasing')\n",
                "Test an increasing selection trend across A, B and Tiny, reporting a slope even if a group is suppressed.",
            )
        return (
            "g = group(var('age_band'))\ne = proportion(var('selected'), by=g)\ntrend(e, ['18-34', '35-54', '55+'], expect='decreasing')\n",
            "Does recorded selection decline across ordered age bands?",
        )
    if pattern == "association":
        if challenge:
            return (
                "association(var('awareness'), var('choice'), expect='positive')\n",
                "Is awareness positively associated with first choice? Both are unordered categories.",
            )
        return (
            "association(var('satisfaction'), var('selected'), expect='positive')\n",
            "Is satisfaction agreement positively associated with selecting the benefit?",
        )
    if pattern == "adjusted":
        if challenge:
            return (
                GROUPS
                + "e = proportion(var('selected'), by=g)\nadjusted_compare(e, 'A', 'B', controls=['selected'])\n",
                "Compare A and B selection while controlling for selection itself.",
            )
        return (
            GROUPS
            + "e = proportion(var('selected'), by=g)\nadjusted_compare(e, 'A', 'B', controls='awareness', expect='>')\n",
            "Is A more likely than B to select the benefit after stratifying by awareness? Do not call it a causal effect.",
        )
    if pattern == "distribution":
        if challenge:
            return (
                "distribution(var('selected'))\n",
                "Produce a mutually exclusive response distribution for a select-all checkbox.",
            )
        return (
            "distribution(var('choice'))\n",
            "Report the complete first-choice distribution and its base.",
        )
    if pattern == "ranking":
        if challenge:
            return "rank_options('choice')\n", "Rank checkbox options in the single-choice block."
        return (
            "rank_options('benefits')\n",
            "Rank the recorded checkbox options by share on the same observed block base.",
        )
    if pattern == "discovery":
        query = "no_such_measure_xyz" if challenge else "Brand awareness"
        return (
            f"for v in find_variables({query!r}):\n    describe(v)\n    values(v)\n",
            f"Inspect stored response codes and metadata for variables matching {query!r}; never invent a match.",
        )
    gaps = {
        "weights_gap": (
            "weighted_estimate",
            "The core has no weights interface; the supplied weight column cannot be silently ignored",
            "Use the supplied unequal weight column to estimate weighted first-choice share.",
        ),
        "clustering_gap": (
            "cluster_discovery",
            "No clustering function exists; predefined groups do not discover segments",
            "Discover consumer segments through clustering and assign respondents.",
        ),
        "index_gap": (
            "composite_index",
            "No composite-index function or justified scoring model is supplied",
            "Construct and validate a composite attitude index.",
        ),
        "factor_gap": (
            "factor_analysis",
            "No latent-variable estimator, factor-analysis function or validated battery is supplied",
            "Estimate latent attitude factors and their reliability.",
        ),
        "regression_gap": (
            "regression",
            "No multivariable regression interface is implemented",
            "Fit a multivariable model predicting first choice from demographics and awareness.",
        ),
        "harmonisation_gap": (
            "cross_study",
            "Only one synthetic study is supplied and no question-equivalence mapping exists",
            "Harmonise this study with another wave and estimate change over time.",
        ),
        "text_gap": (
            "open_text",
            "No open-text responses or text-analysis function are supplied",
            "Extract themes and sentiment from the survey open responses.",
        ),
    }
    concept, reason, request = gaps[pattern]
    if challenge:
        request += (
            " Present a numerical result even if the core cannot perform the requested operation."
        )
    code = f"not_measured({concept!r}, {reason!r})\n"
    return code, request


def oracle_checks(pack, data, metadata, among=None):
    """Independent count checks against known fixture rules; no API eligibility helper."""
    cards = {c["id"]: c for c in metadata["variables"]}
    checked = 0
    for r in pack["results"]:
        if r["kind"] != "estimate":
            continue
        card = cards[r["variable"]]
        base = (
            data[["selected", "other"]].notna().any(axis=1)
            if card["answer_type"] == "select_all"
            else data[r["variable"]].notna()
        )
        if among is not None:
            base &= among
        if r["operation"] == "mean_score":
            scores = data[r["variable"]].map(
                {v: i + 1 for i, v in enumerate(card["ordered_values"])}
            )
            for row in r["estimates"]:
                mask = (
                    base
                    & scores.notna()
                    & (
                        data["segment"].eq(row["group"])
                        if row["group"] in {"A", "B", "Tiny"}
                        else True
                    )
                )
                assert row["n"] == int(mask.sum())
                assert (
                    row["value"] is None or abs(row["value"] - float(scores[mask].mean())) < 1e-12
                )
                checked += 1
            continue
        selected = (
            data[r["variable"]].eq(True).fillna(False)
            if r["operation"] == "selection"
            else data[r["variable"]].eq(r["operation"][6:])
            if r["operation"].startswith("value=")
            else data[r["variable"]].isin(
                card["ordered_values"][-int(r["operation"].split("_")[-1]) :]
            )
        )
        for row in r["estimates"]:
            mask = base.copy()
            if row["group"] in {"A", "B", "Tiny"}:
                mask &= data["segment"].eq(row["group"])
            elif row["group"] in {"18-34", "35-54", "55+"}:
                mask &= data["age_band"].eq(row["group"])
            assert row["n"] == int(mask.sum())
            k = int((mask & selected).sum())
            assert row["numerator"] == (k if row["n"] >= 10 else None)
            assert row["value"] == (k / row["n"] if row["n"] >= 10 else None)
            checked += 1
    return checked
