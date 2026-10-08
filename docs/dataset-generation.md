# How we generate the datasets

See the [glossary](glossary.md) for definitions of claims, contracts, evidence
and dataset identities.

We teach two capabilities: turn a research claim into an executable analysis
(SFT1), then turn its evidence into a grounded research perspective (SFT2).
An example is therefore a **claim, study, analysis contract, executed reference
and response target**, rather than an isolated question/answer pair.

The target is evidence support, contradiction and uncertainty for each material
proposition. It is not a binary verdict on whether the whole claim is true.

## Where the dataset CLI fits

The overall strategy is **prepare studies and reference cases → train the two
adapters → select on validation → evaluate the frozen pipeline on test**.
`odyn-dataset` owns the preparation phase. It produces examples that teach the
behaviour, together with private contracts that let evaluation check it. It
does not fit weights or score the deployed pipeline's generated answers.

| Strategic need | Dataset actions | Why it exists |
|---|---|---|
| Generate controlled training studies | `synthetic curriculum/validate-curriculum/filter` | Build seeded paired curricula and select small experiments |
| Understand a supplied study | `metadata build` | Give claims and execution a consistent catalogue |
| Define the client benchmark | `evaluation audit/generate` | Check feasibility and construct claims with reference contracts |
| Create learning and held-out partitions from a supplied study | `splits audit/generate/validate` | Build paired examples while isolating analytical families |
| Supply SFT2 response targets | `synthesis export` or `write-targets` + `assemble` | Choose existing evidence summaries or gated written perspectives |
| Test sensitivity to wording | `variants validate-verifier/generate/apply` | Vary the input without changing its intended analysis |
| Check capability and grading coverage | `use-cases build`, `rubrics build/validate` | Construct controlled API fixtures and evaluation criteria |
| Accept a frozen package for use | `splits validate` or `synthesis validate` | Verify hashes, alignment and split integrity |

These are tools for different preparation needs, not a checklist to run every
time. With an existing SFT2 package, the path is simply **validate package →
`odyn-sft prepare/train` → held-out generation and judging**. SFT1 remains a
separate adapter; SFT2 consumes executed evidence. Rebuilding claims, adding
paraphrases and rewriting targets are optional dataset changes, not required
steps when launching another experiment.

## 1. Start from a study and an analytical contract

The study contains respondent answers and a catalogue describing their meaning:
variable IDs, stored responses, ordering, blocks and documented limitations.
For synthetic training studies, questionnaire structures and respondents are
generated with controlled seeds. This allows varied domains, missingness,
sample sizes and answer formats without training on the client respondents.
The supplied 2022 survey provides real-study calibration and test candidates.

Before writing a claim, define what it asks us to estimate: the population/base,
measure, groups, comparison direction and any linked proposition. A claim can
combine something measured with something the study cannot establish. That
second part becomes an explicit limitation, not an invented variable.

For example: “Among respondents aged 18–34, most selected the travel option,
because our advertising reaches them.” The analysis needs an observed share
against 0.5 and a separate assessment of advertising/causal identification.
The share cannot, by itself, establish the proposed explanation.

## 2. Bind claim concepts to recorded answers

Study bindings are analyst-defined comparison groups using **exact catalogue
IDs and stored values**. Build them by reading the catalogue, choosing meaningful
comparisons and writing the bindings JSON. The packaged
`survey_2022_bindings.json` contains 24 such definitions.

For example, the group label `18-34` combines the recorded responses `18-24`
and `25-34`; `55+` combines `55-64` and `65 and above`. These aggregations are
analytical choices, not additional response categories supplied by the survey.
A group label must describe its members accurately: those first two bands do
not measure ages 18–40.

The loader validates the schema, unique values, disjoint groups and exclusion
of the exact refusal value. Catalogue checks retain definitions whose IDs and
values exist in the study. Bindings guide dataset generation; the bindings JSON
is not inserted into the inference prompt. The prompt gives the claim and study
catalogue, from which SFT1 must reconstruct the requested grouping.

## 3. Construct claims, then execute their reference analyses

Templates express controlled propositions and relationships: majorities, group
differences, restricted populations, multiple outcomes, proxies and missing
causal, historical or population evidence. Synthetic examples vary the study as
well as the claim. Minimal API tutorials isolate one function; research claims
combine functions to address an actual analytical question.

Generate a reference Python program from the contract, execute it through the
Survey API and retain its evidence: bases, numerators, estimates, intervals,
comparison labels and limitations. Reference semantic checks and, where
implemented, independent count checks provide additional verification. An API
test, a generator and a scorer must actually support a capability before we
claim that the dataset covers it.

For the paired client-survey generator, comparison direction and family split
ownership are fixed before reference outcomes are computed. Coverage selection
can subsequently use observed evidence categories. This prevents wording or
significance from determining split ownership; it does not make the benchmark
a random sample of real client questions.

## 4. Produce two aligned training views

| View | Input | Target |
|---|---|---|
| SFT1 | Claim + catalogue + authoring API instructions | Executable reference program |
| SFT2 | Same claim + reference program + its executed evidence + interpretation instructions | Evidence-grounded research response |

Both views preserve claim and record identities. For SFT1, the reference program
is one valid solution; the evaluation contract permits equivalent analyses.
For SFT2, deterministic targets summarise executed evidence with citations and
limitations. Optional OpenAI-written targets add more natural perspectives and
are retained only after grounding checks and a separate semantic judgment.
The currently selected Mistral SFT2 subset reuses deterministic targets.

We know the intended analysis and its reference numbers. That provides ground
truth for **what the supplied data show under the declared analysis**, not for
whether an unobserved causal explanation or population claim is universally
true. Synthetic hidden mechanisms can support generator diagnostics, but they
are not supplied to the model as otherwise unavailable evidence.

Standalone SFT2 training/evaluation uses reference execution. Real E2E evaluation
instead gives SFT2 the code and evidence generated by SFT1, exposing upstream
errors. These measure different things; see [the walkthrough](../WALKTHROUGH.md).

## 5. Balance coverage and record simple complexity

Coverage is deliberate: measured, partially measured and unavailable claims;
supporting, contradicting and inconclusive evidence; answer formats; and
single versus linked propositions. Unsupported cells and capacity shortages
remain visible. More examples do not compensate for a missing capability.

Existing synthetic records store inexpensive heuristics: `Q_study` describes
catalogue structure, `Q_case` approximates mapping difficulty from catalogue
size and whether labels are exposed, and `P` approximates program complexity
from subclaims, estimates, filters, recodes and gap declarations. Low/medium/high
are selection aids, not experimentally measured model difficulty. Raw features
and the heuristic version are retained so subsets can be explained.

API tutorials are useful for SFT1 mechanics. The current research-only SFT2
subset excludes `task_type=api_tutorial` from train and validation, selecting
1,000 training records across use-case/complexity strata and retaining 898
validation records. Research claims still retain their executed API evidence.

## 6. Separate learning, selection and final assessment

Training fits the adapters; validation selects checkpoints and development
choices; client calibration checks compatibility before protocol freeze; test
assesses the frozen system. Synthetic train/validation studies are separate.
Within the single client survey, isolate analytical families and their atomic
components between calibration and test. Rewording the same analysis does not
make it a new independent case or justify moving it into another split.

Optional OpenAI paraphrasing changes wording while retaining the intended
analysis, with a separate meaning verifier. Variants stay in the original split
and carry their lineage. Report both row counts and underlying claim counts.

Store record/study/family IDs, component keys, seeds, generation method,
task type, complexity, prompt/API versions and file hashes. Validate alignment
and split isolation before use. Measure model-specific completed-conversation
tokens when preparing data; exclude whole overlength examples rather than
truncating their claim or target. Test is not tokenized for fitting or used for
checkpoint selection.

Executed references and automated judges can still contain errors. Packages
marked pending expert review are development artifacts; neither balanced
coverage nor a passing automated judge gate certifies client acceptance.

## Commands and implementation boundary

Run from this folder with the package installed. `odyn-dataset` and
`python -m odyn_sft.dataset_generation` expose the same interface. Every command
supports `--help`; generator configs are separate from training configs.

`dataset_generation/cli.py` registers the workflows and formats their reports.
Each module in `dataset_generation/commands/` defines one workflow’s arguments
and handler; synthetic uses `synthetic/cli.py`. These CLI modules defer
implementation imports until execution, keeping help lightweight.

| Command | Purpose | OpenAI calls |
|---|---|---|
| `synthetic curriculum/validate-curriculum/filter` | Build/check seeded paired curricula and select small experiments | No |
| `metadata build` | Build metadata from a downloaded survey | No |
| `evaluation audit/generate` | Plan/render client evaluation candidates | No |
| `splits audit/generate/validate` | Plan, build and validate paired packages | No |
| `synthesis export` | Reuse an existing synthetic paired package's response targets | No |
| `synthesis write-targets` | Write and check optional research-response targets | Yes |
| `synthesis assemble/validate` | Assemble accepted targets and check the package | No |
| `variants validate-verifier/generate` | Check the verifier, then write/verify paraphrases | Yes |
| `variants apply` | Assemble an existing set of accepted variants | No |
| `use-cases build` | Build the controlled API fixture | No |
| `rubrics build/validate` | Build/check judge contracts | No |

The seeded synthetic generator is included in `dataset_generation/synthetic/`.
It constructs a questionnaire and finite population, simulates recruitment and
responses, then writes separate visible study files and hidden simulation truth.
Executable reference targets use the visible responses; hidden population truth
is diagnostic and never included in the model's prompt.

`templates.py`/`spec.py` define studies, `population.py`/`survey.py` simulate them,
`curriculum_studies.py` scales questionnaire width, and `curriculum_cases.py`
builds research claims and minimal API lessons. `curriculum.py` assembles paired
SFT1/SFT2 rows; `complexity.py` labels them with inexpensive filtering heuristics.
Studies, families and component keys are isolated across splits. Generated
metadata records seeds, study/domain, task type, API use case and complexity.

```bash
python -m pip install '.[dataset]'
odyn-dataset synthetic curriculum --config configs/synthetic-curriculum-small.json \
    --output data/synthetic-small
odyn-dataset synthetic validate-curriculum --run data/synthetic-small
odyn-dataset synthetic filter --run data/synthetic-small --split train \
    --limit 10 --task-type api_tutorial --output data/api-lessons
```

The first curriculum build may download the small `o200k_base` tokenizer
vocabulary into the tiktoken cache; later builds can use that cache offline.
These token counts are filtering estimates, not model-specific context audits.

The example config disables client drafts and creates only synthetic cases.
`spec` and `study` create one study; `build`/`validate` create/check an analysis
benchmark; `curriculum`/`validate-curriculum` create/check paired curricula;
`filter` exports aligned subsets, and `resume-client` finishes optional client
drafts from a partial build. None calls an LLM. Outputs refuse overwrites.
Raw curriculum manifests use their own schema: validation here checks generation,
not acceptance by the SFT task loader. Reviewed rows must be assembled into a
supported trainer package before fitting; generating a curriculum does not
launch training or replace an existing production dataset.

The supplied-study workflow is metadata → audit → review plan → generate →
validate. Audit may execute candidate reference analyses to check feasibility,
but writes a plan rather than training samples. Generation requires
`--acknowledge-plan` and unchanged input/code hashes.

For SFT2, choose either existing deterministic targets or accepted written ones:

```bash
odyn-dataset synthesis export --source-suite /path/to/paired-suite --out data/sft2-suite
# Alternative, using already accepted written targets:
odyn-dataset synthesis assemble --targets-dir /path/to/accepted-targets \
  --source-suite /path/to/paired-suite --out data/sft2-written-suite
odyn-dataset synthesis validate --suite data/sft2-suite
```

Use fresh output directories. Assembly optionally accepts `--token-config` and
`--max-example-tokens` to filter complete train/validation examples. See
[Mistral's research subset](mistral-sft2-research-1000.md) for the current
1,000/898 selection. Training, inference and held-out judging are separate
operations; their commands are explained in the walkthrough.
