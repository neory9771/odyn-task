# Glossary

The system answers: **“What does this research data show about the claim?”**
It offers supporting, contradicting and inconclusive evidence, with limitations,
so the user can make a judgment.

Examples below are illustrative, not reported study findings. Example IDs and
values are valid only in the study that defines them.

## Claims and analyses

| Term | Meaning | Example |
|---|---|---|
| **Claim** | The research statement supplied by the user, possibly containing several propositions and explanations. | “Most younger respondents know our brand because advertising reaches them.” |
| **Proposition / subclaim** | One material part of a claim that needs its own evidence or limitation. “Most know our brand” and “advertising caused that awareness” are separate propositions. | c1: younger respondents mostly know the brand; c2: advertising explains their awareness. |
| **Analysis contract** | Private evaluation requirements defining what an acceptable analysis must cover: measures, populations, groups, comparisons, directions and relationships. It allows equivalent programs rather than requiring identical source code. | Require the awareness share for ages 18–34, a comparison against 0.5, and a causal limitation. |
| **Synthesis contract / expectations** | Requirements for the written perspective: explain relevant findings, cite evidence, retain limitations and avoid unsupported conclusions. The judge's checklist is built from executed evidence. | Explain the observed 75% share, cite its result ID, and say advertising causation is not identified. |
| **Reference program** | A known, checked implementation of the intended analysis, used to create targets and reference evidence. It is one acceptable solution. | `e = proportion("awareness", "Yes"); compare(e, "All observed respondents", 0.5, expect=">")`. |
| **Reference evidence** | The structured results returned when the reference program executes against the study. | An estimate records 30 “Yes” answers out of 40, share 0.75, its interval and flags. |
| **Gold** | The reference requirements and results used for evaluation. “Gold” describes their role, not a guarantee that they are error-free or expert-approved. | The held-out answer key requires that same base and numerator, plus an explicit causal gap. |
| **Ground truth** | The declared reference quantity. Observed-data ground truth includes checked counts and shares; synthetic simulation truth can also describe hidden population or causal quantities. These are different sources of truth and are not interchangeable. | Observed truth: 30/40 answered “Yes”. Hidden simulation truth: 60% of the artificial population would answer “Yes”. |
| **Estimand** | The quantity an analysis intends to measure—for example, the share of observed respondents aged 18–34 who report brand awareness. | The share answering “Yes” among observed 18–34-year-olds, rather than among all residents. |

### One example

**Claim:** “Most respondents aged 18–34 know our brand, because our advertising
reaches them.”

**Analysis contract:** select the requested age bands, measure the recorded
awareness answer, compare its share against 0.5, and address the advertising
explanation. If measurement or causal identification is missing, record that
limitation rather than dropping the proposition.

**Reference evidence:** the executed analysis's base, awareness numerator,
share, interval, comparison result and limitation notes.

**Response target:** a perspective explaining what those results support or
contradict, what remains uncertain and which evidence IDs support each finding.
The claim is model input; the private evaluation contract is not supplied as
an answer key during inference.

## Studies and measurements

| Term | Meaning | Example |
|---|---|---|
| **Study** | Respondent answers plus the metadata needed to interpret them. | `respondents.parquet` plus `metadata.json` for an awareness survey. |
| **Catalogue** | The model-facing description of available variables, responses, ordering, question blocks and documented warnings. | A card lists awareness ID 42, label “Brand awareness” and allowed answers “Yes”/“No”. |
| **Variable ID / numeric alias** | The exact identifier used to select a survey measure through the API. A numeric alias is a compact, catalogue-specific mapping to that identifier; it is not an answer code. | Exact ID `brand_awareness` is represented as 42 in one frozen catalogue; “Yes” is its response value. |
| **Label / block ID** | A label describes a variable and need not be unique. A block ID identifies its question block; functions such as `rank_options` take the block ID. | Variable label “Audi”; block ID `car_brands`, used by `rank_options("car_brands")`. |
| **Stored value** | An answer category actually recorded in the dataset, such as `25-34` or `Agree`. | `"25-34"` is recorded; `"18-40"` cannot be guessed as a recorded answer. |
| **Question block** | Related survey variables presented together, such as the options of a select-all question. It is not a respondent group. | “Which brands do you know?” has separate Audi, BMW and Volkswagen checkbox variables. |
| **Comparison-group definitions / study bindings** | Dataset-generation configuration mapping exact response values to named groups. For example, `18-34` combines `18-24` and `25-34`. | `{"18-34": ["18-24", "25-34"], "55+": ["55-64", "65 and above"]}`. |
| **Population / analysis base** | The respondents to whom the requested analysis applies. An observed survey base is not automatically representative of all residents. | A claim concerns Florida residents; the available base contains only Florida survey respondents. |
| **Eligibility / routing** | Eligibility determines who should receive a question; routing implements that rule. An observed answer base does not prove that the true routing is known. | Only people answering “Own a car: Yes” should see the car-brand question. |
| **Missingness / refusal** | A blank answer may reflect routing, nonresponse or checkbox non-selection. The exact value `Prefer not to say` is excluded by API base rules; other combined response labels retain their own semantics. | Blank Audi with BMW selected counts as Audi unselected; an entirely blank brand block is excluded. |
| **Ordered values / top box** | An ordered response sequence; top box counts its last k values. Use it only when that end matches the claim, rather than assuming every sequence means increasing agreement. | For Strongly disagree → Strongly agree, `top_box(v, k=2)` counts Agree and Strongly agree. |
| **Recode / filter** | A recode combines stored categories into analytical groups. A filter restricts respondents; neither changes the underlying recorded answers. | Recode two age bands as “18-34”; `among=where("region", "Florida")` limits the analysis to Florida answers. |
| **Denominator** | The number of respondents included in a particular estimate, after eligibility, observed-answer and population/group rules. | 50 people are in a group, but only 40 give an eligible awareness answer: the share uses 40. |
| **Proxy** | An explicitly justified indirect measure of a concept; it remains distinct from directly measuring that concept. | Self-reported ad recall is used as a declared proxy for exposure; it does not measure actual impressions. |
| **Not measured / identification gap** | A limitation stating that the study cannot support the requested inference. A causal gap can exist even when both variables are measured. | `not_measured("causal effect", "The cross-sectional survey does not identify advertising effects")`. |
| **Estimate / contrast** | An estimate measures a quantity such as a share. A contrast tests a specified difference, threshold, trend or relationship; its label describes evidence under the API rules. | Estimate: younger awareness is 75%. Contrast: younger minus older awareness is 45 percentage points. |
| **Confidence interval / multiple-testing adjustment** | An interval quantifies sampling uncertainty under stated assumptions. Holm adjustment accounts for the program’s multiple tests; neither repairs selection bias or causal identification. | A share has a 95% interval; three claim-implied comparisons also receive Holm-adjusted p-values. |
| **Confounding / adjusted comparison** | A third factor relates to the compared groups and outcome. Adjustment compares within recorded control strata; it does not establish that all confounding has been removed. | Age relates to both ad recall and awareness; an adjusted comparison stratifies by recorded age bands. |
| **Evidence pack** | The API's structured output: estimates, bases, intervals, comparisons, flags and limitation notes with citeable IDs. | `results` contains estimate `c1.r2`, contrast `c1.r3` and a causal-limitation note under c2. |

## Dataset construction

| Term | Meaning | Example |
|---|---|---|
| **Record / example** | One claim-and-study task, with its input, target and metadata. SFT1 and SFT2 have aligned views of the same task. | One record contains an awareness claim, its study input, reference output and generation metadata. |
| **Target / completion** | The desired assistant output used in training: Python code for SFT1, a research perspective for SFT2. | SFT1 target: calls computing awareness; SFT2 target: prose explaining those results with citations. |
| **Synthetic study** | A generated survey with controlled structure and respondents, used to teach capabilities without fitting on client respondents. | Generate 20,000 artificial people, then simulate survey answers for a recruited sample. |
| **API tutorial** | A minimal example isolating API usage. Useful for SFT1 mechanics; excluded from the current research-only SFT2 train/validation subset. | A two-variable survey teaches `group` + `proportion` + `compare` without a large catalogue. |
| **Research claim example** | A task requiring analysis of a substantive proposition or relationship, including its inferential limits. | “Younger respondents are more aware, and advertising caused the difference” needs comparison and limitation. |
| **Paraphrase / claim variant** | Different wording of the same intended analysis. It remains in the original split and is not an independent underlying claim. | “Awareness is higher among younger people” rewords the same younger-versus-older analysis. |
| **Family** | A group of examples asking for the same underlying analysis—for example, comparing brand awareness between ages 18–34 and 55+ using the same measure and group definitions. Different wording does not create a new family. Its examples stay together in train, validation or test so evaluation does not repeat an analysis learned during training. | “Are younger people more aware?” and “Is awareness higher at ages 18–34 than 55+?” share the same measure and groups. |
| **Atomic component / component key** | A constituent analysis identity within a case. Component keys detect shared analysis parts even when the overall claims differ. | A claim about awareness and satisfaction has two components; another awareness-only claim shares the first key. |
| **Split isolation / leakage** | Isolation keeps the required studies, families and component identities disjoint across splits. Leakage means evaluation examples share protected information or identities with learning/development examples. | Putting a claim in train and its paraphrase in test leaks the same analysis across splits. |
| **Lineage** | Metadata linking a record to its source study, family, components, split, seeds, generation method and versions. | A record points to study `demo_01`, family F1, split train, study seed 7 and its generator version. |
| **Coverage** | The capabilities and methodological situations represented in the examples. Balanced coverage does not make the dataset representative of real user traffic. | Include majority, group-difference, proxy and causal-gap cases, with supporting and contradicting results. |
| **`Q_study`** | Structural feature counts for a study, including catalogue width, blocks, answer formats, repeated labels and routing indicators. It is a feature object, not a low/medium/high score. | `variable_count=256`, `block_count=40` and `answer_format_count=3` describe one catalogue. |
| **`Q_case`** | Low/medium/high mapping heuristic using catalogue size and whether required labels appear in the claim. It approximates finding the requested measures. | A 256-variable catalogue with required labels absent from the claim receives high mapping difficulty. |
| **`P`** | Low/medium/high reference-program heuristic using subclaims, estimands, filters, recodes and gap calls. These complexity labels are filtering aids, not measured model difficulty. | A reference with three `subclaim` blocks receives high program complexity under the current heuristic. |
| **Token budget / overlength** | A limit on the completed input-plus-target conversation. Generator token estimates differ from exact model/chat-template counts; overlength examples are rejected or excluded whole, never silently truncated. | An exact 33,000-token conversation exceeds a 32,768-token cap and is excluded whole. |
| **Manifest / fingerprint** | File hashes and recorded versions/settings identifying a dataset or run, used to detect changes and unsafe cache reuse. | A manifest stores the SHA-256 of a split file; editing that file causes integrity validation to fail. |

## Synthetic generation

| Term | Meaning | Example |
|---|---|---|
| **StudySpec** | Private configuration defining a synthetic population, latent relationships, questionnaire, measurement and survey process. It is not supplied to the answering model. | A private spec declares 20,000 people, age bands, latent awareness, survey items and response probabilities. |
| **Domain template / seed** | A reusable questionnaire design and a recorded random seed controlling its generated realization. Reproduction also requires the same configuration and implementation. | Use the `brand_tracking` template with seed 7 to reproduce its study under the same code/configuration. |
| **Data-generating process (DGP)** | The simulation rules that generate demographics, latent constructs and responses, including specified causal relationships. They describe this artificial world, not established facts about client respondents. | A simulated age effect and random noise generate latent awareness, which produces the intended response. |
| **Latent construct / measurement error** | An underlying simulated attribute and the difference between it or its intended response and what the survey records. The model sees recorded answers, not latent states. | A person’s latent awareness would produce “Yes”, but simulated response error records “No”. |
| **Finite population / observed sample** | All simulated people versus those whose answers reach the survey dataset after sampling and response processes. Sample estimates need not equal population quantities. | The simulator contains 20,000 people; only 1,200 have recorded responses after recruitment/nonresponse. |
| **Recruitment / unit versus item nonresponse** | Recruitment determines who is invited. Unit nonresponse loses a person's survey; item nonresponse loses an individual answer. Either can change the observed base. | Invite 2,000 people; 800 give no survey, and 100 of the remaining people skip awareness. |
| **Population truth / causal truth** | Private simulation quantities before the observation process, including population response shares and effects of interventions in the DGP. Knowing them in the generator does not make them identifiable from the visible survey. | Simulation knows a 60% population awareness share and an intervention effect; respondents alone may reveal neither. |
| **Visible artifacts / hidden truth** | Visible study files contain recorded responses and disclosed metadata. Hidden files contain the StudySpec, population and simulation diagnostics; they are excluded from model inputs. | Visible: `respondents.parquet` and metadata. Hidden: `study_spec.json` and `population_truth.json`. |
| **Demographic contrast / scenario rules** | Generator definitions of demographic groups and which measures can support scenario templates. These select cases; they are not extra measured answers or model-facing facts. | A generator compares Under 45 with 45+ and marks ad-recall measures as usable in media-related templates. |
| **Independent count oracle** | Separate pandas calculations checking reference numerators and denominators without using the API's own filter/base implementations. It checks observed counts, not every inference or hidden causal effect. | Separate pandas masks recover 30 awareness selections out of 40 and check the API’s numerator/base. |
| **Stressor** | A simulated challenge such as sparse subgroups, nonresponse bias, unclear routing, measurement error or confounding. Intended stressors and realized conditions may differ. | Simulate lower response probability among unaware people, producing nonresponse bias. |
| **Blueprint / controlled realization** | A declared analysis plan and its template-rendered claim. Checks verify that wording retains the committed groups, population and direction; this is not an LLM paraphrase or expert review. | Commit awareness, younger/older groups and “higher”; render a claim retaining those choices before execution. |
| **Curriculum / paired views** | A collection of API tutorials and research claims. Each underlying case has aligned analysis-code and evidence-response views, so two task rows do not mean two independent claims. | 100 underlying claims produce 100 SFT1 rows and 100 aligned SFT2 rows, not 200 independent claims. |
| **Use case / API operation** | A use case is an analytical situation, such as a majority or causal gap; an API operation is a function used to address it. One use case may require several operations. | Use case: majority. Operations: `proportion` followed by `compare(..., 0.5, expect=">")`. |
| **Benchmark dimensions / strata** | Metadata describing coverage categories, such as evidence direction, measurement availability and inferential limits. Strata are categories used when selecting a balanced subset. | Select cases within support/contradict/inconclusive categories and low/medium/high complexity strata. |
| **Raw curriculum / trainer package** | Generator output and a reviewed package in a schema accepted by the SFT task loader. Successful generator validation alone does not establish trainer compatibility. | A `synthetic-curriculum-1.0.0` output passes its generator checks but still needs assembly into a supported trainer schema. |
| **Client draft / pending expert review** | Candidate evaluation cases awaiting methodological acceptance. Automated execution, count checks and judges do not confer expert approval. | A checked Florida-awareness claim remains a draft until its measurement mapping and interpretation are reviewed. |
| **Partial build / resume / filtered subset** | An unfinished output preserved for recovery; continuing that build; or selecting aligned task rows and required study files using recorded metadata. | A failed build preserves `.partial`; resume eligible client drafts, or filter a completed build to ten API tutorials. |

## Training and evaluation

| Term | Meaning |
|---|---|
| **SFT** | Supervised fine-tuning: fit a model to produce the supplied target completions from their inputs. |
| **SFT1 / SFT2** | The analysis-code model and the evidence-interpretation model, respectively. They use separate adapters. |
| **BF16 LoRA / NF4 QLoRA** | Train LoRA adapters on an unquantized BF16 base or a frozen 4-bit NF4 base, respectively. Neither is full-parameter fine-tuning. |
| **Checkpoint / selection loss** | Saved training state or adapter; the validation loss used to select an adapter. Selection loss is not generated-output accuracy. |
| **Base model / LoRA adapter** | The starting language model and the smaller trainable weight additions representing a fine-tuned task. |
| **Train / validation / test** | Data for fitting weights, selecting checkpoints/development choices, and assessing the frozen system, respectively. |
| **Client calibration** | A separate small client-study set for compatibility checks before freezing the evaluation protocol; not SFT training data. |
| **Teacher-forced loss** | Prediction loss on reference completion tokens. It does not establish that freely generated code runs or that generated prose is grounded. |
| **Standalone SFT2 / reference conditioning** | Evaluate interpretation with the executed reference analysis supplied as input. This isolates SFT2 from SFT1 generation errors. |
| **End-to-end (E2E)** | The actual chain: claim → SFT1 code → API execution → SFT2 perspective → evaluation. SFT2 receives SFT1's actual evidence. |
| **Deterministic checks** | Code-based checks of execution, measured contracts, citations, numbers and suppression. They cover specified rules, not every aspect of research meaning. |
| **LLM-as-a-judge** | A separate model that assesses generated code or prose against explicit requirements, with supporting quotations and structured decisions. |
| **Judge gate / planted controls** | Automated checks that the judge accepts constructed correct cases, catches deliberate errors and gives repeatable decisions. This is not expert calibration. |
| **Unknown judgment** | A provider error or invalid judge response. It is reported separately and cannot count as a pass. |
| **Standard E2E pass** | SFT1 executes and meets its measured contract, and SFT2 passes grounding, critical expectations and generation checks. |
| **E2E with both judges** | Standard E2E plus a separate SFT1 meaning review of the same generated program. Currently assembled by the pilot scripts. |

For the generation strategy, see [dataset-generation.md](dataset-generation.md).
For execution order, commands and exact pass-rate definitions, see
[WALKTHROUGH.md](../WALKTHROUGH.md).
