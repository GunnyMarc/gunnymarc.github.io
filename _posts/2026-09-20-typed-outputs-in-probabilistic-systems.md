---
title: "Typed Outputs in Probabilistic Systems: Four Tiers, Not Four Generations"
date: 2026-09-20
permalink: /posts/2026/09/typed-outputs-four-tier-comparison/
tags:
  - llm
  - structured outputs
  - constrained decoding
  - classifiers
  - calibration
  - evaluation
  - benchmarks
  - ai architecture
---
Software that consumes model output needs types. Models that generate text do not natively produce them. Four approaches to closing that gap are now in production use, and the launch of TypeSafe AI's Jev this month has produced a wave of coverage that treats the newest one as though it obsoleted the other three.

It didn't. The four approaches solve different problems, and the justification most often offered for the newest one — that LLMs emit malformed JSON — describes a failure class the industry closed in 2024.

This is an attempt to evaluate all four against a single set of criteria and, more importantly, a single evidence standard.

## A note on evidence before any numbers

Two distinctions do most of the work in this post, and blurring them is how the current discourse went sideways.

**Guarantees versus measurements.** A *guarantee* is a structural property of a mechanism that holds by construction and cannot be falsified by a dataset. A *measurement* is an empirical result on some distribution, which may not transfer. "Cannot emit an out-of-schema value" is a guarantee. "Returns well-calibrated probabilities" is a measurement. I keep them in separate vocabulary throughout, and flag vendor language that merges them.

**Evidence classes.** Claims here are tagged vendor-reported, independently reproduced, documented behavior, or unsupported. Unsupported claims are removed rather than hedged — including several that appeared in an earlier draft of my own analysis and turned out not to trace to any published source.

All pricing, latency, and benchmark figures are as published in September 2026. Jev is in waitlisted early access, so read its pricing as early-access pricing, not a steady-state rate.

Acronyms, once: JSON (JavaScript Object Notation), XML (Extensible Markup Language), LLM (large language model), RLHF (reinforcement learning from human feedback), RLCD (reinforcement learning for calibrated decisions), ECE (expected calibration error), NLL (negative log-likelihood).

## Three failure modes, routinely conflated

Before comparing mechanisms, separate the problems:

1. **Structural invalidity** — the output does not parse, or parses but violates the expected schema.
2. **Semantic error** — the output is structurally perfect and substantively wrong.
3. **Opaque uncertainty** — the system cannot tell, at decision time, which outputs are likely to be in category 2.

These have different solutions and different costs. Conflating (1) with (3) is exactly what produces the claim that a type-safe model "cannot hallucinate." It cannot commit failure (1). Failures (2) and (3) are untouched by any typing mechanism, in any tier.

## Tier 1 — Prompt-level structural scaffolding

**Mechanism.** Wrap instructions, reference material, examples, and variable inputs in delimiters — XML tags or JSON objects — so the boundaries between content types are explicit to the model.

```json
{
  "user_metadata": { "account_tier": "enterprise", "region": "North America" },
  "feedback_text": "The latest software update crashes when I export reports.",
  "required_analysis": ["sentiment", "bug_category", "severity"]
}
```

**Guarantee.** None. Scaffolding is a request, not an enforcement mechanism. A model asked to reply in JSON may simply not.

**Measured property.** Improved attribution of content to role. Anthropic's documentation states that XML tags help Claude parse prompts more accurately when a prompt mixes instructions, context, examples, and variable inputs, and recommends wrapping each content type in its own tag, using consistent descriptive names, and nesting where the content has natural hierarchy.

**Evidence class.** Documented vendor guidance. There is no published head-to-head benchmark quantifying the effect for any major model family. If you have seen a "15 percent accuracy improvement from XML structuring" or a "98 percent parsing success rate under JSON mode" cited — I cited both myself in an earlier draft — neither is traceable to a published source.

**Limits.** The mechanistic explanations for why scaffolding works (tags as positional anchors, prose producing diffuse attention) are intuitive but are not established interpretability findings, and I assert none of them here. Scaffolding addresses failure mode (1) probabilistically and (2) and (3) not at all.

## Tier 2 — Constrained decoding

**Mechanism.** A grammar or schema is compiled into a token mask applied at each sampling step. Tokens that would violate the schema get zero probability before sampling. The model does not comply with the schema; it is prevented from departing from it.

**Guarantee.** Structural validity, by construction. OpenAI drew the line explicitly at launch: JSON mode improves reliability for producing valid JSON but does not guarantee conformance to a particular schema, whereas Structured Outputs is designed to ensure outputs exactly match developer-supplied JSON Schemas — so a required key cannot be omitted and an invalid enum value cannot be produced. The open-source stack offers the same guarantee: vLLM supports guided decoding through outlines, lm-format-enforcer, or xgrammar, constraining output to a fixed set of choices, a regular expression, a JSON schema, or a context-free grammar.

**Measured property.** OpenAI reports 100 percent reliability in its own evaluations with `gpt-4o-2024-08-06` under Structured Outputs. Note what that is: a measurement confirming a guarantee, not establishing one.

**Evidence class.** Documented behavior plus vendor evaluation, with the mechanism publicly implemented in open-source serving stacks — which makes it inspectable in a way no closed model's claims are.

**Limits, and the correction this section carries.** Constrained decoding closes failure mode (1) completely. Any argument that motivates a new architecture by citing dropped brackets, misspelled keys, or conversational preamble before the object is arguing against unconstrained generation, which production systems abandoned in 2024.

What constrained decoding does *not* touch:

- **Latency.** The answer is still produced token by token, and delimiters consume output budget. For a decision that resolves to a single label, syntax may dominate the generated sequence.
- **Unit cost.** Output tokens are priced above input tokens across frontier providers. Every bracket is billed.
- **Calibration.** A schema can require a `confidence` field. It cannot make the value in that field honest. Token log-probabilities, with post-hoc methods such as temperature scaling, give a more usable signal than verbalized self-assessment — but neither route yields probabilities the model was trained to calibrate.

Those three gaps, not bracket errors, are the legitimate motivation for the tiers that follow.

**And one thing Tier 2 already delivers: task decomposition.** A schema with several typed fields *is* a decomposed decision graph, and constrained decoding can enforce one today. Hold that thought — it turns out to explain a large share of the Tier 4 benchmark.

## Tier 3 — Task-specific encoder classifiers

This tier gets routinely omitted from comparisons that jump straight from "frontier LLM emitting JSON" to "System One model," which materially overstates the novelty of the latter.

**Mechanism.** A fine-tuned encoder, or a zero-shot span-extraction model of the GLiNER family, maps input directly to a label distribution. No tokens are generated.

**Guarantee.** Structural validity, by construction — the output is a distribution over a closed label set and cannot be anything else. Identical in kind to the guarantee Tier 4 offers.

**Measured property.** Accuracy and calibration on the target distribution, both typically strong after fine-tuning, at latency and cost well below any generative approach.

**Evidence class.** Independent, public, small-n. This is, notably, the tier with the most public runnable head-to-heads against Jev:

- A probability-aware benchmark (AbdelStark, `jev-benchmarks`) comparing Jev with a GLiNER-class zero-shot classifier across three conditions found a clear accuracy and Brier-score advantage for Jev on AG News and Banking77 — while on the DAIR Emotion set the accuracy difference was unresolved and Jev was substantially *worse* calibrated: Brier 0.846 against 0.668, NLL 5.588 against 1.381.
- A second benchmark (elcronos, `jev-vs-open-decision-models`) compared Jev against two open zero-shot decision models, PrismNLI and Laya, across four datasets, and included a supervised logistic-regression baseline. The supervised baseline beat every zero-shot system, Jev included, on every dataset.

**Limits.** Fine-tuned encoders require labeled data and a training cycle per task, and do not transfer to a label set they have not seen. That is the real gap Tier 4 addresses.

**Consequence for the thesis.** Typed output is not novel. Cheap typed output is not novel. What is novel in Tier 4 is *zero-shot generality* at encoder-class latency and cost, with no per-task training. The elcronos result sharpens the point from the other side: where labeled data exists, a trained classifier of trivial capacity still outperforms zero-shot Jev. Tier 4's advantage is confined to the case where no such data is available.

## Tier 4 — System One models

**Mechanism.** The model takes unstructured program state as input and returns typed, probabilistic decisions in a single parallel pass — evaluating all declared questions simultaneously, returning typed answers plus probabilities directly, with no text generation and nothing to parse. The category name borrows Kahneman's distinction between fast, intuitive System 1 thinking and slower, deliberate System 2 reasoning. Training uses RLCD rather than RLHF. (That acronym collides with an unrelated 2023 method, reinforcement learning from contrastive distillation.)

The correct technical description is non-autoregressive parallel evaluation of a declared question set. It is *not* "parallel sampling," which in the literature means drawing multiple independent completions from a single prompt.

**Interface.** Three primitives: `Choice`, one-of-N with up to 255 options; `Score`, a position on a 2-to-10 level scale; and `Noul`, a yes/no probability. Input state may be a JSON object or a plain string.

**Guarantee.** The model cannot return a value outside the declared schema. That is the entire scope of the guarantee. It *can* return the wrong valid value. It is not trained to generate text at all, and therefore cannot produce code or prose.

**Measured property, stated separately because that is the whole discipline here.** Calibrated confidence is a measurement, not a guarantee. It is reported to hold on the vendor's workflows and has already been shown to degrade on at least one public dataset. Vendor and secondary coverage describing the model as one that "never hallucinates" merges the guarantee and the measurement; even sympathetic reviews caution against reading zero-hallucination language as a promise of semantic correctness.

**Evidence class.** Vendor-reported for all headline figures, with a handful of small independent tests. The architecture is unpublished, the weights are not released, and the benchmark behind the headline multipliers is a set of internal workflow evaluations rather than a public leaderboard.

## The benchmark, and what the accuracy column actually measures

All figures below are **vendor-run, consensus-labeled, and not independently reproduced.** That provenance belongs in the table rather than a footnote, because it changes what the numbers mean: the "accuracy" column is agreement with a reference policy built by averaging GPT-6 Astra and Claude Fable 5.1 at high thinking. That measures agreement with two frontier models, not correctness — and TypeSafe acknowledges the reference biases toward OpenAI and Anthropic. The dashboard covers 711 cases across four tasks under a shared harness, which makes it useful for comparing workflow behavior and unsuitable as a correctness claim.

**Table 1 — Four-workflow evaluation.** Source: TypeSafe AI internal dashboard, September 2026, as reported in secondary coverage (OrcaRouter; Anhaia, DEV Community; DataCamp). Reference labels: mean of two frontier models. Not independently reproduced.

| Model | Aggregate | Security incidents | Agent-trace observability | Invoice processing | Customer service | Cost / case | Latency / case |
|---|---|---|---|---|---|---|---|
| Jev | 67.8% | 61.7% | 71.6% | 61.8% | 76.0% | $0.0004 | 0.4 s |
| GPT-5.6 Luna | 66.8% | — | — | — | — | $0.0033 | 12.9 s |
| GPT-5.6 Terra | 67.9% | — | — | — | — | $0.0304 | 10.1 s |
| Claude Sonnet 5 | 67.8% | — | — | — | — | $0.1174 | 78.1 s |
| Claude Opus 5 | 73.1% | 66.2% | 76.6%\* | 79.1%\* | — | $0.1761 | 37.8 s |
| GPT-5.6 Sol | 74.1% | — | — | — | — | $0.0836 | 23.3 s |

\* Reported as the best comparator on that workflow; the first per-workflow comparison is attributed explicitly to Opus 5. On customer service the best comparator scored 78.3%, but secondary reporting does not identify which model that was, so the figure sits in the prose rather than in a row. Cells marked "—" are not published in the sources consulted, and are left blank rather than estimated. The dashboard does not state which Jev build produced these figures.

**Reading the table.** Jev sits level with a mid-tier frontier model on aggregate agreement and 5–6 points below the top comparators, with the gap widening sharply on invoice processing — 61.8% against 79.1%, a 17-point deficit — and narrowing to about two points on customer service (76.0% against 78.3%).

The most-cited framing in secondary coverage is that a model scoring 67.8% is matched exactly by Claude Sonnet 5 at 293x the cost per case and 195x the latency. That framing is accurate and also incomplete in two directions. It holds against the *matched* comparator, not the best one. And it ignores the *nearest-cheap* comparator: GPT-5.6 Luna scores one point behind Jev at $0.0033 and 12.9 s per case, which puts Jev's advantage at roughly 8x on cost and 32x on latency. Single-digit on cost, not two or three orders of magnitude.

Which comparator you pick determines the multiplier, and the honest range runs from 8x to 440x.

## The decomposition finding, which is the most important number in the benchmark

The vendor's own evaluation page reports that *every* frontier comparator improved when the same workflow was posed as a set of typed Choice/Score/Noul questions rather than as a single open-ended task. The clearest published instance is GPT-5.6 Luna, which moved from 51.9% to 66.8% on aggregate under decomposition.

Roughly fifteen points of Luna's score came from splitting the task, not from changing the model.

Decomposition into typed fields is exactly what a constrained-decoding schema does. A Tier 2 pipeline that adopts the same decomposition should be expected to capture most of the same gain. The Tier 4 architectural contribution, net of decomposition, is the residual cost, latency, and calibration difference — which is the whole thesis, now stated with the vendor's own evidence.

No one has published the controlled comparison that would settle this: the same decomposed question set run through a constrained-decoding pipeline and through Jev.

**Table 2 — Independent tests.** Different evidence class, so a separate table.

| Source | Task | Result | Jev build |
|---|---|---|---|
| Every | Extraction vs. Claude Fable 5.1 | ~25x faster, ~580x cheaper; 0.35 s vs. 8.83 s per passage | Not stated |
| Every | Judgment throughput | 777 judgments in under 0.7 s for roughly a quarter of a cent | Not stated |
| NearHere (UK events site) | Listing moderation vs. Gemini Flash-Lite | 96% vs. 86%, 58x cheaper per decision. **Single-source; primary post not located. Treat as unverified.** | Not stated |
| AbdelStark, jev-benchmarks | Zero-shot classification vs. GLiNER-class model | Jev ahead on AG News and Banking77; worse calibrated on DAIR Emotion (Brier 0.846 vs. 0.668) | jev-1.13.0 |
| elcronos, jev-vs-open-decision-models | Zero-shot classification vs. PrismNLI and Laya, four datasets, plus supervised baseline | Supervised baseline beat every zero-shot system, Jev included, on every dataset | jev-1.13-20260917 |

The samples are small, and the two benchmarks that record a model version record *different* ones. The defensible summary: the speed and cost claims survive contact with a third party, the accuracy claim sits a notch below the frontier, a trained classifier still wins where training data exists, and nobody should draw a production conclusion from this volume of external testing.

## Re-deriving the headline multipliers

**Published list figures.** $0.042 per million input tokens, with output priced at zero, described by the company as "too cheap to meter." End-to-end latency of 70 to 500 milliseconds, against 3 to 329 seconds for the frontier comparators.

**A unit correction worth making.** $42 per billion input tokens equals $0.042 per million — the arithmetic holds — but $0.042 is **4.2 cents**, not a fraction of a cent. The sub-cent figure is the *per-decision* cost of approximately $0.0004. Those measure different things: one is a token rate, the other is the cost of a whole decision at a typical input length. I merged the two in an earlier draft; they are not interchangeable.

TypeSafe reports Jev is 193.6x faster and 444.6x cheaper than frontier LLMs in peak in-house testing. Deriving from Table 1:

- **Cost vs. most expensive comparator (Opus 5):** $0.1761 ÷ $0.0004 ≈ **440x**. Close enough to the published 444.6x to conclude the cost multiplier is the Jev-versus-Opus-5 pairing.
- **Cost vs. accuracy-matched comparators:** ≈ **76x** (Terra), ≈ **293x** (Sonnet 5).
- **Cost vs. nearest-cheap comparator (Luna):** ≈ **8x**.
- **Latency vs. slowest comparator (Sonnet 5):** 78.1 s ÷ 0.4 s ≈ **195x** — within one percent of the published 193.6x.
- **Latency vs. accuracy-matched comparator (Terra):** ≈ **25x**.
- **Latency vs. nearest-cheap comparator (Luna):** ≈ **32x**.

The likely reading — an inference from the table, not something TypeSafe has confirmed — is that the two headline multipliers come from two *different* comparators: cost from the Opus 5 pairing, latency from the Sonnet 5 pairing. Which makes quoting "193.6x faster and 444.6x cheaper" as a single comparison against "frontier LLMs" misleading in a specific way: no single model in the table is both 194x slower and 445x more expensive than Jev.

**If you are citing these numbers**, quote ~8x cheaper and ~32x faster against the nearest-cheap comparator; 76x cheaper and ~25x faster against the accuracy-matched comparator; 440x cheaper and ~95x faster against the most expensive comparator; 293x cheaper and ~195x faster against the slowest. Treat 193.6x/444.6x as vendor peak figures that pair two different comparators.

**Pricing durability.** These are September 2026 early-access rates under waitlisted access. Whether an independent benchmark confirms accuracy parity, and whether the pricing holds once the subsidy runs out, are the two open commercial questions, and neither is answerable yet. Any total-cost model built on these figures needs a sensitivity analysis, not a spreadsheet cell containing $0.042.

## Observability: what you gain and what you lose

**Gained: uncertainty as a first-class, thresholdable output.** Probabilities come back alongside every decision rather than described in prose, which makes it straightforward to automate clear cases and route uncertain ones to review. The threshold is an application-level policy, not a model property — published integration guidance directs teams to test decision thresholds against the cost of mistakes in their own application, and to keep policy and execution in application code when the model sits inside an agent loop.

**Lost: the reasoning trace.** A model that emits no text cannot explain why it chose a label. Under this architecture a debugging trace contains the input state, the declared question set, the returned value, and its probability — and nothing about the path between them. Prior structured-output approaches let a trace separate reasoning content from final-answer content. That capability does not survive the move to Tier 4.

This is a trade, not an upgrade. For high-volume routing where the cost of an individual error is bounded, exposed uncertainty is worth more than an explanation. For decisions that will be audited, contested, appealed, or explained to a regulator, an unexplainable classifier with a good confidence score may be the worse instrument regardless of its accuracy.

**One attribution correction.** Two benefits commonly credited to structured output are not caused by it. Token accounting and per-call cost attribution come from API usage metadata and are identical whether the response is JSON or prose. And per-field latency profiling inside a JSON object is not a real capability — you can timestamp tokens in a stream, but attributing elapsed time to a specific key has no operational meaning when compute and billing are per-token.

## What you take on: the integration and evaluation burden

Token overhead does not vanish. It converts into engineering work. Adopting Tier 4 obliges a team to own:

- **Label design.** The closed answer set becomes a schema artifact under version control, with migration costs when it changes. `Choice` is bounded at 255 options.
- **Threshold selection.** Every decision point needs an automation threshold, chosen against the measured cost of a false positive and a false negative *in that specific workflow*.
- **Fallback policy.** What happens below threshold — human queue, retry with a generative model, conservative default — is application logic that did not exist before.
- **A local evaluation set.** Because the vendor's accuracy figure measures agreement with other models rather than correctness, you cannot inherit it. Label your own cases and measure against them.
- **Calibration monitoring.** Calibration is a measurement on a distribution, and production drift degrades it silently. Track ECE or Brier on a held-out slice over time, not once at onboarding.
- **No explanation for incident review.** When a decision is wrong, there is nothing to read.

## When not to reach for this

The intended deployment is two models, not one: the typed decision model makes the call, and a generative model handles anything requiring prose, code, or an explanation. A System One model is the wrong instrument when:

- **The output must be read by a person.** Chat, drafting, summarization, code generation. It is not trained to generate text.
- **An explanation is part of the deliverable.** Audit trails, regulated decisions, anything appealable.
- **The answer set is open or unstable.** Outputs must be enumerable in advance; `Choice` tops out at 255 options.
- **Volume is low.** The advantage is per-decision cost and latency at scale. At a hundred decisions a day, the integration and evaluation work above exceeds the savings.
- **Frontier accuracy is required.** On the vendor's own benchmark it trails the best comparators by 5–6 points on aggregate and 17 on the weakest workflow.
- **A fixed, well-labeled task already exists.** With training data and a stable label set, a fine-tuned encoder may be cheaper, faster, and better calibrated on your distribution.
- **Structural validity is the whole problem.** Constrained decoding already solves it, at no architectural cost.

## Open questions

1. Accuracy trails the frontier on the vendor's own benchmark, materially on at least one workflow.
2. The reference policy is model-derived; "accuracy" means agreement, not correctness.
3. Calibration is task-dependent and has degraded on a public dataset in independent testing.
4. Pricing is early-access and its durability is untested.
5. Access is waitlisted, which limits the pool of potential replicators.
6. The architecture is undisclosed, so the calibration claim cannot be analyzed from first principles.
7. Independent testing totals a handful of small studies, one of which (NearHere) I could not trace to a primary source.
8. Model versions are not pinned across sources — `jev-1.13.0` in one benchmark, `jev-1.13-20260917` in another, unstated on the vendor dashboard. Differences between sources may reflect version drift rather than task or method, and there is no way to separate the two from outside.
9. The decomposition finding means the benchmark does not isolate the architectural contribution from the workflow contribution of task splitting.

## Where this lands

Read as four tiers rather than four generations, the picture resolves cleanly. Prompt scaffolding improves role attribution and remains good practice; it guarantees nothing. Constrained decoding closed structural invalidity as a failure class in 2024, which removes the most commonly cited justification for architectural change. Encoder classifiers have delivered cheap, fast, well-calibrated typed output for years, at the cost of per-task training. System One models extend that to zero-shot generality.

Jev's contribution is latency, unit cost, and exposed calibrated uncertainty — not type safety, which two earlier tiers already provide. Those gains are real and, on the speed and cost axes, have survived initial independent contact. They come with an accuracy deficit against the best frontier comparators, an evaluation burden transferred to the application, and the elimination of any reasoning trace.

For high-volume, repeated, closed-set decisions with bounded error cost and no explanatory requirement, a typed decision model is now the better-shaped instrument. Outside that envelope it is not a replacement, and its vendor does not present it as one. The claim worth carrying forward is narrower than the launch coverage suggests: an architecture well matched to one class of decision, with published numbers nobody outside the vendor has yet reproduced at scale.

---

## References

**Constrained decoding and prompt structure**

- OpenAI. "Introducing Structured Outputs in the API." [openai.com](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- OpenAI. "Structured model outputs — API documentation." [developers.openai.com](https://developers.openai.com/api/docs/guides/structured-outputs)
- Anthropic. "Use XML tags to structure your prompts." [platform.claude.com](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags)
- Anthropic. "Prompting best practices." [platform.claude.com](https://platform.claude.com/docs/claude/prompt-library)
- vLLM. "Structured Outputs." [docs.vllm.ai](https://docs.vllm.ai/en/v0.8.2/_sources/features/structured_outputs.md)
- vLLM. "Structured Decoding in vLLM: a gentle introduction." [vllm.ai](https://vllm.ai/blog/struct-decode-intro)

**Jev and System One models**

- TypeSafe AI. "Introducing System One Models and Jev." [typesafe.ai](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- Vercel. "What is Jev, TypeSafe AI's System One model?" [vercel.com](https://vercel.com/i/what-is-jev)
- Vercel. "TypeSafe AI's Jev now available on AI Gateway." [vercel.com](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway)
- The Register. "TypeSafe AI debuts model for machines that plays Doom." [theregister.com](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/)
- DataCamp. "System One Models: Jev." [datacamp.com](https://www.datacamp.com/blog/system-one-models-jev)
- Anhaia, G. "Your agent burns LLM money on switch statements: Jev claims 444x less." DEV Community. [dev.to](https://dev.to/gabrielanhaia/your-agent-burns-llm-money-on-switch-statements-jev-claims-444x-less-23l4)
- DEV Community. "How to Use Jev: A practical guide to TypeSafe's System One model." [dev.to](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e)
- OrcaRouter. "Jev: TypeSafe's Decision Model, Speed and Cost Explained." [orcarouter.ai](https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know)
- LangChain. "Building a harness with Jev." [langchain.com](https://www.langchain.com/blog/building-a-harness-with-jev)
- Arize AI. "TypeSafe Jev: Can Decision Models Replace LLM Judges?" [arize.com](https://arize.com/blog/typesafe-jev-llm-judge/)
- Forkast. "TypeSafe AI's Jev Is Not an LLM — And That May Be the Point." [forkast.news](https://forkast.news/typesafe-ais-jev-is-not-an-llm-and-that-may-be-the-point/)
- Kingy AI. "TypeSafe Jev Review: The AI Model That Doesn't Generate Text." [kingy.ai](https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/)

**Independent benchmarks**

- AbdelStark. "jev-benchmarks: probability-aware evaluation for typed decision models." [github.com](https://github.com/AbdelStark/jev-benchmarks)
- elcronos. "jev-vs-open-decision-models: Jev vs. PrismNLI and Laya on four classification datasets." [github.com](https://github.com/elcronos/jev-vs-open-decision-models)
- AbdelStark. "awesome-typesafe: a curated index of TypeSafe AI and Jev resources." [github.com](https://github.com/AbdelStark/awesome-typesafe)
