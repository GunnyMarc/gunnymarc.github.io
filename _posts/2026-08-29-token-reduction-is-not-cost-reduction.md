---
title: "Token Reduction Is Not Cost Reduction: Make It an Observability Problem"
date: 2026-08-29
permalink: /posts/2026/08/token-reduction-is-not-cost-reduction/
tags:
  - finops
  - observability
  - opentelemetry
  - llm
  - ai agents
  - cost engineering
---
There is a paper worth your afternoon if you run coding agents at any scale: *Token Reduction Is Not Cost Reduction* (Weinberger and Hozez, PointFive, arXiv:2607.12161v5, July 2026). The headline result is the kind of thing that should make a platform engineer sit up. Across 5,493 billed executions, the most aggressive context-compression setup cut delivered tool-output tokens by 38.4 percent and *increased* the actual bill by 6.8 percent. A second layer cut visible tokens too and raised billed cost by 48.4 percent.

Nobody was lying. The compression worked exactly as advertised at the layer it measured. The problem is that the layer it measured has almost nothing to do with the invoice.

This is not really a prompt engineering story. It is a telemetry story. The teams shipping these layers had no instrumentation between "bytes I removed" and "dollars the provider charged," so they optimized the only number they could see. If you have ever watched someone tune a cache hit ratio while p99 latency quietly doubled, you already know this shape.

## Where the money actually goes

Start with the cost decomposition, because it explains everything downstream. The authors modeled billed cost as four components and reproduced individual invoices with a median per-run residual around 1 percent. The four components are not priced alike:

| Component | Price multiplier vs. nominal input |
|---|---|
| Uncached input | 1.00x |
| Cache write | 1.25x |
| Cache read | 0.10x |
| Generated output | separate, higher rate |

Now the share of actual billed cost across 2,848 analyzed runs:

```
Cache creation      ####################################  44.3%
Cache reads         #############################         35.4%
Generated output    ########                              10.4%
Unattributed        #######                                8.7%
Uncached input      #                                      1.3%
```

Cache write plus cache read is roughly 80 percent of the bill. The mean run moved about 117,000 cache-read tokens, about 12,000 cache-creation tokens, and only about 715 generated tokens. When 35 percent of your spend is priced at one tenth of nominal, deleting a thousand tokens from that stream saves you the equivalent of a hundred. A token counter reports the thousand.

## The addressability ceiling

The second structural fact is worse for the compression thesis. The authors attributed cost to agent components:

```
Harness base (system prompt + tool defs)   74.7%   not modifiable
Hidden thinking + residual                 19.4%   not directly modifiable
Tool outputs                                3.3%   <- compression surface
Tool-call arguments                         1.4%   <- accessible
Retrieved files                             0.8%   <- accessible
Conversation history                        0.5%   <- accessible
```

Everything a user-side layer can touch adds up to about 6 percent, with a realistic ceiling near 5 percent after you account for what has to be preserved. So that impressive 38.4 percent tool-output reduction translates to roughly 1.3 percent of total cost, before any second-order effects. And the second-order effects are where the sign flips.

Worth flagging: this ceiling is a property of the measured workload, not a law. The same team looked at 41 real interactive sessions and found the framework prefix falls to roughly 8 percent of input tokens, pushing the accessible surface toward 30 percent of cost and the realized-savings bound from about 1.5 percent to about 9 percent. They present that as an upper bound on an unmeasured setting rather than a forecast. Which is the correct thing to do, and also a strong hint that your own workload mix is the variable that matters most. You will not learn it from a paper. You learn it from your own traces.

## The closed loop is what gets you

Here is the mechanism that turns a saving into a loss. An agent is not a pipe. It reads context, decides, acts, and reads again. Strip something it needed and it goes back for it, and every additional turn re-transmits the entire context prefix.

```
   COMPRESSION ON
        |
        v
  [ tool output shrinks 38% ]   <-- the number everyone reports
        |
        v
  [ model loses a detail it needed ]
        |
        v
  [ extra search turn ] --> [ extra diagnosis turn ] --> [ later first edit ]
        |                          |                            |
        +-----------+--------------+----------------------------+
                    v
        [ full prefix re-sent per turn, cache write at 1.25x ]
                    v
             BILL GOES UP 6.8%
```

The trajectory analysis (13,620 classified turns) makes this concrete. One arm's downstream cost increases were about four times its retrieval-phase savings: first edit arrived 0.94 turns later, post-edit retrieval re-entries rose by 0.32 per run, repeated-search tokens rose by 301 per run. Another arm diverged at orientation, accumulating 62 percent of its excess cache reads before the first edit and carrying 38.7k of cache-read context at first edit against a baseline of about 25.9k.

The per-task correlation between tool-output reduction and billed-cost change was Pearson r = 0.15 with a confidence interval crossing zero, and Spearman rho of 0.013. The two numbers are close to unrelated at task level. If you are reporting one and hoping it implies the other, you are running an open loop.

## The failure modes are ordinary engineering bugs

None of the documented failures are exotic AI problems. They are integration problems, and they are exactly the kind of thing observability catches.

**Downstream consumer incompatibility.** A layer rewrote search output into ranked summaries. That output was being piped into shell counting pipelines (`grep | cut | sort | uniq -c`) that expected raw `file:line:content`. The pipeline did not error. It produced wrong counts, silently. This accounted for all four genuine failures in one campaign: tool tokens fell 40.3 percent while success fell to 93.3 percent against 100 percent elsewhere. The fix was a stdout-consumer guard that suppresses ranking when stdout feeds a pipe or a redirect.

**Edit-anchor alteration.** Aggressive compression rewrote the byte-exact spans that SEARCH/REPLACE patching depends on. Patch applies dropped from 27/40 to 15/40, with a 73 percent token cut and a 59 percent cost cut on the runs that no longer did the job. Cheap failure is not efficiency, and the authors deliberately kept failed runs in the cost numerator so it could not look like efficiency.

**Inactive code paths.** Some comparisons were invalid because the thing under test was not actually in the request path. Hence the free byte-diff activation proofs. If you cannot prove your layer touched the bytes, your A/B is measuring noise.

## What to instrument

This is the part that matters for anyone running agents in production. The paper's real contribution is a measurement standard, and a measurement standard is an instrumentation spec in disguise. Here is how I would wire it.

### Emit a span per turn, not per session

Session-level totals hide the mechanism. The unit of analysis has to be the turn, because turns are what re-transmit the prefix.

```
span: agent.run                      (task_id, model, effort, arm, repo_sha)
 |
 +-- span: agent.turn  seq=1  phase=retrieval
 |    +-- span: gen_ai.chat          input/output/cache tokens, cost
 |    +-- span: tool.exec  name=rg   raw_bytes, delivered_bytes, transformed=true
 |
 +-- span: agent.turn  seq=2  phase=diagnosis
 |    +-- span: gen_ai.chat
 |    +-- span: tool.exec  name=read_file
 |
 +-- span: agent.turn  seq=3  phase=implementation
      +-- span: gen_ai.chat
      +-- span: tool.exec  name=apply_patch   anchor_match=false   <-- signal
```

OpenTelemetry's GenAI semantic conventions give you a starting vocabulary: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`. Those conventions are still moving, so check the current spec before you standardize on names.

What they do not yet cover well is the thing that is 80 percent of your bill. Add it under your own namespace and be explicit that it is yours:

| Attribute | Why it exists |
|---|---|
| `agent.cache.read_tokens` | 35.4 percent of spend, priced at 0.1x |
| `agent.cache.write_tokens` | 44.3 percent of spend, priced at 1.25x |
| `agent.cost.usd_total` | provider-returned, never estimated from text length |
| `agent.turn.seq` | turn growth is the dominant cost driver |
| `agent.turn.phase` | retrieval / diagnosis / implementation / test |
| `agent.tool.raw_bytes` | denominator for compression ratio |
| `agent.tool.delivered_bytes` | numerator |
| `agent.layer.active` | activation proof, boolean, per request |
| `agent.layer.version` | which build is in the path |
| `agent.arm` | baseline vs. treatment, for paired analysis |

One rule above all others: take `total_cost_usd` from the provider response and cross-check it against the usage fields and the published pricing schedule. Never derive cost from character counts. That is the original sin the entire paper is about.

### Pipeline shape

```
  agent process
      |  OTel SDK (traces + metrics + logs)
      v
  OTel Collector
      |
      +--> processor: attach pricing schedule, compute derived cost fields
      +--> processor: redact repo content from tool payloads
      |
      +--> traces  --> tracing backend        (per-turn drill-down)
      +--> metrics --> time-series backend    (dashboards, alerts)
      +--> logs    --> log store              (raw tool stdout samples)
```

Putting the pricing schedule in a Collector processor rather than in the agent is deliberate. Prices change, and you want to recompute historical cost without redeploying agents. Redaction belongs there too, since tool output is customer source code.

### Metrics that are worth a dashboard

Derive these from spans and alert on them:

- **Cost per successful execution (CPS).** The terminal metric. Total billed cost divided by successful runs, per task family. Everything else is a leading indicator.
- **Cache read and write token ratio per turn.** If write tokens climb, your prefix is churning and you are paying 1.25x for it.
- **Turns per task, p50 and p95.** Cost grows with turns on every arm tested. A layer that adds turns must beat that growth before per-turn savings become net savings.
- **Context size at first edit.** The single best early-warning signal in the paper. A layer that inflates orientation shows up here long before it shows up in the monthly invoice.
- **Retrieval re-entry count.** Times the agent returns to retrieval after the first edit. This is the closed loop, made countable.
- **Repeated-search token volume.** Near-duplicate queries within a run.
- **Delivered-to-raw byte ratio.** Keep it, but demote it. It is a diagnostic, not a KPI.
- **Anchor match rate on patch application.** Catches the edit-anchor failure mode directly.
- **Transformed-stream-to-pipe count.** Fires when a transformed tool output is written to a non-tty consumer. This is a boolean bug detector, and it would have caught 4 out of 4 of those silent failures.

Use exemplars so a spike on the CPS chart links straight to the trace of an offending run. That link is the whole point of doing this in a tracing system instead of a spreadsheet.

## Map your signals to the evidence ladder

The paper proposes eight layers of evidence, with one rule: a claim at layer k requires measurement at layer k. Recall is not task success. Markers preserved is not task success. Bytes removed are not billed savings. Here it is with the telemetry that satisfies each layer.

```
L1  compression ratio        <- span attrs: raw_bytes / delivered_bytes
L2  preservation quality     <- offline eval harness, marker recall
L3  model-visible change     <- diff of assembled context, logged
L4  production activation    <- agent.layer.active + byte-diff proof
L5  task success             <- deterministic judge, span status
L6  billed cost              <- provider total_cost_usd
L7  cost per success         <- L5 and L6 joined per task
L8  trajectory and failures  <- turn spans, phase, re-entry counts
```

Most vendor claims live at L1 and get read as L7. Your dashboard should make that gap visually obvious. I would put L1 and L7 on the same panel, side by side, so nobody can quote one without seeing the other.

## Rollout workflow

Paired, randomized, holdout-confirmed. Anything less and you are measuring task assignment.

```
  [ task pool ]
        |
        +--> split: pooled set  /  frozen holdout
        |
        v
  for each task x model x effort:
        run ALL arms from identical fresh working copies,
        randomized order, isolated $HOME, pinned binary hashes
        |
        v
  [ per-run ledger: cost, tokens, turns, success, activation proof ]
        |
        v
  task-clustered paired bootstrap  -->  CI on delta cost and CPS
        |
        v
  confirm on frozen holdout  ---> ship / kill
```

The holdout step is not ceremony. In the paper, one arm's modest 2.7 percent saving was pooled-only and its holdout interval crossed zero. The two cost *increases* replicated cleanly on holdout. Wins are fragile in a way that regressions are not, and the honest version of your dashboard should reflect that asymmetry.

One more caution about effective sample size. Intraclass correlation across repetitions ran 0.37 to 0.55, which reduced 712 runs per arm to a Kish effective n of roughly 38 to 45 tasks. Repetitions of the same task are cheap and they do not buy you much power. Task diversity does. Budget accordingly.

## The result nobody expected

The largest improvement in the entire study came from doing the opposite of compression. A grounded-completion study on SWE-bench-derived Go tasks, single shot, 29 rows across four arms:

| Arm | Applied | Resolved | Cost |
|---|---|---|---|
| Raw context only | 22/29 | 1 | 1.00x |
| Compressed context only | 9/29 | 1 | 0.45x |
| **Grounded raw** | **26/29** | **5** | 2.18x |
| Grounded compressed | 24/29 | 2 | 1.82x |

Grounded raw costs more than twice as much per attempt and has the lowest cost per *resolved* row. Adding byte-exact grounding windows beat removing context. Compression cut cost per attempted row by 17 percent and raised cost per resolved row to 2.08x.

The authors point out that a "successes per million tokens" metric would have inverted this verdict entirely. That is not a hypothetical. That is a reporting hazard sitting in a lot of internal decks right now.

Also worth noting for anyone tempted to generalize: the same compression family on a different agent harness and model produced a 12.5 percent cost *reduction* at equal success, with wall time up 238.5 percent. The effect is a property of the layer, harness, model, and workload combination, not of the layer. Which means the only trustworthy answer for your stack comes from your stack.

## What I would build first

If you are starting from nothing, three things, in order.

Emit per-turn spans with provider-returned cost and cache token splits. That alone puts you ahead of most teams, because it converts an invoice into a queryable object.

Add an activation attribute and a byte-diff proof for every context layer in the path. Untested code paths make every comparison meaningless, and this is a two-hour job.

Chart cost per successful execution as your terminal metric and give compression ratio a much smaller box on the same page.

The uncomfortable takeaway from this paper is that the entire user-addressable surface in the measured setting was around 6 percent of cost, and pushing hard on it made things worse more often than better. The comfortable takeaway is that this is a measurement failure, and measurement failures are the kind that engineers already know how to fix.

---

*Source: Weinberger and Hozez (PointFive), "Token Reduction Is Not Cost Reduction," arXiv:2607.12161v5, July 2026. Partial artifacts at `github.com/PointFiveLabs/ai-efficiency-benchmark`, tag v1.0.0, CC BY 4.0. The authors are explicit that they built and configured all evaluated arms themselves without vendor review, and that they decline to rank the systems.*
