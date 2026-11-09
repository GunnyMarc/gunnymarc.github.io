---
title: "Catching LLMJacking Before the Invoice Does: Observability, Tokenomics, and Jev as a Triage Layer"
date: 2026-11-08
permalink: /posts/2026/11/catching-llmjacking-observability-tokenomics-jev-triage/
render_with_liquid: false
tags:
  - llm
  - security
  - llmjacking
  - observability
  - opentelemetry
  - token economics
  - cost engineering
  - finops
  - classifiers
  - system one models
---

*Source threat report: [The Rise of LLMJacking: How Threat Actors Monetize Compromised Cloud and Model Resources](https://cloud.google.com/blog/topics/threat-intelligence/llmjacking-monetizing-compromised-cloud-models), Google Threat Intelligence Group / Mandiant, November 2, 2026.*

---

## TL;DR

LLMJacking is not cryptojacking with a new name. Mandiant reports charges above **$100,000 in a single day** and cumulative charges reaching **$1 million** at some victims before access was revoked. A monthly billing alert will not save you. The detection signal has to live where the spend is created: the model gateway, in tokens, in near real time.

This post builds a three-layer defense:

1. **Observability**: one structured event per model call, so every key, principal, model, and token is attributable.
2. **Tokenomics**: treat tokens as a metered currency with per-key baselines, burn-rate alerts, and hard budgets enforced in the request path.
3. **Jev**: a fast typed decision model used out-of-band to triage the ambiguous events that rules and statistics cannot settle (for example, "is this traffic still the product we shipped?").

Rules and math stay in code. Jev only makes the judgment calls, and its own cost is part of the design.

---

## 1. What the attackers actually do (and what that means for telemetry)

The GTIG report describes a consistent kill chain. Each stage leaves a different observable footprint.

| Stage | Observed behavior (per GTIG) | Telemetry that exposes it |
|---|---|---|
| Initial access | Exposed Ollama endpoints probed via `/api/create`; stolen IAM credentials, service account keys, GitHub PATs, session tokens; LLM keys harvested from public GitHub repos | Auth events, new source ASN/geo per key, first-seen user agents |
| Validation sweeps | Automated credential checks with minimal single-token requests (`max_tokens=1`) | Many tiny requests, near-zero output tokens, many distinct keys from one source |
| Privilege escalation | Token-creation abuse; a LiteLLM gateway `/key/regenerate` user ID rebind (CVE-2026-40217) elevating virtual keys to `PROXY_ADMIN` | Key-management API calls, role changes, key creation by non-owners |
| Scaling | New API keys created, services enabled, Pay-As-You-Go agreements, premium models selected | Control-plane audit events (enablement, deployment writes, model agreements) |
| Monetization | LLM Mixers pooling stolen keys: one actor made 500,000+ calls in under eight hours, passing nearly $1M in token costs to the victim | Sustained token throughput far above baseline; traffic unrelated to the app's purpose |
| Evasion | Querying or altering model invocation logging and log sinks | Changes to logging configuration (Bedrock, Azure diagnostic settings, GCP sinks) |

Two numbers from that report are worth deriving, because they set your alert thresholds. GTIG reports one actor making more than 500,000 calls in less than eight hours, which is at least **17.4 requests/second** sustained. GTIG also says the Jet Night token-routing service, powered by that actor's stolen keys, passed nearly $1M in token costs to the victim. Dividing the two gives roughly **$2 per call** (our inference, not a GTIG figure), which suggests long-context, premium-model calls rather than tiny chat turns. Both are trivially visible if you are counting tokens per key, and invisible if you are only watching request counts.

---

## 2. Layer one: observability at the gateway

The control plane and the data plane both matter. GTIG's detection guidance lists audit-log events per cloud (CloudTrail for `bedrock.amazonaws.com`, Azure Activity/Diagnostic logs under `Microsoft.CognitiveServices`, GCP Cloud Audit Logs for `aiplatform.googleapis.com`). Ship those to your SIEM. But audit logs lag and do not carry token counts reliably, so build your primary signal on a **gateway event** you control.

### The event schema

One event per model call. Everything downstream, whether metrics, rules, or Jev triage, derives from it.

```ts
// llm-call-event.ts
export interface LlmCallEvent {
  ts: string;                // ISO-8601
  key_id: string;            // hashed API key / virtual key id, never the secret
  principal: string;         // service account / user / workload identity
  tenant?: string;
  provider: string;          // bedrock | azure-openai | vertex | self-hosted
  model: string;
  route: string;             // chat.completions | invoke | stream ...
  src_ip_asn: number;
  src_country: string;
  user_agent: string;
  input_tokens: number;
  output_tokens: number;
  max_tokens_requested?: number;
  temperature?: number;
  status: number;            // HTTP status
  error_class?: string;      // Throttling | Validation | NotFound ...
  latency_ms: number;
  cost_usd: number;          // computed at the gateway, see section 3
}
```

Emit it as an OpenTelemetry log record or span attributes so the same event feeds metrics and traces. The point is that **cost is computed at write time**, so every dashboard and alert is in dollars and tokens rather than "requests".

### Deterministic detections (no model needed)

These map directly to GTIG's hunting guidance and should be plain rules:

- **Validation sweep**: `max_tokens_requested <= 1` (or `output_tokens <= 1`) from a key or source with a high distinct-key count, or a burst of such calls on a key that normally sends none.
- **Error-class spikes**: sudden `ThrottlingException`, `ValidationException`, or `ResourceNotFound` surges, which indicate model enumeration or quota probing.
- **Target scoping**: calls to list models, endpoints, or inference profiles from principals that never did before.
- **Control-plane changes**: key creation, service enablement, new deployments, new model agreements, and any write to logging configuration. GTIG advises treating unauthorized enablement as a critical security event. Do the same.
- **Source drift**: a key seen from a new ASN or country, or a CI-only key used from a residential or hosting ASN.

```sql
-- Example: validation-sweep candidates (adapt to your log store)
SELECT src_ip_asn,
       COUNT(DISTINCT key_id) AS keys_probed,
       COUNT(*)               AS calls
FROM llm_call_events
WHERE ts > now() - interval '10 minutes'
  AND max_tokens_requested <= 1
GROUP BY src_ip_asn
HAVING COUNT(DISTINCT key_id) >= 3 OR COUNT(*) >= 50;
```

Also patch the plumbing. The `/key/regenerate` escalation in the report targeted a model gateway, which means the gateway is part of your attack surface and belongs in your vulnerability management program.

---

## 3. Layer two: tokenomics as a security control

Tokens are the unit of spend, so tokens are the unit of defense. Treat each key as having a *monetary velocity* and a *budget*, the same way a payments system treats a card.

### 3.1 Price every call at the gateway

```ts
// pricing.ts: fill from your provider's current price sheet; do not hard-code from memory
const PRICE_PER_MTOK: Record<string, { in: number; out: number }> = {
  "model-a": { in: /* $ per 1M input tokens */ 0, out: /* $ per 1M output tokens */ 0 },
};

export const costUsd = (model: string, tin: number, tout: number) => {
  const p = PRICE_PER_MTOK[model];
  return (tin * p.in + tout * p.out) / 1_000_000;
};
```

### 3.2 Baseline per key, then alert on burn rate

Static thresholds fail because legitimate usage varies by key. Keep per-key rolling baselines of tokens/minute and USD/hour (an EWMA plus a robust deviation estimate), and alert on the **ratio** to baseline and on **absolute dollar velocity**.

Borrow the multi-window idea from SLO burn-rate alerting: a fast window catches the spike, and a slow window prevents flapping.

```
burn_fast  = usd(last 5 min)  / budget_per_5min
burn_slow  = usd(last 1 hour) / budget_per_hour

PAGE    if burn_fast > 10 and burn_slow > 2
TICKET  if burn_slow > 1 for 3 consecutive hours
```

For context: $100,000/day is about **$4,167/hour** or **$1.16/second**. If a key's normal envelope is, say, $20/hour, a 200x deviation is detectable in the first minute, hours before a daily billing export would show anything.

### 3.3 Enforce budgets in the request path

Detection tells you; budgets stop it. GTIG explicitly recommends granular quotas and spending caps, and scoping keys by model endpoint, environment, and usage limit. Implement them as a token bucket per key with a *dollar-denominated* hard cap:

```ts
// budget-guard.ts: runs before the upstream call
export async function guard(ev: Pick<LlmCallEvent, "key_id" | "model">, estIn: number, maxOut: number) {
  const est = costUsd(ev.model, estIn, maxOut);          // worst-case cost of THIS call
  const spent = await store.incrByFloat(`spend:${ev.key_id}:${hourBucket()}`, 0);
  const cap = await store.get(`cap:${ev.key_id}`);       // per-key hourly hard cap
  if (spent + est > cap) throw new BudgetExceeded(ev.key_id);
  await store.incrByFloat(`spend:${ev.key_id}:${hourBucket()}`, est); // reserve
}
// after the response, reconcile reserve -> actual cost
```

Three design details matter:

- **Reserve worst case, reconcile actual.** Use `max_tokens` and measured input size to reserve spend before the call. This stops one long-context call from blowing a cap.
- **Cap the blast radius per key, not just per account.** A stolen key should be able to burn one key's hourly cap, never the whole account's budget. The $1M incidents are what happens when the cap is account-wide, or absent.
- **Fail closed on key creation.** New keys start with a low cap that a human raises. This neutralizes the "create more keys and scale" step.

### 3.4 The attacker's economics are your detection surface

LLM Mixers resell access for profit, which means the attacker wants **maximum tokens per stolen key per unit time**, before revocation. That creates signatures you can price:

- tokens per request at the high end of context limits, repeatedly;
- premium-model skew compared with the key's history;
- flat, machine-regular request cadence (no diurnal curve);
- output-heavy completions with no downstream consumer in your product (no matching user session).

None of these needs an LLM to compute. They need the event schema above and a baseline.

---

## 4. Layer three: where Jev fits (and where it does not)

Rules and baselines catch volume anomalies. They are weak at a question like: *"This key's volume is only 3x normal. Is this still our support bot, or is someone running their own workload through it?"* GTIG notes that earlier abuse involved manipulating customer-support chatbots for unauthorized tasks such as code generation, and later campaigns exposed endpoints rebranded under personas such as `D1337-KAIROS`. That is a semantic question about content and purpose.

This is the shape of problem Jev was designed for. Jev is TypeSafe's decision model: you give it a **state** (a string or JSON object) and a set of **typed questions**, and it returns typed answers with probabilities. It does not generate text or explain itself. Per its docs, the primitives include `Choice` (classification with a probability distribution), `Noul` (probability a yes/no proposition is true), `Score` (ordinal rating), and, as of Jev 1.14, `Box`. Most calls complete in roughly 100 ms, and Jev is trained for calibrated probabilities, which is what makes threshold-based routing safe.

In the internal Jev use-case list this layer combines Guardrails (02), Classification (07), and Prioritization (09), and it is the security counterpart of the observability triage engine from the earlier Jev series. (That taxonomy comes from our own series, not from TypeSafe's public docs.)

### 4.1 Division of labor

| Concern | Owner | Why |
|---|---|---|
| Token counts, cost, rate, burn ratio | Code | Arithmetic; Jev is not for computation |
| Hard caps, revocation, allow/deny | Code | Deterministic control flow must stay in code |
| "Is this traffic on-purpose for this app?" | Jev | Semantic judgment over unstructured input |
| "Which kind of abuse does this look like?" | Jev | Classification with a probability distribution |
| "How urgent is this for the on-call?" | Jev | Prioritization/score feeding routing |

### 4.2 Decompose into narrow, atomic questions

Jev works best when a broad judgment is split into small independent questions evaluated against one shared state. Build the state from a **compact, redacted summary** of a key's recent window rather than raw prompts: counts, token stats, a handful of sampled prompt prefixes, route, and the declared purpose of the key.

```ts
// Tested against the live Jev API (jevascript 0.1.1, model jev-1.13.0, 2026-11-08).
import { choose, is, score } from "jevascript";

const state = {
  key_purpose: "customer-support chatbot, English, product Q&A only",
  window: "last 15 min",
  stats: { calls: 4120, p50_in_tokens: 61000, p50_out_tokens: 2800, distinct_src_asn: 6 },
  sampled_prompt_prefixes: [ /* redacted/truncated samples */ ],
};

// Noul questions are propositions; `detailed: true` returns { value, probability }.
const questions = {
  on_purpose: is("This traffic is consistent with the declared key_purpose.", { detailed: true }),
  codegen_or_roleplay: is(
    "The sampled prompts show code generation, roleplay personas, or resale-style bulk usage unrelated to key_purpose.",
    { detailed: true },
  ),
  // Choice options are described, not just named.
  abuse_pattern: choose(
    {
      normal: "Traffic matches the declared purpose of the key.",
      validation_sweep: "Many tiny calls (e.g. max_tokens=1) that only check whether the key works.",
      resale_proxying: "Varied third-party workloads relayed through the key, often with persona or reseller markers.",
      agentic_workload_hijack: "Long, tool-heavy or code-generation agent loops unrelated to the key's purpose.",
      unclear: "The evidence does not support any of the other options.",
    },
    { instructions: "Which best describes the traffic?", detailed: true },
  ),
  // Score levels are 2 to 10 described situations, 0-indexed (not a count).
  urgency: score("How urgent is it to revoke this key now?", {
    levels: [
      "No action needed; traffic is expected.",
      "Worth a look during business hours.",
      "Contain soon: lower the cap and open a ticket.",
      "Contain now: likely abuse that is burning budget.",
      "Revoke now: clear abuse with severe ongoing spend.",
    ],
    detailed: true,
  }),
};
```

### 4.3 Route on probabilities, not verdicts

The decision stays in code, and uncertainty becomes an explicit branch.

```ts
const r = await semantic(state).batch(questions); // one request, four typed answers

const offPurpose = 1 - r.on_purpose.probability;
const pattern    = r.abuse_pattern;                  // { value, probabilities, confidence }

if (burnFast > 10 && offPurpose > 0.9 && pattern.confidence > 0.8) {
  await revokeKey(keyId);                // automatic: strong volume signal AND strong semantic signal
  await page("key auto-revoked", r);
} else if (offPurpose > 0.6 || pattern.value === "unclear") {
  await lowerCap(keyId, 0.1);            // contain first
  await openTicket("review key", r);     // human decides
} else {
  record(r);                             // no action, keep the evidence
}
```

Note what the model is *not* allowed to do: it never revokes anything on its own, and revocation requires an independent deterministic signal (burn rate) as well. Jev raises or lowers confidence, and the circuit breaker in code makes the call. Datadog's Jev post advises measuring Jev's agreement with human reviewers and its repeatability on your own traffic before relying on it, so calibrate against human reviewers' decisions on your own traffic before you enable any automatic action.

### 4.4 Keep it out of the hot path

Jev is fast, but it is still a network call, and you do not want the defense to become the outage. Run it **out-of-band** on a stream of already-flagged windows. This is also how Datadog's Jev-based evaluation integration works: evaluations execute asynchronously against spans, and results are submitted back to the observability backend. Hard caps and the deterministic rules guard the synchronous path; Jev works the queue behind them.

### 4.5 Observe the observer: the tokenomics of the defense itself

Jev does not generate text, so its cost is driven by **input tokens**. That makes the defense cheap but not free, and unbounded if you feed it every call. Control it:

- **Prefilter**: send only windows that tripped a cheap rule (burn ratio > 2, new ASN, sweep pattern, control-plane change). Most keys on most days never reach Jev.
- **Summarize**: one compact state per key-window, not one per request.
- **Cache**: identical states hit the cache; jevascript reports a `cached` flag per evaluation.
- **Instrument it**: jevascript emits one event per evaluation with latency, request count, token usage, cache status, and the definition that produced it. Wire it into the same metrics pipeline:

```ts
import { createSemantic, jev } from "jevascript";

export const semantic = createSemantic({
  provider: jev({ model: "jev-1.13.0" }),   // pinned; reads JEV_API_KEY from the environment
  observability: {
    onEvaluation(e) {
      metrics.histogram("defense.jev.latency_ms", e.latencyMs, { op: e.operation, metric: e.metric?.name });
      metrics.counter("defense.jev.requests", e.requestCount);
      metrics.counter("defense.jev.input_tokens", e.usage?.inputTokens ?? 0);
      if (e.cached) metrics.counter("defense.jev.cache_hits");
    },
  },
});
```

Then add one meta-alert: **defense spend per day versus the loss it prevented.** If the triage layer's token bill ever approaches a meaningful fraction of the burn it intercepts, your prefilter is too loose. Pin the model version (for example `jev-1.13.0` rather than `latest`) so thresholds calibrated against one release do not silently shift under you, and group dashboards by definition `name@version`.

---

## 5. Putting it together

```
 clients ─▶ LLM gateway ──────────────▶ upstream model
              │  (1) guard(): per-key $ cap, reserve worst case      ← synchronous, deterministic
              │  (2) emit LlmCallEvent (tokens, $, key, ASN, ...)
              ▼
        event stream ─▶ metrics: tokens/min, $/h, burn_fast, burn_slow
              │              │
              │              ├─▶ deterministic rules (max_tokens=1 sweeps, error spikes,
              │              │      control-plane + logging-config changes)
              │              ▼
              │        prefilter: "is this window worth a judgment call?"
              │              │
              │              ▼
              │        Jev: on_purpose / abuse_pattern / urgency   ← async, out-of-band
              │              │
              ▼              ▼
        SIEM + audit logs   router: auto-revoke │ lower cap + ticket │ log
        (CloudTrail, Azure, GCP)
```

### Mapping to GTIG's mitigation list

| GTIG recommendation | Where it lands here |
|---|---|
| Budget alerts, granular quotas, spending caps | Section 3: per-key dollar caps, burn-rate alerts |
| Granular API key scopes | Per-key model/env scoping; low default caps on new keys |
| Validation sweep monitoring (`max_tokens=1`) | Section 2 deterministic rule |
| Monitor native cloud audit logs | Section 2: ship audit events; alert on enablement and logging-config changes |
| Gateway patch management (LiteLLM, CVE-2026-40217) | Treat the gateway as production attack surface |
| Credential hygiene, secret scanning, MFA, least privilege | Prevention: still the first line; this stack is the safety net for when it fails |
| Shadow LLM discovery | Alert on calls that bypass the gateway (provider-side usage with no matching gateway event) |

The last row is a free win: **reconcile provider-side usage against gateway events.** Any provider spend without a matching gateway event is, by definition, traffic that did not come through your control point.

---

## 6. Rollout checklist

1. Put a gateway in front of every model provider and emit `LlmCallEvent` with cost computed at write time.
2. Set a per-key hourly dollar cap on every key. New keys start low.
3. Turn on the deterministic rules: sweeps, error spikes, enablement, logging-config changes, source drift.
4. Build per-key baselines and the fast/slow burn-rate alerts.
5. Add the provider-vs-gateway reconciliation job.
6. Add Jev triage on prefiltered windows, out-of-band, with state summaries (not raw prompts) and a pinned model version.
7. Calibrate against human reviewer decisions for a few weeks before enabling any automatic revocation, and require a deterministic signal alongside the model's before auto-revoking.
8. Instrument the defense (`onEvaluation`) and alert on its own spend.

## Caveats

- The Jev snippets in section 4 were run against the live Jev API on 2026-11-08 with `jevascript` 0.1.1 and the model pinned to `jev-1.13.0`; the raw wire shapes were cross-checked with `typesafe_sdk` 0.7.5. The routing thresholds and the example states are placeholders: in our test windows the automatic-revoke branch never fired, because no live window met all three conditions at once (one high-burn window scored off-purpose 0.96 but pattern confidence 0.70; another scored pattern confidence 0.85 but off-purpose 0.82), so that branch was only exercised with jevascript's mock provider. Calibrate thresholds on your own traffic.
- Price tables, thresholds, and caps are placeholders: set them from your provider's current price sheet and your own baselines.
- The "$2 per call" figure is our inference from two GTIG numbers, and "17.4 requests/second" is a lower bound derived from "500,000 calls in less than eight hours"; GTIG states neither directly.
- This stack detects and limits abuse after credentials are compromised. It does not replace credential hygiene, secret scanning, least privilege, and phishing-resistant MFA, which remain GTIG's first recommendations.

## References

- Google Threat Intelligence Group, *The Rise of LLMJacking: How Threat Actors Monetize Compromised Cloud and Model Resources*, Nov 2, 2026. https://cloud.google.com/blog/topics/threat-intelligence/llmjacking-monetizing-compromised-cloud-models
- Datadog, *Using TypeSafe's Jev for evals in Datadog Agent Observability*. https://www.datadoghq.com/blog/jev-evals-agent-observability/
- jevascript docs, *Observability*. https://jevascript.org/docs/observability
- TypeSafe docs, *System One* and *How to build with System One*. https://docs.typesafe.ai/concepts/system-one
