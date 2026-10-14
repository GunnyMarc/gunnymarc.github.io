---
title: "Jev-Powered Observability: Semantic Triage of Telemetry at Classification Speed"
date: 2026-10-14
permalink: /posts/2026/10/jev-powered-observability-semantic-triage/
render_with_liquid: false
tags:
  - llm
  - system one models
  - classifiers
  - ai architecture
  - inference
  - cost engineering
  - observability
  - opentelemetry
---
## Introduction

The Support Triage Engine (covered in the [nine-use-case deep dive](/posts/2026/10/jev-engineering-deep-dive-9-use-cases/)) solves a familiar problem: high-volume incoming signals, heterogeneous urgency, and finite human capacity to respond. The architecture is clean — field extraction, semantic classification, threshold evaluation, routing. It beats a benchmark LLM by 7.2× on throughput and 16× on cost per decision.

Production observability is the same problem. You just swapped support tickets for telemetry.

A modern service emitting to an OpenTelemetry Collector at 10,000 log lines per second is not meaningfully different from 10,000 support tickets per hour, except the timescales are compressed by three orders of magnitude and the cost of a missed critical signal isn't a churned customer — it's an outage. The same structural mismatch applies: LLMs are too slow and too expensive to classify every event in the stream. Static threshold alerting is too brittle. The label spaces — *is this an OOM signal? a cascade? a regression? noise?* — are predefined. Jev fits.

This article is a production engineering specification for applying Jev to application observability. It assumes familiarity with distributed tracing, the three pillars (logs, metrics, traces), OpenTelemetry, and alerting pipelines. It is not an introduction to observability. It is an architectural proposal for replacing, or augmenting, the semantic classification layer inside one.

---

## Why Observability Is a Classification Problem

Standard observability stacks make classification decisions constantly. They just do it badly:

**Static threshold alerting** (`p99_latency > 500ms → page`) encodes a single threshold for a highly context-dependent signal. The same p99 spike means different things during a traffic ramp, a downstream dependency failure, or a GC pause. False positive rates on static threshold alerts commonly run 60–80% in production, burning on-call engineer capacity on noise.

**Regex-based log pattern matching** (`ERROR.*connection refused`) is brittle by construction. It requires manual maintenance, misses semantic variants ("failed to reach", "timeout connecting to"), and generates no confidence signal — every match is treated as equivalent regardless of surrounding context.

**LLM-based log analysis** (feeding batches of log lines to a language model for semantic triage) solves the brittleness problem but introduces a new one: latency. A 1.8–8 second round-trip on each batch is incompatible with real-time alerting SLOs. And the cost profile — outputting tokens describing what it found — is misaligned with the task, which only requires a classification decision, not a paragraph.

All three approaches are attempting the same underlying operation: *given a telemetry event and its surrounding context, assign it to one of a finite set of categories and derive an action*. That is a classification task with a predefined label space. That is exactly what Jev is engineered to do.

---

## The Observability Label Space

Before any architecture, you must enumerate your label space. This is the foundational step the original use-case analysis identifies: **define your label space first**. Anti-pattern 1 in that analysis is using Jev for open-domain questions. Observability must not fall into that trap.

For a typical production service, the observability classification space looks like this:

### Severity Classification (Multi-class, Pick-One)

```
[ critical | high | medium | low | noise ]
```

| Class | Operational Meaning |
|:------|:--------------------|
| `critical` | Service is degraded or down; page on-call immediately |
| `high` | Elevated error rate or latency; alert to Slack + ticket |
| `medium` | Anomalous but within tolerance; log for trend analysis |
| `low` | Expected variance; no action |
| `noise` | Chatter, heartbeats, health checks; suppress |

### Error Category Classification (Multi-class, Pick-One)

```
[ dependency_failure | resource_exhaustion | regression | config_error | external_client | unknown ]
```

| Class | Routing Target |
|:------|:--------------|
| `dependency_failure` | Alert the owning team of the downstream service |
| `resource_exhaustion` | Trigger autoscaling runbook |
| `regression` | Correlate against recent deployment; alert release owner |
| `config_error` | Alert platform/SRE team |
| `external_client` | Suppress from on-call; escalate to product if sustained |
| `unknown` | Escalate to human triage |

### Binary Flags (Parallel, Per-Event)

| Question | Threshold Action |
|:---------|:----------------|
| "Is this a novel error pattern not seen in the past 24 hours?" | > 0.7 → escalate even if severity is `low` |
| "Does this event indicate a cascading failure?" | > 0.6 → trigger blast-radius expansion protocol |
| "Is this correlated with a deployment in the past 30 minutes?" | > 0.65 → notify release owner; suppress general on-call page |
| "Is this event actionable by the on-call engineer?" | < 0.4 → suppress; route to async queue regardless of severity |
| "Does this event indicate data integrity risk?" | > 0.5 → immediate escalation path regardless of all other signals |

### Urgency Score (Continuous 0–1)

```
"Rate the operational urgency of this event given the service tier and current context."
```

This score is used as a final multiplier against the severity classification. A `medium` event with urgency score `0.92` during a peak traffic window may warrant the same response as a `high` event at 3 AM with urgency `0.31`.

---

## Architectural Design

### High-Level Data Flow

```
Telemetry Sources
  │
  ├─ Application Logs (structured JSON)
  ├─ Distributed Traces (OTLP spans)
  └─ Metrics (Prometheus / OTLP)
  │
  ▼
OpenTelemetry Collector
  │ (fanout pipeline)
  │
  ▼
Observability Classification Service  ◄─── Service Manifest Store
  │                                         (service dependencies,
  │   [Context Assembly]                     SLOs, ownership, tier)
  │   [Jev Batch Invocation]
  │   [Threshold Evaluation]
  │
  ▼
Action Router
  ├─ PagerDuty / OpsGenie (critical/high)
  ├─ Slack #alerts (medium)
  ├─ Runbook Trigger (resource_exhaustion, dependency_failure)
  ├─ JIRA Ticket (regression + deployment correlation)
  └─ Suppress → async analytics queue (low/noise)
```

### The Service Manifest

This pattern is lifted directly from Use Case 9 (Smart Home Control), where a device manifest grounds Jev's interpretation of a voice command. In observability, the **Service Manifest** serves the same function: it gives Jev the context it needs to interpret an ambiguous signal correctly.

The manifest is a structured, pre-computed document injected into the Jev context alongside each telemetry event. It contains:

```json
{
  "service_id": "payment-processor",
  "tier": "P0",
  "dependencies": ["postgres-primary", "redis-session", "fraud-api"],
  "slo_targets": {
    "p99_latency_ms": 300,
    "error_rate_pct": 0.1,
    "availability_pct": 99.95
  },
  "recent_deployments": [
    { "sha": "a3f9d12", "deployed_at": "2026-10-14T11:42:00Z", "deployer": "ci-pipeline" }
  ],
  "current_traffic_percentile": 87,
  "known_noisy_patterns": [
    "health check connection reset",
    "redis NOAUTH during startup"
  ]
}
```

This manifest is fetched from a lightweight service registry at classification time and cached with a 30-second TTL. The cache hit rate in practice exceeds 99.9% — service manifests change rarely within any 30-second window.

The manifest solves a critical ambiguity problem: a `connection refused` to `redis-session` from `payment-processor` (P0, SLO 99.95%) is not the same signal as `connection refused` to `redis-session` from `analytics-worker` (P3, best-effort). Same log line; different classification. Without the manifest, Jev cannot distinguish them. With it, the severity label space is grounded.

---

## Context Assembly

The input to Jev is assembled from three sources: the raw telemetry event, a windowed event buffer, and the service manifest.

### Windowed Event Buffer

Raw telemetry events are not classified in isolation. A single `ERROR` log line is almost always insufficient context. Jev receives a **sliding window** — the last *N* events from the same service, trace, or pod — as part of its context.

Window configuration:

| Signal Type | Window Size | Window Strategy |
|:-----------|:------------|:----------------|
| Log lines | Last 10 lines from same pod | Sliding, 60s TTL |
| Trace spans | All spans from same trace ID | Trace-bounded |
| Metrics anomaly | Last 5 metric samples | Tumbling, 30s interval |
| Exception | Stack trace + last 3 preceding log lines | Event-bounded |

The windowing is implemented as a per-pod, per-trace ring buffer in Redis with a 60-second expiration. Every event is written to the buffer before it is sent to the classification service. The classification call for event *N* reads events *N-10* through *N-1* from the buffer.

This is the same pattern as the Video Moment Scoring use case (30-second sliding windows over transcript segments). The core insight is identical: a signal's classification depends on its neighbors. Jev does not know this automatically; your context assembly must make it explicit.

### Context String Construction

The assembled context string fed to Jev is structured plaintext — not raw JSON. Jev is a semantic classifier, not a JSON parser. The context must be written in a way that enables semantic interpretation.

```
SERVICE: payment-processor (tier: P0)
SLO: p99 < 300ms, error_rate < 0.1%, availability 99.95%
TRAFFIC: currently at 87th percentile
RECENT DEPLOY: a3f9d12 deployed 12 minutes ago by ci-pipeline
DEPENDENCIES: postgres-primary [OK], redis-session [UNKNOWN], fraud-api [OK]

RECENT EVENTS (last 60s, same pod):
  [11:49:51] INFO  transaction initiated: amount=4200, currency=USD
  [11:49:51] INFO  fraud-api check: result=PASS latency=34ms
  [11:49:52] ERROR connection to redis-session refused: ECONNREFUSED 10.0.1.22:6379
  [11:49:52] ERROR session lookup failed: redis unavailable
  [11:49:52] ERROR payment processing failed: session_id=null
  [11:49:52] WARN  retrying session fetch (attempt 1/3)
  [11:49:53] ERROR redis-session connection refused (attempt 2)
  [11:49:53] ERROR redis-session connection refused (attempt 3)
  [11:49:53] ERROR max retries exceeded; returning 503 to client

CURRENT EVENT:
  [11:49:53] CRITICAL circuit breaker opened: redis-session; downstream: payment flow
```

This context string is deterministically generated by the Context Assembly component and cached per event batch.

---

## Jev Question Batch Design

This is the highest-leverage engineering decision in the system. Recall the core principle from the original analysis: **batch everything**. A single Jev call with *N* questions is far more efficient than *N* Jev calls with one question each. Anti-Pattern 2 in the source analysis explicitly prohibits single-question invocations.

The full question batch for observability:

```json
{
  "context": "<assembled context string above>",
  "questions": [
    {
      "id": "severity",
      "type": "multi_class",
      "question": "What is the operational severity of the current event?",
      "options": ["critical", "high", "medium", "low", "noise"]
    },
    {
      "id": "error_category",
      "type": "multi_class",
      "question": "What category best describes the root cause of this event?",
      "options": [
        "dependency_failure",
        "resource_exhaustion",
        "regression",
        "config_error",
        "external_client",
        "unknown"
      ]
    },
    {
      "id": "is_novel",
      "type": "binary",
      "question": "Is the pattern of errors in this event window novel or unexpected given the service's known noisy patterns?"
    },
    {
      "id": "is_cascade",
      "type": "binary",
      "question": "Does this event suggest the beginning or continuation of a cascading failure across multiple components?"
    },
    {
      "id": "deployment_correlated",
      "type": "binary",
      "question": "Is this failure pattern likely related to the recent deployment in the last 30 minutes?"
    },
    {
      "id": "is_actionable",
      "type": "binary",
      "question": "Is there a concrete action an on-call engineer can take right now to mitigate this event?"
    },
    {
      "id": "data_integrity_risk",
      "type": "binary",
      "question": "Does this event suggest risk to data integrity, consistency, or durability?"
    },
    {
      "id": "urgency_score",
      "type": "score",
      "question": "Rate the operational urgency of this event from 0.0 (can wait) to 1.0 (requires immediate human intervention), considering the service tier and current traffic load."
    }
  ]
}
```

**8 questions in a single round-trip.** The Jev response returns all 8 scores simultaneously at under 300ms end-to-end. Compare this to the cost of a single GPT-4o call for log analysis: 1.8–3 seconds, 20–80× the cost, and an unstructured text response that requires additional parsing.

---

## Threshold Evaluation Engine

The threshold evaluation layer consumes the Jev response and applies routing logic. This is **pure application code** — no model inference, no external API calls. Per the original analysis: *thresholds are business logic*.

```typescript
interface JevObservabilityResponse {
  severity: Record<'critical' | 'high' | 'medium' | 'low' | 'noise', number>;
  error_category: Record<string, number>;
  is_novel: number;
  is_cascade: number;
  deployment_correlated: number;
  is_actionable: number;
  data_integrity_risk: number;
  urgency_score: number;
}

function routeEvent(scores: JevObservabilityResponse, manifest: ServiceManifest): AlertAction[] {
  const actions: AlertAction[] = [];

  // Hard escalation overrides — data integrity risk bypasses all other logic
  if (scores.data_integrity_risk > 0.5) {
    return [{ type: 'page', channel: 'pagerduty', priority: 'P0', reason: 'data_integrity' }];
  }

  const topSeverity = argmax(scores.severity);          // e.g., "critical"
  const topCategory = argmax(scores.error_category);    // e.g., "dependency_failure"

  // Severity routing baseline
  if (topSeverity === 'critical' || topSeverity === 'high') {
    if (scores.is_actionable > 0.4) {
      actions.push({ type: 'page', channel: 'pagerduty', priority: manifest.tier });
    } else {
      // High severity but not actionable = alert, not page
      actions.push({ type: 'alert', channel: 'slack', reason: 'high_non_actionable' });
    }
  }

  // Novel signals escalate regardless of severity tier
  if (scores.is_novel > 0.7 && topSeverity !== 'noise') {
    actions.push({ type: 'alert', channel: 'slack', reason: 'novel_pattern', priority: 'elevated' });
  }

  // Cascade signals trigger blast-radius expansion
  if (scores.is_cascade > 0.6) {
    actions.push({ type: 'expand_blast_radius', check_services: manifest.dependencies });
  }

  // Deployment correlation reroutes to release owner instead of on-call
  if (scores.deployment_correlated > 0.65 && manifest.recent_deployments.length > 0) {
    const deployer = manifest.recent_deployments[0].deployer;
    actions.push({ type: 'alert', channel: 'slack', mention: deployer, reason: 'deployment_correlated' });
    // Remove PagerDuty page if deployment correlation is strong — release owner first
    return actions.filter(a => a.type !== 'page').concat({ type: 'alert', channel: 'pagerduty_low', reason: 'release_related' });
  }

  // Noise suppression
  if (topSeverity === 'noise' || (topSeverity === 'low' && scores.urgency_score < 0.3)) {
    return [{ type: 'suppress', queue: 'async_analytics' }];
  }

  // Urgency multiplier for medium events: elevate if urgency is very high
  if (topSeverity === 'medium' && scores.urgency_score > 0.85) {
    actions.push({ type: 'alert', channel: 'slack', reason: 'medium_high_urgency' });
  }

  // Category-specific runbook triggers
  if (topCategory === 'resource_exhaustion' && scores.error_category.resource_exhaustion > 0.7) {
    actions.push({ type: 'runbook', runbook_id: 'autoscale-trigger', service: manifest.service_id });
  }

  return actions.length > 0 ? actions : [{ type: 'log', queue: 'async_analytics' }];
}
```

**Key implementation notes:**

1. `argmax` is a utility that returns the highest-probability class from a multi-class Jev response. The probability itself is then compared to a secondary confidence threshold if needed.

2. The deployment correlation branch explicitly restructures the action list — it doesn't add to the default routing, it replaces the on-call page with a release-owner alert. This is domain logic that cannot live in the model. It lives here.

3. The `data_integrity_risk` check is an early-return hard override. Some signals are categorically non-negotiable regardless of other classification scores. These are explicitly encoded in threshold logic, not learned.

4. All thresholds are stored in a configuration file, not hardcoded in the function above. The function above reads from a `ThresholdConfig` object. This enables per-environment, per-service-tier threshold configuration without code deployment.

---

## Latency Budget Analysis

The classification pipeline must complete within a latency budget that is compatible with real-time alerting SLOs.

| Stage | Latency |
|:------|:--------|
| Event ingestion from OTLP Collector | ~2ms |
| Ring buffer write (Redis) | ~3ms |
| Service manifest fetch (cache hit) | ~1ms |
| Context string assembly | ~2ms |
| Jev API call (8 questions) | 100–280ms |
| Threshold evaluation (pure computation) | <1ms |
| Action routing dispatch | ~5ms (async) |
| **Total (cache hit path)** | **~115–295ms** |
| **Total (cache miss path)** | **~135–315ms** |

For a P0 service, this means a critical event from emission to PagerDuty page trigger completes in under 320ms end-to-end. The equivalent LLM-based pipeline (1.8s–8s Jev call + output parsing) would run 6–25× slower and would likely require async batching that introduces additional queue delay.

**Throughput characteristics:**

- At 300ms per Jev call (worst case), a single classification worker handles approximately 3.3 events/second per thread.
- With a worker pool of 20 threads (easily achievable on a single t3.large instance): ~66 events/second per instance.
- At 10,000 events/second (a mid-scale service), this requires a horizontally scaled pool of approximately 152 classification workers.
- Worker cost at this scale: significantly less than the equivalent LLM-based classification at the same volume.

The Jev cost model (billed per input token, no output token cost) inverts the cost structure of LLM-based alerting. Each 8-question Jev call processes roughly 600–900 input tokens (the context string dominates). At Jev's published price point, classifying every log event rather than a sampled 10% is economically viable. This matters: **alert systems that sample are systems that miss incidents**.

---

## Trace-Level Classification: Applying the Sliding Window Pattern

Log-line classification handles the most common case. For distributed traces, a different windowing strategy is required.

A distributed trace is a tree of spans. A single request through a microservice mesh may produce 40–200 spans. The failure signal often sits deep in the tree — a timeout on a database call three hops from the entry point. Naive log-line classification of the terminal error span lacks the context of the entire request path.

The trace-level classification pattern borrows from Use Case 8 (Video Moment Scoring with 30-second windows over transcript segments) but operates over the span tree rather than a time series:

### Trace Completion Detection

An OTLP span processor listens for root span completion signals. When the root span of a trace closes (or a 10-second timeout is reached for incomplete traces), the full span set is assembled.

### Span Tree Flattening

The span tree is serialized into a flat, human-readable context string ordered by time, with indentation indicating depth:

```
TRACE: 8f3a9b2c (entry: payment-processor /api/v1/charge)
DURATION: 847ms (SLO: 300ms) — EXCEEDED

├─ [  0ms] payment-processor: validate_request (12ms) [OK]
├─ [  12ms] payment-processor: fraud_check → fraud-api (36ms) [OK]
├─ [  48ms] payment-processor: session_lookup → redis-session (START)
│  └─ [  48ms] redis-session: GET session:user:88f3a (TIMEOUT after 300ms)
├─ [ 348ms] payment-processor: session_lookup RETRY 1 → redis-session (TIMEOUT 300ms)
├─ [ 648ms] payment-processor: session_lookup RETRY 2 → redis-session (TIMEOUT 198ms)
└─ [ 846ms] payment-processor: return 503 SERVICE_UNAVAILABLE
```

### Question Batch for Trace Classification

The trace-level question batch differs from the log-line batch:

```json
{
  "context": "<trace flattenedstring above>",
  "questions": [
    {
      "id": "root_cause_component",
      "type": "multi_class",
      "question": "Which component is the root cause of the trace failure?",
      "options": ["payment-processor", "fraud-api", "redis-session", "external_client", "unknown"]
    },
    {
      "id": "failure_type",
      "type": "multi_class",
      "question": "What type of failure does this trace represent?",
      "options": ["timeout", "error_response", "resource_exhaustion", "logic_error", "dependency_unavailable"]
    },
    {
      "id": "blast_radius_likely",
      "type": "binary",
      "question": "Is this trace failure likely affecting multiple concurrent users or just this request?"
    },
    {
      "id": "recovery_likely",
      "type": "binary",
      "question": "Based on the retry pattern, is the dependency likely recovering, or is the failure persistent?"
    },
    {
      "id": "latency_regression",
      "type": "binary",
      "question": "Does the latency pattern in this trace suggest a regression from expected baseline behavior?"
    },
    {
      "id": "user_impact_score",
      "type": "score",
      "question": "Rate the user-facing impact severity of this trace outcome from 0.0 (no visible impact) to 1.0 (complete service failure for the user)."
    }
  ]
}
```

The `recovery_likely` question is particularly useful: in the trace above, three consecutive timeouts with no improvement suggest a persistent failure rather than a transient hiccup. Jev's semantic interpretation of the retry pattern sequence ("TIMEOUT after 300ms" × 3) correctly assigns low probability to recovery. This causes the threshold engine to immediately escalate rather than wait for the standard "3 failed traces in 5 minutes" counter-based alert.

---

## Calibration and Threshold Management

### The Calibration Problem

Every threshold in the routing logic above is wrong when first deployed. "Critical" and "high" and "novel" all depend on what "normal" means for a specific service at a specific time. A `0.7` threshold on `is_novel` may be appropriate for a stable, mature service. For a service in active development with frequent intentional changes, `0.7` will generate constant false positives.

The calibration methodology:

**Phase 1: Shadow Mode (2 weeks)**

Deploy the classification pipeline in shadow mode. Every event is classified and logged, but no actions are taken. Collect the following distribution per service:
- Distribution of `severity` classifications per day/hour
- Distribution of `urgency_score` values
- Correlation between `is_novel` > 0.7 and human-confirmed incidents in post-mortems
- False positive rate of `deployment_correlated` against actual deployment timelines

**Phase 2: Threshold Grid Search**

For each threshold parameter, sweep a grid of values against the shadow mode dataset and compute:
- **Precision**: What fraction of `critical` classifications corresponded to actual incidents?
- **Recall**: What fraction of actual incidents were classified as `critical` or `high`?
- **False positive rate**: What fraction of pages would have been suppressed noise?

The objective function is domain-specific. For a P0 payment service, you optimize for recall at the cost of precision — you accept more noise pages to ensure zero missed incidents. For a P3 analytics service, you optimize for precision — minimize on-call interruptions even if some incidents are delayed in detection.

**Phase 3: Active Deployment with Feedback Loop**

The threshold engine logs every routing decision alongside the Jev scores that drove it. When an on-call engineer resolves an incident, the resolution classification (true positive, false positive, false negative) is written back to an incident feedback store. This store feeds a weekly calibration job that adjusts thresholds using logistic regression over the scored examples — not Jev, not an LLM, just a standard binary classifier over your own accumulated ground truth.

This is Jev's anti-pattern avoidance in practice: **Anti-Pattern 3** from the original analysis — using Jev for high-stakes autonomous decisions without audit trails — is directly addressed here. Every classification decision is logged, every routing action is stored, every on-call disposition is captured. The audit trail is first-class.

---

## What Jev Cannot Do in This Context

Applying the "What Jev Fails At" framework from the original analysis to the observability domain:

### Novel Failure Mode Detection

Jev requires a predefined label space. If a new class of failure emerges — one that doesn't map to `dependency_failure`, `resource_exhaustion`, `regression`, `config_error`, or `external_client` — Jev will force it into the closest existing class and return scores that may be weakly confident across the board. The `unknown` class in the error category schema is the escape hatch: when no class exceeds a confidence threshold (e.g., all options below 0.35), the event is routed to human triage regardless of other signals. This is correct behavior, not a failure — but teams must design the `unknown` class explicitly and monitor its rate. A sustained increase in `unknown` classifications is a signal that the label space needs expansion.

### Root Cause Analysis

Jev can classify that a failure is likely a `dependency_failure` with `deployment_correlated > 0.7`. It cannot explain *why* the deployment caused the dependency failure, *what changed* in the deployment that affected the dependency, or *how* to fix it. That reasoning layer belongs to an LLM, a runbook, or an engineer. Jev's role is to route the right signal to the right place at the right time. It is not the analyst.

### Continuous Metric Regression Detection

Time-series anomaly detection (e.g., "is this p99 latency trend a statistically significant regression from the 7-day baseline?") is a regression problem over a continuous signal, not a semantic classification problem. Jev should not be used for this. Standard anomaly detection methods (ARIMA, Prophet, or simpler z-score against rolling baseline) are the appropriate tool. Jev enters downstream: after an anomaly detection system fires a candidate signal, Jev classifies whether that signal merits an alert.

### High-Cardinality Entity Attribution

"Which of our 10,000 microservices is responsible for this aggregate latency increase?" is a search problem across a high-cardinality space, not a fixed-class classification. Jev's label space must be predefined and reasonably sized (generally under 50 options per question for reliable discrimination). Service attribution at scale requires different tooling — distributed tracing with trace sampling and aggregation, not Jev.

---

## Integration with Existing Observability Tooling

### OpenTelemetry Collector Pipeline

The classification service is deployed as a sidecar or as a standalone service consuming from an OTLP exporter. The Collector's pipeline configuration routes a copy of the logs/traces pipeline to the classification service via an OTLP exporter or a Kafka topic.

```yaml
# otel-collector-config.yaml (relevant excerpt)
exporters:
  otlp/backend:
    endpoint: "jaeger:4317"
  kafka/classification:
    brokers: ["kafka:9092"]
    topic: "telemetry-classification-input"
    encoding: otlp_proto

service:
  pipelines:
    logs:
      exporters: [otlp/backend, kafka/classification]
    traces:
      exporters: [otlp/backend, kafka/classification]
```

The classification service consumes from the `telemetry-classification-input` Kafka topic. This provides backpressure, replay capability (for recalibration), and decoupling from the hot path.

### Prometheus / Alertmanager Integration

For metric-based signals, the classification service exposes a webhook endpoint that Alertmanager can POST to instead of (or in addition to) its default routing tree:

```yaml
# alertmanager.yml
route:
  receiver: 'jev-classification-webhook'
  routes:
    - match:
        alertname: HighErrorRate
      receiver: 'jev-classification-webhook'

receivers:
  - name: 'jev-classification-webhook'
    webhook_configs:
      - url: 'http://jev-classifier:8080/classify/metric-alert'
        send_resolved: true
```

Alertmanager fires the webhook with the full alert payload. The classification service enriches it with the service manifest and current event window, runs the Jev classification, and either suppresses the alert (returning a 200 to Alertmanager) or re-fires it to the actual destination (PagerDuty / Slack) with enriched severity and category metadata.

This allows Jev to operate as a semantic filter layer within an existing Alertmanager-based alerting topology without replacing any of the existing tooling.

---

## Cost Model and Operational Economics

### Per-Event Classification Cost

Assume a service emitting 1,000 log events per second to the classification pipeline. Context string length averages 700 tokens (windowed log context + service manifest + question batch).

At Jev's published price point (approximately $0.00006 per call, based on the ~16× cost advantage cited in the original analysis over Kimi's ~$0.001 per call):

| Metric | Value |
|:-------|:------|
| Events per second | 1,000 |
| Events per day | 86,400,000 |
| Jev calls per day | 86,400,000 |
| Estimated cost per call | ~$0.00006 |
| **Daily classification cost** | **~$5,184** |
| Equivalent LLM cost (16× higher) | ~$82,944/day |

At $5,184/day for full classification coverage of a high-volume service, the economics are aggressive but defensible for a P0 service — particularly when the alternative is sampling (which misses incidents) or static alerting (which generates noise that erodes on-call trust, which erodes incident response quality, which increases MTTR).

For lower-volume services (P1–P3), the cost scales linearly down. A service at 100 events/second costs approximately $518/day to classify fully. For a P1 service with a $100K MTTR cost per incident, the break-even is approximately one avoided incident every six months.

The more significant cost lever is **noise reduction**. If the classification pipeline suppresses 65% of events as `noise` or `low` urgency (a conservative estimate for mature services with noisy health checks and heartbeats), the effective cost per actionable classification drops by 65%. The Jev cost model — billed per input token, no output token cost — is aligned with this: suppressed events cost the same as escalated ones, but the value of suppressed events is recovering on-call engineer time and focus.

---

## Anti-Patterns Specific to Observability

Beyond the general anti-patterns in the original analysis, the observability context introduces several domain-specific failure modes:

**Anti-Pattern: Classifying without context windows.** A single log line almost never carries enough signal for accurate severity classification. A single `ERROR` in isolation will produce uninformative Jev scores. Always assemble windowed context.

**Anti-Pattern: Static service manifests.** A manifest that doesn't reflect current deployment state produces incorrect `deployment_correlated` scores. The manifest must include recent deployments with timestamps and must be invalidated on each new deploy event.

**Anti-Pattern: Over-trusting the `urgency_score` without calibration.** The urgency score is a continuous regression output from a classification model. It will be systematically miscalibrated for your specific service without domain-specific calibration data. Treat it as a relative ranking signal, not an absolute measure.

**Anti-Pattern: Using Jev for alert deduplication.** Deduplication is a structural problem (identical alerts within a time window) that should be solved with exact matching and fingerprinting, not semantic classification. Jev's semantic interpretation of "nearly identical" alerts is overkill and may incorrectly deduplicate alerts that are similar but represent distinct failure paths.

**Anti-Pattern: Single-question per event call.** This anti-pattern from the original analysis is especially costly in observability contexts because event volume is high. Running 8 sequential Jev calls per event instead of 1 batched call multiplies both latency and cost by 8×.

---

## Production Readiness Checklist

Before deploying the Jev observability classification pipeline to production:

- [ ] Label space is documented and version-controlled
- [ ] Service manifest schema is defined and registry is seeded for all services in scope
- [ ] Ring buffer (Redis) TTL is configured per signal type
- [ ] All thresholds are externalized to configuration — no hardcoded values in routing logic
- [ ] Shadow mode has been run for at least 2 weeks with results analyzed
- [ ] Per-service threshold profiles are configured (not one global profile)
- [ ] `unknown` class rate monitoring is alerting if unknown > 5% of classifications
- [ ] Jev API key is stored in secrets management, not in application config
- [ ] Circuit breaker is in place: if Jev API is unavailable, fall back to severity-based static routing
- [ ] Classification decisions are logged to an audit store with full Jev score vectors
- [ ] Incident feedback loop is implemented (on-call disposition → feedback store → weekly recalibration)
- [ ] Alertmanager / PagerDuty webhook endpoint is authenticated and rate-limited
- [ ] Horizontal scaling plan for classification workers is load-tested at 2× expected peak volume

---

## Conclusion

The observability classification problem and the customer support triage problem share the same structure because they are the same problem. High-volume incoming signals, predefined severity categories, threshold-based routing to human or automated responders. The only differences are the token vocabulary (log lines vs. support ticket prose), the latency SLA (sub-320ms vs. sub-second acceptable), and the cost of a missed critical event (production outage vs. churned customer).

Jev's sub-300ms latency and 16× cost advantage over LLMs make full-coverage telemetry classification economically viable for the first time. Sampling is a cost-reduction strategy that accepts the risk of missing incidents. With Jev, you do not have to make that trade-off.

The implementation complexity is not in the Jev integration — the API is simple and well-specified. The complexity is in the context assembly (windowing, manifest management), the threshold calibration methodology (shadow mode, precision/recall tuning per service tier), and the feedback loop (closing the loop from incident disposition to threshold recalibration).

Get the label space right. Batch your questions. Treat thresholds as configuration. Build the feedback loop.

Jev is a navigator, not a writer. In this application, what you need it to navigate is the difference between noise and an outage — and it will do it 300 milliseconds after the event emits.

---

*References:*
- *Jev by Typesafe AI — typesafe.ai*
- *[Jev: A Production Engineering Deep Dive into 9 Real-World Use Cases](/posts/2026/10/jev-engineering-deep-dive-9-use-cases/) — Marc Buraczynski, October 4, 2026*
- *OpenTelemetry Specification — opentelemetry.io*
- *Google SRE Book — Beyer et al., O'Reilly Media*
- *Alertmanager Configuration — prometheus.io/docs/alerting/latest/configuration*
