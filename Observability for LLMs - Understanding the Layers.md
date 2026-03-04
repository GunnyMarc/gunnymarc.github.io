---
---

# Observability for LLMs: Understanding the Layers

*A practical guide to monitoring, debugging, and optimizing Large Language Model applications in production -- with implementation examples for OpenTelemetry, AppDynamics APM, and Splunk Observability Cloud.*

---

## Table of Contents

1. [Introduction: Why Your LLM Needs a Check Engine Light](#introduction-why-your-llm-needs-a-check-engine-light)
2. [What is Observability and Why Does It Matter?](#what-is-observability-and-why-does-it-matter)
3. [The Restaurant Kitchen: An Analogy for LLM Pipelines](#the-restaurant-kitchen-an-analogy-for-llm-pipelines)
4. [Traces and Spans: The Backbone of Observability](#traces-and-spans-the-backbone-of-observability)
5. [The Five Layers of LLM Observability](#the-five-layers-of-llm-observability)
6. [Why Each Layer Matters: Debugging, Cost, and Drift](#why-each-layer-matters-debugging-cost-and-drift)
7. [Implementation with OpenTelemetry](#implementation-with-opentelemetry)
8. [Integration with AppDynamics APM](#integration-with-appdynamics-apm)
9. [Integration with Splunk Observability Cloud](#integration-with-splunk-observability-cloud)
10. [Component-Level Evaluation: Beyond Black-Box Testing](#component-level-evaluation-beyond-black-box-testing)
11. [Best Practices for Production LLM Observability](#best-practices-for-production-llm-observability)
12. [Conclusion](#conclusion)

---

## Introduction: Why Your LLM Needs a Check Engine Light

Imagine driving a car with no dashboard. No speedometer, no fuel gauge, no check engine light. You press the gas, the car moves, and everything seems fine -- until it doesn't. When the car breaks down on the highway, you have no idea why. Was it the engine? The transmission? Did you run out of oil? Without instruments, you're left guessing.

This is exactly the situation many organizations find themselves in after deploying Large Language Model (LLM) applications to production. The application receives a user's question, something happens in the middle, and an answer comes out the other end. When that answer is wrong, slow, or expensive, teams scramble to figure out why -- and they often can't.

Traditional software engineering solved this problem decades ago with **observability**: the practice of instrumenting systems so that their internal state can be understood from the outside. Web applications have had distributed tracing, metrics dashboards, and structured logging for years. But LLM applications introduce entirely new layers of complexity. A single request might flow through an embedding model, a vector database, a context assembly step, and finally the language model itself. Each of those steps can fail independently, each has its own latency profile, and each carries its own cost.

This article breaks down the **layers of observability** that production LLM systems require. We'll use everyday analogies to make the concepts accessible, then dive into concrete Python implementations using three major platforms: OpenTelemetry (the open standard), AppDynamics APM (Cisco's enterprise solution), and Splunk Observability Cloud. Whether you're a technical lead instrumenting a RAG pipeline or a product manager trying to understand why your AI feature is underperforming, these layers will give you the mental model to diagnose, optimize, and trust your LLM applications.

---

## What is Observability and Why Does It Matter?

**Observability** is the ability to understand what a system is doing on the inside by examining what it produces on the outside. In software, that means collecting three types of signals:

- **Traces** -- the end-to-end journey of a single request through your system.
- **Metrics** -- numerical measurements aggregated over time (latency, error rate, throughput).
- **Logs** -- timestamped records of discrete events ("user submitted query," "embedding model returned 1536 dimensions").

Together, these three signals form the **three pillars of observability**. Think of them as three different types of medical tests. A blood test (metrics) tells you aggregate health numbers. An MRI scan (traces) shows you the detailed internal structure of a single area. A patient's symptom diary (logs) provides a chronological record of events. No single test is sufficient; you need all three for a complete diagnosis.

For traditional web applications, observability is well-established. When a user clicks "Submit Order" on an e-commerce site, a trace follows that request through the API gateway, the inventory service, the payment processor, and the notification service. If the order fails, engineers can open the trace and see exactly which service failed and why.

LLM applications need the same treatment -- but with additional layers that traditional software doesn't have. When a user asks an AI assistant a question, the request doesn't just hop between microservices. It undergoes *transformations*: text becomes vectors, vectors become search results, search results become context, and context becomes a generated response. Each transformation is a potential point of failure, and each requires its own type of monitoring.

The stakes are high. Unlike a failed API call that returns an error code, an LLM can fail *silently*. It can hallucinate a confident-sounding answer that is completely wrong. It can use the wrong context and produce a plausible but irrelevant response. Without observability at every layer, these silent failures go undetected until a user complains -- or worse, acts on bad information.

---

## The Restaurant Kitchen: An Analogy for LLM Pipelines

To understand why LLM observability needs multiple layers, imagine a high-end restaurant kitchen.

A customer places an order: "I'd like the pan-seared salmon with seasonal vegetables." That order goes through several stations before a plate arrives at the table:

1. **The Host Stand (Query Intake)** -- The server writes down the order, noting any allergies or special requests. If the server mishears the order, everything downstream goes wrong.

2. **The Prep Station (Embedding)** -- The ingredients are washed, measured, and prepared. Raw ingredients are transformed into something the kitchen can work with. If the prep cook grabs the wrong fish, it doesn't matter how well the chef cooks it.

3. **The Walk-In Cooler (Retrieval)** -- The cook goes to the refrigerator and selects the specific ingredients needed for this dish. If the cooler is disorganized or the labels are wrong, the cook might grab tilapia instead of salmon.

4. **The Assembly Station (Context)** -- All the components are gathered onto one workstation: the fish, the vegetables, the sauce, the garnish. The chef reviews everything before cooking. If the plate is overcrowded or missing components, the final dish suffers.

5. **The Stove (Generation)** -- The chef cooks the dish. This is the most time-consuming and expensive step. Even with perfect ingredients, a distracted chef can burn the fish.

Now, here's the critical insight: if the customer sends the dish back because it "doesn't taste right," the head chef needs to figure out *which station* made the mistake. Was it a bad ingredient from prep? The wrong cut from the cooler? Too much sauce at assembly? Or did the cook simply over-season it?

Without cameras and thermometers at each station, the head chef is left guessing. That's what running an LLM application without layer-by-layer observability feels like.

In our analogy, the **trace** is the complete life of that single order -- from the moment the customer spoke to the moment the plate arrived. The **spans** are the individual station operations: host, prep, retrieval, assembly, cooking. Each span has a start time, an end time, and metadata about what happened (which ingredient was pulled, what temperature the stove was set to, how long the cook waited for a burner).

---

## Traces and Spans: The Backbone of Observability

Let's formalize the restaurant analogy into engineering terms.

A **trace** is a record of the complete journey of a single request through your system. When a user asks your RAG application "What is retrieval-augmented generation?", a unique **Trace ID** is generated. Every operation that happens as part of fulfilling that request carries this same Trace ID, linking them together like beads on a string.

A **span** is a single named operation within a trace. Each span records:

- **Name** -- what operation this is ("embed_query," "vector_search," "llm_generate").
- **Start time** and **end time** -- how long this operation took.
- **Attributes** -- key-value metadata (model name, token count, relevance score).
- **Status** -- did this operation succeed or fail?
- **Parent span** -- which operation triggered this one?

The parent-child relationship between spans creates a tree structure. The **root span** is the overall request. Its children are the major pipeline steps. Those children might have children of their own (for example, the retrieval span might contain child spans for "encode query" and "search index").

Here's what a trace looks like laid out as a timeline. Notice how the trace encompasses all spans, and each span occupies a distinct time window:

```
Time (ms)   0       50      100     200     250     300          520
            |       |       |       |       |       |            |
Trace    [================================================================]
         trace_id: a]7f2-bc91-4e03

Query    [------]
         0-40ms    "What is RAG?"

Embed            [--------]
                 45-105ms   model: text-embedding-3-small

Retrieve                   [-----------]
                           110-210ms   top_k: 5, results: 5

Context                                [-----]
                                       215-260ms   tokens: 3,847

Generate                                      [========================]
                                              265-520ms   model: gpt-4o
                                              input_tokens: 4,102
                                              output_tokens: 287
```

If your system processes 1,000 queries in an hour, you get 1,000 traces. Each trace contains five spans (in our RAG example), but they're all linked by their unique Trace ID. This means you can aggregate across traces to compute averages ("What's the median retrieval latency this week?") or drill into a single trace to debug a specific bad response ("Why did trace `a7f2-bc91` return nonsense?").

Think of it this way: if traces are individual patient visits to a hospital, spans are the steps in each visit -- check-in, triage, blood draw, doctor consultation, prescription. The hospital administrator can look at one visit in detail or analyze thousands of visits to find systemic bottlenecks.

---

## The Five Layers of LLM Observability

Now that we understand traces and spans, let's examine the five observability layers that a production RAG pipeline requires. Each layer corresponds to a span, and each captures distinct signals that the others cannot.

```
+===================================================================+
|  LAYER 5: GENERATION                                              |
|  The LLM produces a response                                     |
|  Monitor: input/output tokens, latency, cost, model, temperature  |
+===================================================================+
|  LAYER 4: CONTEXT ASSEMBLY                                        |
|  Retrieved documents + system prompt are merged                   |
|  Monitor: total token count, template version, truncation events  |
+===================================================================+
|  LAYER 3: RETRIEVAL                                               |
|  Vector database similarity search                                |
|  Monitor: top-k, relevance scores, result count, DB latency       |
+===================================================================+
|  LAYER 2: EMBEDDING                                               |
|  User query is converted into a vector                            |
|  Monitor: model name, dimensions, token count, API latency        |
+===================================================================+
|  LAYER 1: QUERY INTAKE                                            |
|  User submits their question                                      |
|  Monitor: raw input, timestamp, session ID, user metadata          |
+===================================================================+
```

### Layer 1: Query Intake

Every journey begins with a question. The **query span** captures the raw user input, a timestamp, session identifiers, and any metadata about the user or conversation history. This span is usually fast (a few milliseconds), but it's essential for two reasons. First, it anchors the trace -- everything that follows is a child of this span. Second, it preserves the original question before any transformation happens. If the final answer is wrong, you'll want to compare it against the exact input to understand whether the question was ambiguous, malformed, or perfectly clear.

Back to the restaurant: this is the host stand writing down the order. It's quick, but if the server writes down "steak" instead of "salmon," every subsequent station will execute flawlessly on the wrong dish.

### Layer 2: Embedding

The user's text query is now converted into a numerical vector -- a list of hundreds or thousands of numbers that represent the meaning of the query in a way that machines can compare. The **embedding span** tracks which model performed this conversion, how many tokens were processed, the dimensionality of the output vector, and how long the API call took.

This is the prep station transforming raw ingredients into something the kitchen can use. If the prep cook uses a dull knife (slow embedding API) or the wrong cutting technique (mismatched embedding model), everything downstream suffers. Monitoring this layer catches rate limits, model version changes, and latency spikes before they cascade.

### Layer 3: Retrieval

The vector goes to your **vector database** (Pinecone, Weaviate, Chroma, pgvector, etc.) for a similarity search. The database returns the top-k most relevant document chunks. The **retrieval span** records the number of results, their relevance scores, the search latency, and the specific documents retrieved.

This is the cook visiting the walk-in cooler. If the cooler is poorly organized (bad chunking strategy), if the labels are wrong (stale embeddings), or if the cook only grabs one item when they need five (wrong top-k value), the dish will suffer. Our experience -- and the broader industry's -- suggests that **retrieval is where most RAG problems hide**. Bad chunks, low relevance scores, and misconfigured similarity metrics are the silent killers of RAG quality. The retrieval span exposes all of it.

### Layer 4: Context Assembly

The retrieved document chunks are now assembled together with your system prompt and any conversation history into the final prompt that will be sent to the LLM. The **context span** records the total token count, which template was used, and whether any truncation occurred.

This is the assembly station where all components come together on one plate. If the plate is overcrowded (context exceeds the model's window), ingredients get removed, and the dish loses coherence. If a key ingredient is missing (important document chunk was dropped), the final output suffers. This span is your last chance to inspect *exactly* what the LLM will see before it generates a response.

### Layer 5: Generation

The LLM processes the assembled prompt and produces a response. The **generation span** is typically the longest and most expensive operation in the pipeline. It records the model used, input token count, output token count, latency, temperature setting, and any finish reason (did the model stop naturally, or was it cut off by a token limit?).

This is the stove -- the most time-consuming and expensive station. Even with perfect ingredients, a cook can burn the dish. Monitoring this span is critical for cost management (tokens directly translate to dollars), performance optimization, and detecting when a model version change affects output quality.

---

## Why Each Layer Matters: Debugging, Cost, and Drift

Having five layers of observability serves three distinct purposes.

### Debugging: Finding the Needle in the Haystack

Without span-level tracing, debugging an LLM application is like being told "the food was bad" with no further detail. You know the output was wrong, but you don't know if the problem was bad retrieval, bad context, or the LLM hallucinating.

With layer-by-layer spans, you can follow a systematic diagnostic process:

```
Response quality is poor. Where is the problem?
|
+-- Check QUERY span
|   Is the input clean and well-formed?
|   +-- NO --> Input validation / sanitization issue
|   +-- YES --> Move to next layer
|
+-- Check EMBEDDING span
|   Did the embedding complete normally?
|   +-- HIGH LATENCY --> API bottleneck or rate limiting
|   +-- ERROR --> Authentication / quota issue
|   +-- OK --> Move to next layer
|
+-- Check RETRIEVAL span
|   Are the retrieved documents relevant?
|   +-- LOW SCORES --> Bad chunking strategy or stale index
|   +-- EMPTY RESULTS --> Vector DB issue or index misconfiguration
|   +-- OK --> Move to next layer
|
+-- Check CONTEXT span
|   Is the assembled prompt correct?
|   +-- TOO LONG --> Context window exceeded, data truncated
|   +-- MISSING DATA --> Template bug or assembly error
|   +-- OK --> Move to next layer
|
+-- Check GENERATION span
    The LLM itself is the issue.
    +-- HALLUCINATION --> Tighten prompt constraints or lower temperature
    +-- HIGH COST --> Reduce max tokens or use a smaller model
    +-- SLOW --> Consider a faster model or streaming
```

This decision tree is only possible when each layer emits its own span with meaningful attributes. Without it, you're left with trial and error.

### Cost Tracking: Following the Money

LLM tokens cost money. Embedding API calls cost money. Vector database queries cost money. Span-level tracking lets you attribute costs to specific pipeline components.

You might discover that 70% of your spend is on generation (expected), but 20% is on embedding because you're re-embedding queries that were already embedded in a previous conversation turn. Or you might find that your retrieval step is pulling 20 chunks when 5 would suffice, inflating your context tokens and therefore your generation cost.

Without layer-level cost attribution, you only see the total bill. With it, you see exactly where optimization will have the biggest impact.

### Drift Detection: Catching Silent Degradation

AI systems degrade over time. What worked last month might not work today. Document indexes go stale. Embedding model providers push silent updates. LLM behavior shifts across versions. User query patterns change seasonally.

Span-level metrics let you catch drift early. If your retrieval relevance scores drop by 15% over two weeks, you know your index needs refreshing -- even if end-to-end output quality hasn't visibly degraded yet. If your embedding latency suddenly doubles, you know the provider changed something before your users start complaining about slow responses.

Think of it as the difference between annual physicals and continuous vital sign monitoring. The annual physical (end-to-end testing) catches problems after they've developed. Continuous monitoring (span-level metrics) catches the early warning signs.

---

## Implementation with OpenTelemetry

**OpenTelemetry** (OTel) is the open, vendor-neutral standard for observability instrumentation. It provides APIs and SDKs for generating traces, metrics, and logs that can be exported to any compatible backend. Using OTel means your instrumentation code isn't locked to a specific vendor -- you can switch from one observability platform to another by changing configuration, not code.

Here's how to instrument a RAG pipeline with all five observability layers using the OpenTelemetry Python SDK:

```bash
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc
```

```python
"""
RAG Pipeline with Full OpenTelemetry Instrumentation
Demonstrates all five layers of LLM observability.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode
import time

# ── Setup ──────────────────────────────────────────────────
# Create a resource that identifies this service.
resource = Resource.create({
    "service.name": "rag-pipeline",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

# Configure the tracer provider with an OTLP exporter.
# The endpoint can point to any OTel-compatible collector.
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("rag.pipeline", "1.0.0")


# ── Layer 1: Query Intake ──────────────────────────────────
def process_query(user_input: str, session_id: str) -> dict:
    """Full RAG pipeline with five instrumented layers."""

    with tracer.start_as_current_span("rag.query") as query_span:
        query_span.set_attribute("rag.query.text", user_input)
        query_span.set_attribute("rag.query.session_id", session_id)
        query_span.set_attribute("rag.query.timestamp", time.time())
        query_span.set_attribute("rag.query.char_count", len(user_input))

        # ── Layer 2: Embedding ─────────────────────────────
        with tracer.start_as_current_span("rag.embed") as embed_span:
            embed_span.set_attribute("gen_ai.system", "openai")
            embed_span.set_attribute(
                "gen_ai.request.model", "text-embedding-3-small"
            )

            query_vector = embed_query(user_input)

            embed_span.set_attribute(
                "rag.embed.dimensions", len(query_vector)
            )
            embed_span.set_attribute("rag.embed.token_count", 12)

        # ── Layer 3: Retrieval ─────────────────────────────
        with tracer.start_as_current_span("rag.retrieve") as retrieval_span:
            retrieval_span.set_attribute("rag.retrieve.top_k", 5)
            retrieval_span.set_attribute(
                "rag.retrieve.vector_db", "pinecone"
            )

            results = search_vector_db(query_vector, top_k=5)

            retrieval_span.set_attribute(
                "rag.retrieve.result_count", len(results)
            )
            if results:
                scores = [r["score"] for r in results]
                retrieval_span.set_attribute(
                    "rag.retrieve.top_score", max(scores)
                )
                retrieval_span.set_attribute(
                    "rag.retrieve.min_score", min(scores)
                )

        # ── Layer 4: Context Assembly ──────────────────────
        with tracer.start_as_current_span("rag.context") as context_span:
            context = assemble_context(user_input, results)

            context_span.set_attribute(
                "rag.context.total_tokens", context["token_count"]
            )
            context_span.set_attribute(
                "rag.context.num_chunks", len(results)
            )
            context_span.set_attribute(
                "rag.context.template_version", "v2.1"
            )
            # Flag if the context is dangerously close to
            # the model's limit.
            if context["token_count"] > 12000:
                context_span.set_attribute(
                    "rag.context.near_limit", True
                )
                context_span.add_event(
                    "context_warning",
                    {"message": "Context approaching token limit"},
                )

        # ── Layer 5: Generation ────────────────────────────
        with tracer.start_as_current_span("rag.generate") as gen_span:
            gen_span.set_attribute("gen_ai.system", "openai")
            gen_span.set_attribute("gen_ai.request.model", "gpt-4o")
            gen_span.set_attribute("gen_ai.request.temperature", 0.3)
            gen_span.set_attribute("gen_ai.request.max_tokens", 1024)

            response = call_llm(context["prompt"])

            gen_span.set_attribute(
                "gen_ai.usage.input_tokens",
                response["usage"]["prompt_tokens"],
            )
            gen_span.set_attribute(
                "gen_ai.usage.output_tokens",
                response["usage"]["completion_tokens"],
            )
            gen_span.set_attribute(
                "gen_ai.response.finish_reason",
                response["finish_reason"],
            )
            # Cost estimate: $2.50/1M input, $10.00/1M output
            # for gpt-4o.
            cost = (
                response["usage"]["prompt_tokens"] * 2.50 / 1_000_000
                + response["usage"]["completion_tokens"]
                * 10.00
                / 1_000_000
            )
            gen_span.set_attribute("rag.generate.cost_usd", cost)

        query_span.set_status(StatusCode.OK)
        return {"answer": response["text"], "trace_id": str(
            query_span.get_span_context().trace_id
        )}


# ── Placeholder functions (replace with real implementations) ──
def embed_query(text):
    return [0.1] * 1536  # Simulated 1536-dim vector

def search_vector_db(vector, top_k):
    return [
        {"id": f"doc_{i}", "score": 0.95 - i * 0.05, "text": f"..."}
        for i in range(top_k)
    ]

def assemble_context(query, results):
    chunks = " ".join(r["text"] for r in results)
    prompt = f"Context: {chunks}\n\nQuestion: {query}\nAnswer:"
    return {"prompt": prompt, "token_count": 4102}

def call_llm(prompt):
    return {
        "text": "RAG is a technique that...",
        "usage": {"prompt_tokens": 4102, "completion_tokens": 287},
        "finish_reason": "stop",
    }
```

The key insight in this code is the **nesting**. The `rag.query` span is the parent (root), and all other spans are its children. OpenTelemetry automatically propagates the Trace ID through the `start_as_current_span` context manager, so every span in a request shares the same trace. When you view this trace in a dashboard, you'll see the full tree structure and can drill into any individual layer.

The `gen_ai.*` attributes follow the [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/), ensuring that observability backends can render LLM-specific dashboards without custom configuration.

---

## Integration with AppDynamics APM

**AppDynamics** (part of Cisco's observability portfolio) provides enterprise application performance monitoring with automatic business transaction detection, anomaly detection, and root cause analysis. Modern AppDynamics deployments support OpenTelemetry ingestion, meaning you can send OTel-instrumented traces directly to the AppDynamics controller.

The approach: use the same OpenTelemetry SDK from the previous section, but configure the OTLP exporter to target the AppDynamics OTLP endpoint. AppDynamics maps OTel traces to its concept of **Business Transactions** (BTs), giving you both the vendor-neutral instrumentation and the enterprise analytics.

```bash
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc
```

```python
"""
RAG Pipeline exporting traces to AppDynamics via OTLP.
AppDynamics maps OpenTelemetry traces to Business Transactions.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
import os

# ── AppDynamics-specific configuration ─────────────────────
# These values come from your AppDynamics controller settings.
APPD_OTLP_ENDPOINT = os.getenv(
    "APPDYNAMICS_OTLP_ENDPOINT",
    "https://<your-controller>.saas.appdynamics.com:443",
)
APPD_API_KEY = os.getenv("APPDYNAMICS_API_KEY", "<your-api-key>")

resource = Resource.create({
    "service.name": "rag-pipeline",
    "service.namespace": "ai-applications",
    "service.version": "1.0.0",
    # AppDynamics uses these resource attributes to organize
    # services into tiers and applications.
    "appdynamics.controller.account": "your-account",
    "appdynamics.controller.application": "LLM-RAG-Service",
})

# ── Exporter targeting AppDynamics OTLP ingestion ──────────
# The API key is passed as a header for authentication.
exporter = OTLPSpanExporter(
    endpoint=APPD_OTLP_ENDPOINT,
    headers={"x-api-key": APPD_API_KEY},
)

provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("rag.pipeline.appdynamics", "1.0.0")


def handle_rag_request(user_input: str, session_id: str):
    """
    Each call creates a Business Transaction in AppDynamics.
    The root span name ('rag.query') becomes the BT name.
    Child spans appear as "Exit Calls" or "Service Endpoints"
    in the AppDynamics waterfall view.
    """
    with tracer.start_as_current_span("rag.query") as root:
        root.set_attribute("query.text", user_input)
        root.set_attribute("session.id", session_id)

        # Layer 2 -- AppDynamics shows this as a downstream
        # call with its own timing and error rate.
        with tracer.start_as_current_span("rag.embed") as span:
            span.set_attribute("gen_ai.request.model",
                               "text-embedding-3-small")
            vector = embed_query(user_input)

        # Layer 3 -- The retrieval span surfaces vector DB
        # latency in AppDynamics' "Slowest DB Calls" view.
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("db.system", "pinecone")
            span.set_attribute("rag.retrieve.top_k", 5)
            results = search_vector_db(vector, top_k=5)
            span.set_attribute("rag.retrieve.result_count",
                               len(results))

        # Layer 4
        with tracer.start_as_current_span("rag.context") as span:
            context = assemble_context(user_input, results)
            span.set_attribute("rag.context.total_tokens",
                               context["token_count"])

        # Layer 5 -- Generation latency and token cost are
        # visible per-BT in AppDynamics dashboards.
        with tracer.start_as_current_span("rag.generate") as span:
            span.set_attribute("gen_ai.request.model", "gpt-4o")
            response = call_llm(context["prompt"])
            span.set_attribute("gen_ai.usage.input_tokens",
                               response["usage"]["prompt_tokens"])
            span.set_attribute("gen_ai.usage.output_tokens",
                               response["usage"]["completion_tokens"])

        return response["text"]
```

What makes this valuable from an enterprise perspective is that AppDynamics automatically detects anomalies across your Business Transactions. If your `rag.retrieve` span starts taking 3x longer than its baseline on Tuesday afternoons, AppDynamics flags it and correlates it with infrastructure changes, deployment events, or upstream service degradation. You get the five layers of LLM observability wrapped in enterprise-grade anomaly detection and alerting.

In the AppDynamics Flow Map, your RAG pipeline appears as a chain: `rag.query` calls `rag.embed`, which calls `rag.retrieve`, and so on. Each link shows latency, throughput, and error rate. This visual representation is essentially the trace timeline we discussed earlier, but rendered automatically by the platform.

---

## Integration with Splunk Observability Cloud

**Splunk Observability Cloud** provides real-time monitoring and troubleshooting built natively on OpenTelemetry. Splunk distributes its own packaging of the OTel SDK (`splunk-opentelemetry`) that adds automatic instrumentation for common frameworks and pre-configured export to Splunk's backend.

The Splunk approach has a distinct advantage: because Splunk also provides log analytics (via Splunk Enterprise or Splunk Cloud Platform), you can correlate your LLM observability traces with application logs and infrastructure metrics in a single pane of glass. When your generation span shows high latency, you can pivot to the GPU utilization metrics of the machine running your model, or the error logs from your vector database -- all linked by the same Trace ID.

```bash
pip install splunk-opentelemetry opentelemetry-api opentelemetry-sdk
```

```python
"""
RAG Pipeline exporting traces to Splunk Observability Cloud.
Uses Splunk's OpenTelemetry distribution for streamlined setup.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
import os

# ── Splunk-specific configuration ──────────────────────────
# Obtain from: Splunk Observability > Settings > Access Tokens
SPLUNK_ACCESS_TOKEN = os.getenv("SPLUNK_ACCESS_TOKEN")
SPLUNK_REALM = os.getenv("SPLUNK_REALM", "us0")

# Splunk's OTLP ingest endpoint follows a predictable pattern.
SPLUNK_OTLP_ENDPOINT = (
    f"https://ingest.{SPLUNK_REALM}.signalfx.com/v2/trace/otlp"
)

resource = Resource.create({
    "service.name": "rag-pipeline",
    "deployment.environment": "production",
    "service.version": "1.0.0",
    # Splunk uses this to group services in APM.
    "splunk.distro.version": "1.0.0",
})

# ── Exporter targeting Splunk's OTLP HTTP endpoint ─────────
exporter = OTLPSpanExporter(
    endpoint=SPLUNK_OTLP_ENDPOINT,
    headers={"X-SF-TOKEN": SPLUNK_ACCESS_TOKEN},
)

provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("rag.pipeline.splunk", "1.0.0")


# ── Instrumented RAG Pipeline ──────────────────────────────
def process_rag_query(user_input: str, session_id: str):
    """
    Traces appear in Splunk APM under the 'rag-pipeline'
    service. Each span is visible in the trace waterfall.
    Span tags become indexed fields for filtering and
    alerting in Splunk dashboards.
    """
    with tracer.start_as_current_span("rag.query") as root:
        root.set_attribute("rag.query.text", user_input)
        root.set_attribute("rag.query.session_id", session_id)

        # Layer 2: Embedding
        with tracer.start_as_current_span("rag.embed") as span:
            span.set_attribute(
                "gen_ai.request.model", "text-embedding-3-small"
            )
            vector = embed_query(user_input)
            span.set_attribute("rag.embed.dimensions", len(vector))

        # Layer 3: Retrieval
        # In Splunk, you can create detectors (alerts) on
        # span attributes. Example: alert when
        # rag.retrieve.top_score drops below 0.7.
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("db.system", "chromadb")
            span.set_attribute("rag.retrieve.top_k", 5)
            results = search_vector_db(vector, top_k=5)
            scores = [r["score"] for r in results]
            span.set_attribute("rag.retrieve.result_count",
                               len(results))
            span.set_attribute("rag.retrieve.top_score",
                               max(scores) if scores else 0.0)
            span.set_attribute("rag.retrieve.avg_score",
                               sum(scores) / len(scores)
                               if scores else 0.0)

        # Layer 4: Context Assembly
        with tracer.start_as_current_span("rag.context") as span:
            context = assemble_context(user_input, results)
            span.set_attribute("rag.context.total_tokens",
                               context["token_count"])
            span.set_attribute("rag.context.template_version",
                               "v2.1")

        # Layer 5: Generation
        # Splunk Tag Spotlight automatically surfaces which
        # attribute values correlate with errors or latency.
        with tracer.start_as_current_span("rag.generate") as span:
            span.set_attribute("gen_ai.request.model", "gpt-4o")
            span.set_attribute("gen_ai.request.temperature", 0.3)
            response = call_llm(context["prompt"])
            span.set_attribute(
                "gen_ai.usage.input_tokens",
                response["usage"]["prompt_tokens"],
            )
            span.set_attribute(
                "gen_ai.usage.output_tokens",
                response["usage"]["completion_tokens"],
            )
            # Splunk can aggregate this to show total cost
            # per service, endpoint, or time window.
            cost = (
                response["usage"]["prompt_tokens"] * 2.50
                / 1_000_000
                + response["usage"]["completion_tokens"]
                * 10.00
                / 1_000_000
            )
            span.set_attribute("rag.generate.cost_usd", cost)

    return response["text"]
```

A powerful Splunk-specific feature is **Tag Spotlight**. Once your spans are flowing into Splunk APM, Tag Spotlight automatically identifies which span attributes correlate with errors or high latency. For example, it might surface that requests where `rag.retrieve.top_score < 0.6` are 4x more likely to result in user complaints. This turns your span attributes into automatic diagnostic insights without manual dashboard building.

Another Splunk advantage is the ability to create **detectors** (real-time alerts) on span attributes. You could configure: "Alert the on-call engineer when the p95 latency of `rag.generate` exceeds 5 seconds for 10 consecutive minutes." Or: "Alert when `rag.retrieve.avg_score` drops below 0.65, indicating potential index staleness."

---

## Component-Level Evaluation: Beyond Black-Box Testing

Most teams evaluate their LLM applications as a black box: feed an input, get an output, score the output. This is like taste-testing the final dish without checking any of the ingredient quality, cooking temperature, or preparation steps.

**Component-level evaluation** means running quality checks at each layer of the pipeline independently.

```
+------------------------------------------------------------------+
|                                                                  |
|  BLACK-BOX EVALUATION                                            |
|  Input -------> [ ?? LLM App ?? ] -------> Output ----> Score    |
|                                                                  |
|  "The food was 6/10."                                            |
|                                                                  |
+------------------------------------------------------------------+


+------------------------------------------------------------------+
|                                                                  |
|  COMPONENT-LEVEL EVALUATION                                      |
|                                                                  |
|  Query ----> Score: Is the query well-formed?                    |
|    |                                                             |
|    v                                                             |
|  Embed ----> Score: Is the vector dimensionally correct?         |
|    |                                                             |
|    v                                                             |
|  Retrieve -> Score: Are the retrieved docs relevant?             |
|    |                  (relevance score, context recall)           |
|    v                                                             |
|  Context --> Score: Is the assembled prompt within limits?       |
|    |                  (token count, completeness)                 |
|    v                                                             |
|  Generate -> Score: Is the final answer faithful to context?     |
|                      (faithfulness, answer relevancy)            |
|                                                                  |
|  "The prep was great, retrieval missed a key document,           |
|   the LLM compensated but hallucinated one detail."              |
|                                                                  |
+------------------------------------------------------------------+
```

Frameworks like **DeepEval** and **Ragas** provide pre-built evaluation metrics for each component. For example:

- **Context Recall** -- Did the retrieval step find all the relevant documents? Evaluated at Layer 3.
- **Context Precision** -- Were the retrieved documents actually relevant, or was there noise? Also Layer 3.
- **Faithfulness** -- Does the generated answer stick to facts found in the context, or does it hallucinate? Evaluated at Layer 5.
- **Answer Relevancy** -- Does the response actually address the user's original question? Cross-layer evaluation linking Layer 1 to Layer 5.

By combining observability (traces and spans) with component-level evaluation (quality scores per layer), you build a comprehensive picture of both *performance* and *quality* across your entire pipeline. The observability tells you *how fast* and *how reliably* each layer is running. The evaluations tell you *how well* each layer is doing its job.

Think of it as the difference between knowing that the kitchen cooked the dish in 12 minutes (observability) and knowing that the dish scored 9/10 on flavor (evaluation). You need both to run a great restaurant.

---

## Best Practices for Production LLM Observability

Drawing from the implementation patterns above, here are the practices that separate well-monitored LLM systems from the rest:

**1. Instrument from day one, not after the first incident.** Adding observability after a production failure is like installing smoke detectors after a fire. The cost of instrumentation is low; the cost of blind debugging is high. Every code example in this article can be added to a new pipeline in under an hour.

**2. Use semantic naming conventions for spans and attributes.** Follow the OpenTelemetry Semantic Conventions for GenAI. Using `gen_ai.request.model` instead of `my_model_name` means that every observability backend in the ecosystem can render meaningful dashboards without custom configuration.

**3. Record business-relevant attributes, not just technical ones.** Token counts and latency are essential, but also record session IDs, user segments, query categories, and cost estimates. These attributes enable business-level analysis: "Which customer segment generates the most expensive queries?" or "Are enterprise users experiencing worse retrieval quality than free-tier users?"

**4. Set alerts on leading indicators, not lagging ones.** Alert on retrieval relevance scores dropping (a leading indicator that output quality will degrade) rather than on user complaint rates (a lagging indicator that damage is already done). Span-level attributes make leading-indicator alerts possible.

**5. Sample wisely in high-throughput systems.** If your system handles thousands of queries per second, exporting every trace will overwhelm your observability backend. Use head-based or tail-based sampling: always capture error traces and slow traces in full, and sample normal traces at a lower rate.

**6. Separate evaluation from observability.** Observability tells you *what happened* (latency, tokens, errors). Evaluation tells you *how good* it was (relevance, faithfulness). Run evaluation asynchronously on sampled traces -- don't add LLM-as-judge calls to your hot path.

**7. Version everything.** Record the embedding model version, the prompt template version, the LLM model version, and the vector index version as span attributes. When quality regresses, these version tags let you correlate the regression with a specific change.

**8. Build dashboards that span all five layers.** A single dashboard should show, at a glance: query volume, embedding latency, retrieval relevance distribution, context token usage, and generation cost. This end-to-end view lets you spot inter-layer effects that single-layer dashboards miss.

---

## Conclusion

LLM applications are no longer experiments -- they're production software serving real users with real expectations. And production software demands production-grade observability.

The five-layer model presented in this article -- Query, Embedding, Retrieval, Context, and Generation -- gives you a systematic framework for understanding what's happening inside your LLM pipeline at every step. Each layer corresponds to a distinct operation with its own failure modes, performance characteristics, and cost profile. By instrumenting each layer as a separate span within a trace, you gain the ability to debug specific failures, track costs to their source, and detect quality drift before it reaches your users.

The three implementation examples -- OpenTelemetry, AppDynamics APM, and Splunk Observability Cloud -- demonstrate that the same conceptual model maps cleanly to any observability platform. OpenTelemetry provides the vendor-neutral foundation. AppDynamics wraps it in enterprise anomaly detection and business transaction analytics. Splunk adds log correlation, Tag Spotlight, and real-time detectors.

The restaurant kitchen analogy we used throughout this article carries one final lesson: the best kitchens don't wait for a customer complaint to start monitoring. They have thermometers in every oven, timers at every station, and quality checks at every handoff. Your LLM pipeline deserves the same.

Start with traces. Add spans for each layer. Record meaningful attributes. Build dashboards. Set alerts. And then -- only then -- will you truly understand what's happening between the question and the answer.

---

## References

**A note on the "Five Layers" model.** The five-layer decomposition of LLM observability (Query, Embedding, Retrieval, Context, Generation) used in this article is not a formally standardized framework from a single authoritative source. It is an emergent industry practice pattern that arises from applying distributed tracing concepts -- as standardized by OpenTelemetry [^1][^2] -- to the well-known stages of a Retrieval-Augmented Generation (RAG) pipeline [^3]. The OpenTelemetry GenAI semantic conventions formalize three of the five layers (Inference/Generation, Embedding, and Retrieval) as standard span types. Enterprise observability platforms such as Cisco AppDynamics [^4][^5] and Splunk Observability Cloud [^6][^7][^8] provide the monitoring infrastructure to operationalize this layered model in production.

[^1]: OpenTelemetry Authors. "Semantic Conventions for Generative AI Systems," v1.40.0 (Development). Includes span conventions for Inference, Embeddings, and Retrievals. [https://opentelemetry.io/docs/specs/semconv/gen-ai/](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

[^2]: OpenTelemetry Authors. "Semantic Conventions for Generative Client AI Spans." Defines `gen_ai.*` attributes for model, token usage, temperature, and finish reason used in the code examples. [https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)

[^3]: Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems* 33 (NeurIPS 2020), pp. 9459--9474. The paper that introduced the RAG architecture whose pipeline stages (query encoding, retrieval, context assembly, generation) form the basis of the five observability layers. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

[^4]: Cisco AppDynamics. "OpenTelemetry with AppDynamics." Documents OTLP ingestion and the mapping of OpenTelemetry traces to AppDynamics Business Transactions. [https://docs.appdynamics.com/appd/24.x/en/application-monitoring/opentelemetry](https://docs.appdynamics.com/appd/24.x/en/application-monitoring/opentelemetry)

[^5]: Cisco AppDynamics. "Business Transactions." Describes how AppDynamics discovers, maps, and monitors the performance of application transactions -- the mechanism through which OTel spans surface in the AppDynamics UI. [https://docs.appdynamics.com/appd/24.x/en/application-monitoring/business-transactions](https://docs.appdynamics.com/appd/24.x/en/application-monitoring/business-transactions)

[^6]: Splunk. "Splunk Observability Cloud: APM." Documents Splunk's OpenTelemetry-native APM, including trace visualization, service maps, and Tag Spotlight for span-attribute-driven diagnostics. [https://docs.splunk.com/observability/en/apm/](https://docs.splunk.com/observability/en/apm/)

[^7]: Splunk. "Splunk Distribution of OpenTelemetry Python." Splunk's packaging of the OTel Python SDK with pre-configured exporters and auto-instrumentation for common frameworks. [https://docs.splunk.com/observability/en/gdi/get-data-in/application/python/get-started.html](https://docs.splunk.com/observability/en/gdi/get-data-in/application/python/get-started.html)

[^8]: Splunk. "Create Detectors to Trigger Alerts." Documents how to configure real-time alerting on span attributes in Splunk Observability Cloud. [https://docs.splunk.com/observability/en/alerts-detectors-notifications/create-detectors-for-alerts.html](https://docs.splunk.com/observability/en/alerts-detectors-notifications/create-detectors-for-alerts.html)

[^9]: OpenTelemetry Authors. "OpenTelemetry Python SDK." The vendor-neutral tracing SDK used in all three code examples. [https://opentelemetry.io/docs/languages/python/](https://opentelemetry.io/docs/languages/python/)

[^10]: Confident AI. "DeepEval: The Open-Source LLM Evaluation Framework." Provides component-level evaluation metrics (faithfulness, answer relevancy, context recall) referenced in the Component-Level Evaluation section. [https://github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)

[^11]: Explodinggradients. "Ragas: Evaluation Framework for Retrieval Augmented Generation." Provides RAG-specific evaluation metrics (context precision, context recall, faithfulness) referenced in the Component-Level Evaluation section. [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
