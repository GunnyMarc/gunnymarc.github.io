# Network Modeling in Cisco ThousandEyes: Graph Theory Models & Algorithms

*A technical deep-dive into the graph-theoretic foundations, algorithms, and data structures that power ThousandEyes' network intelligence platform.*

---

## I. Introduction

Modern enterprise infrastructure depends on networks that no single organization owns or controls. A request from a remote employee's browser to a SaaS application may traverse a home ISP, a regional transit provider, one or more Tier-1 backbone networks, a CDN edge node, a cloud provider's internal fabric, and finally the application's load balancer — all before the first byte of response data is generated. When performance degrades, the immediate question — *where is the problem?* — demands visibility across every one of those domains.

Cisco ThousandEyes addresses this challenge by deploying a global mesh of software agents that continuously probe network paths, collect BGP routing tables, and measure application response times. The raw output of these probes — ICMP TTL-exceeded messages, TCP handshake timings, BGP UPDATE messages, SNMP interface counters — is voluminous and, in isolation, unintelligible. What transforms it into actionable intelligence is **graph theory**: the branch of mathematics concerned with pairwise relationships between objects.

Every core visualization and detection capability in ThousandEyes is, at its foundation, a graph operation:

- **Path Visualization** constructs a directed, weighted graph of IP hops from agents to a destination, then overlays performance metrics on nodes and edges.
- **BGP Route Visualization** builds an Autonomous-System-level directed graph from route collector data, enabling detection of hijacks, leaks, and path instability.
- **Device Layer** auto-discovers internal infrastructure via LLDP/CDP and renders it as a Layer-2 adjacency graph enriched with SNMP health telemetry.
- **Internet Insights** aggregates de-identified measurements from the entire ThousandEyes agent fleet into a global provider-infrastructure graph, applying cluster analysis to detect macro-scale outages.

This article examines these graph models in detail: the abstractions they use, the algorithms that build and analyze them, the visualization techniques that make them interpretable, and the programmatic interfaces that allow engineers to extend them.

---

## II. Graph Theory Foundations as Applied in ThousandEyes

Before examining each product capability, it is useful to establish the specific graph-theoretic constructs that ThousandEyes employs and how they map to network engineering concepts.

### 2.1 Core Abstractions

**Nodes (Vertices).** In ThousandEyes, a node represents a distinct network entity. The specific entity depends on the graph model in use:

| Graph Model | Node Represents | Example |
|---|---|---|
| Path Visualization | A unique IP address responding to a probe | `72.14.236.217` (Google edge router) |
| BGP Route Visualization | An Autonomous System (AS) | AS 15169 (Google) |
| Device Layer | A discovered network device | Cisco Catalyst 9300 switch |
| Internet Insights | A provider Point of Presence (PoP) | Comcast PoP, Chicago |

**Edges (Links).** An edge represents a connection or relationship between two nodes. Edges carry attributes — metadata and performance metrics — that are central to the platform's diagnostic value:

- **Path Visualization edges**: Represent a network segment between two consecutive hops. Attributes include forwarding loss (%), link delay (ms), number of traces traversing the link, DSCP markings, and minimum path MTU.
- **BGP edges**: Represent a peering or transit relationship between two ASes. Attributes include the number of path changes observed, reachability percentage, and BGP update counts.
- **Device Layer edges**: Represent Layer-2 connections between device interfaces, discovered via neighbor protocol advertisements.

**Directed vs. undirected.** Path Visualization graphs are inherently directed — traffic flows from agent (source) to destination. ThousandEyes renders this with arrows and supports toggling between source-to-target, target-to-source, and bidirectional views. In Agent-to-Agent tests, both directions are measured independently, producing two distinct directed graphs that may differ substantially due to asymmetric routing. BGP Route Visualization is also directed: edges point from the monitoring vantage point toward the origin AS, following the AS-path attribute in reverse.

**Weighted graphs.** Nearly all ThousandEyes graphs are weighted. The weight on an edge is the value of a selected performance metric — typically latency, loss, or jitter. The platform's color-coding system maps these weights to a green-to-red gradient, providing immediate visual encoding of graph-edge severity.

### 2.2 Graph Representations in the Platform

ThousandEyes employs three primary graph representations internally, each optimized for its use case:

**Interactive Directed Graph (Path Visualization).** The path trace data collected by agents is assembled into a composite directed graph where shared hops across multiple agents are merged into single nodes and divergent routes branch visually. This is conceptually close to a directed acyclic graph (DAG) from agents (sources) to the destination (sink), although routing loops — when detected — introduce cycles and are flagged with a distinct red-loop indicator.

**AS-Level Directed Graph (BGP Route Visualization).** BGP data from public monitors (RIPE-RIS, RouteViews) and customer-deployed private monitors is assembled into a graph where each node is an AS and each edge is a segment of the AS-path. The resulting structure is a directed forest rooted at the monitored prefix's origin AS, with monitor vantage points as leaves.

**Adjacency Graph (Device Layer).** Internal topology is represented as an undirected adjacency graph built from LLDP and CDP neighbor tables. Each device is a node; each discovered neighbor relationship is an edge. SNMP polling enriches nodes with health metrics (CPU utilization, memory consumption, interface error rates, bandwidth utilization), turning the raw adjacency graph into a health-annotated topology map.

### 2.3 Key Graph Properties ThousandEyes Exploits

Several classical graph properties map directly to network monitoring concepts:

**Connectivity.** The fundamental question — *can the agent reach the target?* — is a connectivity query on the path graph. A disconnected graph (no path from agent node to destination node) indicates a reachability failure. ThousandEyes reports this as 100% loss with an incomplete path trace.

**Path multiplicity.** Modern networks use Equal-Cost Multi-Path (ECMP) routing, meaning multiple shortest paths may exist between two points. ThousandEyes exploits this by performing 3 to 10 parallel path traces per agent, each using a unique TCP source port to encourage the network's ECMP hash function to select different paths. The resulting graph captures this multiplicity: split paths are rendered with varying line thickness proportional to the number of traces traversing each link.

**Branching and convergence.** When multiple agents test the same destination, their paths often diverge near the source and converge near the destination. The graph representation merges convergent hops into shared nodes, producing a tree-like structure that clearly shows where paths overlap and where they diverge — critical for determining whether a problem affects one agent or many.

**Cycles (routing loops).** A well-functioning network graph should be acyclic along any given path. When ThousandEyes detects that a packet revisits a previously seen node, it renders a red loop indicator around that node, immediately flagging a routing misconfiguration.

---

## III. ThousandEyes Network Models and Their Graph Structures

### 3.1 Path Visualization Model

Path Visualization is ThousandEyes' signature capability and its most direct application of graph theory. It constructs a composite graph from the path trace data collected by all agents testing a given target, rendering the Internet's routing topology as a navigable, metric-annotated visual.

**Graph construction.** Each ThousandEyes agent — whether a Cloud Agent deployed in a public data center, an Enterprise Agent on a customer's network, or an Endpoint Agent on a user's device — performs path traces to the test target. The agent sends probe packets with incrementally increasing Time-To-Live (TTL) values. Each intermediate router decrements the TTL; when it reaches zero, the router responds with an ICMP Time Exceeded message, revealing its IP address. This process, repeated until the target responds, produces an ordered sequence of IP addresses — a path.

To discover ECMP routes, each agent performs multiple parallel path traces (3 by default, configurable up to 10) using unique, randomized TCP source ports. Since ECMP hash functions typically incorporate the source port into their path-selection decision, different source ports may yield different paths through the network.

The resulting set of paths from all agents is merged into a single directed graph:

1. **Nodes** are created for each unique IP address observed across all path traces. Nodes are categorized as:
   - **Agent nodes** (leftmost): The originating ThousandEyes agents.
   - **Intermediate nodes**: IP addresses of routers along the path, typically belonging to ISPs, transit providers, or cloud fabrics.
   - **Destination node** (rightmost): The target IP address.
   - **Blank nodes**: Placeholders for hops that did not respond to probes (rendered as empty circles).

2. **Edges** connect consecutive nodes in each observed path. When multiple agents share a common hop, the edges converge at that node, producing a merged graph rather than parallel isolated paths.

3. **Edge attributes** encode performance data:
   - **Forwarding loss (%)**: Percentage of probes that were dropped at this hop.
   - **Link delay (ms)**: Estimated minimum transmission delay across this edge.
   - **Jitter (ms)**: Variability in probe round-trip times.
   - **DSCP marking**: The Differentiated Services Code Point value observed in returned packets.
   - **Minimum path MTU (bytes)**: The smallest Maximum Transmission Unit along the path up to this point.
   - **Trace count**: The number of individual path traces that traversed this edge — rendered as line thickness.

4. **Node attributes** include the IP address, reverse DNS hostname, WHOIS-derived network ownership, autonomous system number, and geographic location.

**Temporal dimension.** Path Visualization is not a static snapshot. ThousandEyes collects data in discrete test rounds (typically every 2 minutes), and the visualization can be scrubbed across a timeline. This allows engineers to observe how the graph structure changes over time — routes shifting, new hops appearing, existing hops becoming lossy — providing a temporal graph analysis capability.

### 3.2 BGP Route Visualization Model

While Path Visualization operates at the IP-hop level (Layer 3 forwarding plane), BGP Route Visualization operates at the Autonomous System level (Layer 3 control plane). It models the Internet's routing topology as an AS-path graph.

**Data sources.** ThousandEyes ingests BGP routing data from two categories of monitors:

- **Public BGP monitors**: eBGP sessions maintained with routers participating in the RIPE Routing Information Service (RIPE-RIS) and the University of Oregon's RouteViews project, as well as ThousandEyes' own public BGP collectors. These provide an "outside-in" view — how the global Internet sees a given prefix.
- **Private BGP monitors**: Customer-configured multi-hop eBGP sessions between their own BGP speakers and ThousandEyes' route collectors. These provide an "inside-out" view — how the customer's own network sees external prefixes.

**Graph structure.** For a monitored prefix, the BGP Route Visualization constructs a directed graph where:

- **Nodes** are Autonomous Systems, identified by their ASN and annotated with the organization name (sourced from WHOIS registries, CAIDA, BGP.Tools, APNIC, and RIPE NCC).
- **Edges** represent AS-path segments. An edge from AS *A* to AS *B* means that *B* is the next hop in the AS-path as advertised to the monitor. Edge direction follows the AS-path from the monitor toward the origin AS.
- **Edge metrics** include: the number of path changes observed in a given time window, reachability percentage (what fraction of the time the prefix was visible via this path), and raw BGP update counts.

**AS prepending detection.** A common traffic engineering technique is AS-path prepending, where an AS inserts its own ASN multiple times into the AS-path to make a route appear longer and thus less preferred. In the graph, this manifests as a self-loop on a node — the same ASN appearing consecutively in the path. ThousandEyes highlights these prepended segments, allowing engineers to distinguish genuine path lengthening from artificial manipulation.

**RPKI validation layer.** ThousandEyes validates route origins against the Resource Public Key Infrastructure (RPKI) and annotates the graph accordingly. Each prefix-origin pair is marked as *Valid* (the origin AS is authorized by an ROA), *Invalid* (the origin AS contradicts a published ROA, suggesting a possible hijack), or *Not Found* (no ROA exists for this prefix). This transforms the AS graph into a security-annotated graph where routing anomalies are immediately visible.

### 3.3 Device Layer Topology Model

The Device Layer extends ThousandEyes' graph modeling inward, mapping an organization's own network infrastructure.

**Discovery algorithm.** Starting from Enterprise Agents deployed within the network, ThousandEyes queries LLDP (Link Layer Discovery Protocol) and CDP (Cisco Discovery Protocol) neighbor tables via SNMP. Each neighbor advertisement reveals a connected device and its interface, providing the raw adjacency data for graph construction. The discovery process crawls outward from the agent, building a breadth-first traversal of the network's Layer-2 topology.

**Graph structure.** The resulting graph is an undirected adjacency graph where:

- **Nodes** represent network devices — routers, switches, firewalls, load balancers, wireless controllers — each rendered with a type-specific icon.
- **Edges** represent Layer-2 links between device interfaces.
- **Node attributes** are enriched via SNMP polling: device type, firmware version, CPU utilization, memory consumption, interface error counters, and bandwidth utilization per interface.

**Correlation with Path Visualization.** The Device Layer graph is not isolated — it is correlated with the path trace graph. When an IP address in the Path Visualization corresponds to a discovered device in the Device Layer, the two graphs are linked, allowing engineers to pivot from "this hop has 5% packet loss" to "this hop is interface GigabitEthernet0/1 on switch `core-sw-01`, which is currently at 94% CPU."

### 3.4 Internet Insights — The Aggregate Network Graph

Internet Insights operates at the largest scale: a global graph of Internet infrastructure derived from the collective measurements of the entire ThousandEyes agent fleet.

**Data aggregation.** ThousandEyes agents worldwide — cloud agents, enterprise agents, endpoint agents — collectively perform billions of measurements daily. This data is de-identified (all customer-specific and private-network information is stripped) and aggregated into a global dataset. The result is a graph of Internet provider infrastructure where:

- **Nodes** represent network Points of Presence (PoPs) for ISPs, CDNs, DNS providers, IaaS platforms, UCaaS services, SECaaS providers, and major SaaS applications.
- **Edges** represent observed connectivity between PoPs, derived from the aggregate path trace data.

**Outage detection as graph cluster analysis.** Internet Insights identifies outages by detecting anomalous clusters within this graph:

- **Network outages**: Triggered when a concentration of 100% packet-loss events is detected within a single network PoP in a short time frame. The algorithm continuously monitors lossy interfaces across all networks and PoPs, maintaining baselines for normal loss levels. When loss events significantly exceed the baseline within a PoP, the algorithm classifies the event as an outage, estimating its scope (how many PoPs are affected) and scale (how many vantage points are impacted).
- **Application outages**: Triggered when multiple globally distributed vantage points simultaneously fail to reach an application's servers or receive error responses. The requirement for multi-vantage-point confirmation ensures that isolated agent-side issues are not misclassified as provider outages.

**Geographical and topological views.** The outage graph is rendered both geographically (outages superimposed on a world map) and topologically (outages shown in context of the provider's network structure), allowing engineers to quickly assess scope and impact.

### 3.5 Cloud and SD-WAN Enriched Models

ThousandEyes augments its base graph models with enrichment layers for cloud and SD-WAN environments:

**Cloud network enrichment.** In collaboration with AWS, Azure, and GCP, ThousandEyes maps IP addresses in the path graph to specific cloud services, regions, and availability zones. A raw IP node like `52.93.178.12` is annotated as "AWS S3, us-east-1." For AWS Global Accelerator targets, the platform compares observed TCP latency against expected latency benchmarks, providing a deviation metric directly on the enriched node.

**SD-WAN overlay/underlay dual-layer graph.** For organizations using Cisco SD-WAN or Meraki MX, ThousandEyes constructs a two-layer graph model. The **overlay graph** shows the logical SD-WAN tunnel paths between branch sites and application endpoints. The **underlay graph** shows the physical network paths those tunnels traverse — through ISPs, MPLS circuits, or direct Internet paths. By correlating performance metrics across both layers, engineers can determine whether a problem is in the overlay configuration or the underlay transport.

**Meraki enrichment.** When integrated with Meraki environments, path visualization nodes within the Meraki network are enriched with the hosting network name, MX appliance name, connected client count, and WAN application score — providing campus/branch context directly within the graph.

---

## IV. Algorithms and Computational Methods

### 4.1 Path Discovery and Traversal

The foundation of ThousandEyes' path graph is the **TTL-incrementing probe algorithm** — an engineered variant of traceroute optimized for multi-agent, multi-path environments.

**Basic mechanism.** The agent estimates the path distance to the target and then sends probe packets with incrementally increasing TTL values, starting from TTL=1. Each intermediate router decrements the TTL and, upon reaching zero, responds with an ICMP Time Exceeded message containing the router's source IP address. The agent records the responding IP and its round-trip time, then sends the next probe with TTL+1. The process terminates when a response from the target itself is received or when a maximum TTL is reached without a response (rendering blank nodes for unresponsive hops).

**Multi-path discovery.** To detect ECMP routes, each agent performs 3 parallel path traces by default (configurable up to 10). Each trace uses a unique, randomized TCP source port. Because most ECMP implementations hash on the 5-tuple (source IP, destination IP, source port, destination port, protocol), varying the source port encourages the network to select different forwarding paths. The resulting set of paths is merged into the composite graph, with split paths rendered as branches and their relative usage indicated by edge thickness.

**Protocol selection.** Agents support both TCP and ICMP-based path tracing:
- **TCP mode**: Sends TCP SYN packets; expects SYN+ACK or RST from the target. Preferred for targets behind firewalls that may drop ICMP.
- **ICMP mode**: Sends ICMP Echo Request packets; expects Echo Reply from the target. Useful when TCP ports are filtered.

**Bidirectional tracing.** In Agent-to-Agent tests, both endpoints perform independent path traces toward each other. This produces two directed graphs — source-to-target and target-to-source — which often differ due to asymmetric routing. The visualization allows toggling between these views, providing complete bidirectional path visibility.

**Continuous high-frequency probing.** For tests configured with 1-minute intervals, ThousandEyes sends one probe per second over the entire interval (rather than a burst at the start). This continuous sampling captures intermittent loss events that burst-based probing might miss, and the results are rendered as a sparkline visualization showing per-second packet drop patterns.

### 4.2 Shortest Path and Latency Analysis

While ThousandEyes does not run Dijkstra's algorithm on the path graph in the classical sense (it observes actual forwarding paths rather than computing optimal ones), the platform performs analogous weighted-graph analysis:

**End-to-end latency.** The total latency from agent to target is measured via TCP or ICMP round-trip time. This is the weight of the shortest path in the observed graph — though in practice, the Internet may not route along the latency-optimal path.

**Per-hop delay estimation.** ThousandEyes estimates the transmission delay across each individual link by measuring the round-trip time to consecutive hops and computing the differential. This isolates each edge's latency contribution, enabling engineers to identify the specific link responsible for latency spikes — analogous to computing edge weights in a weighted graph and finding the maximum-weight edge.

**Benchmark comparison.** For cloud-enriched nodes, the platform compares observed latency against provider-published benchmarks. For example, for AWS Global Accelerator targets, ThousandEyes compares the measured TCP connection time against AWS's expected latency for that region, flagging deviations that indicate network-layer problems rather than application-layer issues.

### 4.3 Centrality and Critical Node Identification

Graph centrality measures, while not labeled as such in the ThousandEyes interface, underpin several key diagnostic capabilities:

**Betweenness centrality (shared-hop analysis).** When multiple agents test the same destination, their paths often converge at shared intermediate hops. A node that appears on the paths of many agents has high betweenness centrality in the test graph. If that node begins dropping packets, the impact is proportionally larger — affecting all agents whose paths traverse it. ThousandEyes' visualization makes this immediately apparent: high-betweenness nodes sit at convergence points in the graph, and packet loss on those nodes is visible to every affected agent simultaneously.

**Cut vertices (single points of failure).** A node whose removal would disconnect one or more agents from the destination is a *cut vertex* in graph-theoretic terms — a single point of failure. ThousandEyes' path graph reveals these implicitly: if all agent paths funnel through a single intermediate node before reaching the destination, that node is a cut vertex. Identifying these nodes is critical for resilience planning.

**Loss attribution.** When end-to-end loss is detected, the question is *which node or link is responsible?* ThousandEyes performs per-hop loss analysis by comparing the probe response rate at consecutive hops. If hop *n* responds to 100% of probes but hop *n+1* responds to only 95%, the link between them — or hop *n+1* itself — is attributed with 5% forwarding loss. This is visualized as a red circle around the lossy node and a red-colored link, immediately drawing attention to the responsible edge in the graph.

### 4.4 Clustering and Outage Detection (Internet Insights)

Internet Insights' outage detection is, at its core, a **spatial clustering algorithm** applied to a global graph of Internet measurement data.

**Collective intelligence aggregation.** The input dataset is extraordinary in scale: billions of daily measurements from ThousandEyes agents deployed across thousands of networks worldwide. Before aggregation, all data is de-identified — customer identifiers and private-network information are stripped. The remaining data consists of tuples: *(agent_network, intermediate_hop_IP, hop_network, hop_PoP, loss_flag, timestamp)*.

**PoP-level cluster detection.** The algorithm groups loss events by network and PoP. For each PoP, it maintains a rolling baseline of normal loss-event frequency. When the observed frequency of 100% packet-loss events within a PoP exceeds the baseline by a statistically significant margin within a short time window, the algorithm triggers an outage alert. The outage's **scope** is determined by the number of distinct PoPs within the same network that are simultaneously affected. Its **scale** is determined by the number of distinct agent vantage points and customer tests that are impacted.

**Application outage inference.** For SaaS and cloud applications, the algorithm applies a similar clustering approach at the application level. When multiple globally distributed agents simultaneously fail to receive valid responses from an application's endpoints — and these failures correlate across independent networks and geographies — the algorithm infers an application-level outage. The multi-vantage-point requirement is critical: it prevents false positives from agent-side or local-network issues.

**Correlation with customer tests.** Detected outages are automatically correlated with each ThousandEyes customer's own test data. If a customer's test to Salesforce shows degradation at the same time Internet Insights detects a Salesforce outage, the platform links the two, enabling the customer to immediately determine that the problem is external — not in their own network.

### 4.5 BGP Routing Algorithms

ThousandEyes applies several specialized algorithms to its BGP data:

**Reachability monitoring.** For each monitored prefix, the platform tracks what percentage of BGP monitors can see a valid route. A drop in reachability — visible as a declining metric on the timeline — indicates that the prefix is being withdrawn from portions of the global routing table. The algorithm correlates reachability drops across monitors to distinguish localized issues (one monitor loses the route) from widespread events (many monitors simultaneously lose it).

**Path change detection.** The algorithm continuously compares the current AS-path for each prefix against the previously observed AS-path. Any change — a new transit AS inserted, an existing AS removed, a path lengthened or shortened — triggers a path-change event. Rapid oscillation in AS-paths (route flapping) is flagged as a stability concern.

**Route hijack and leak detection.** A BGP hijack occurs when an unauthorized AS announces a prefix it does not own, diverting traffic. A leak occurs when a route is propagated beyond its intended scope. ThousandEyes detects these by:
1. Monitoring for new origin ASes appearing for a prefix (potential hijack).
2. Checking origin AS authorization against RPKI ROAs (an Invalid RPKI status is a strong hijack indicator).
3. Detecting unexpected AS-paths that suggest a route is being propagated through unintended transit networks (potential leak).

**Stuck route detection.** BGP "zombie" routes are routes that persist in routing tables despite having been withdrawn by the origin. ThousandEyes' Stuck Route Observatory identifies these by comparing the routes seen by monitors against the routes the origin AS is actively advertising. Discrepancies indicate stuck routes, which can cause persistent reachability issues.

**Penalty algorithm.** ThousandEyes employs a penalty-based algorithm to handle BGP monitor data quality issues. When a monitor misses expected updates or exhibits anomalous behavior, the algorithm assigns penalty scores and, above a threshold, triggers corrective actions such as excluding the monitor from aggregate calculations until it stabilizes.

### 4.6 Topology Discovery (Device Layer)

The Device Layer's graph construction algorithm is a **breadth-first crawl** of the network's neighbor tables:

1. **Seed nodes**: Enterprise Agents serve as the starting points. The agent queries its local network for directly connected devices via SNMP.
2. **Neighbor table crawl**: For each discovered device, ThousandEyes reads its LLDP and CDP neighbor tables via SNMP, revealing adjacent devices and their connecting interfaces.
3. **Recursive expansion**: Newly discovered devices are queried in turn, and their neighbors are added to the graph. The process continues until no new devices are found or the configured discovery scope is exhausted.
4. **Graph assembly**: The collected adjacency data is assembled into an undirected graph. Duplicate edges (device A reports device B as neighbor; device B reports device A as neighbor) are deduplicated.
5. **SNMP enrichment**: Each device node is polled for health metrics — CPU, memory, interface errors, bandwidth — which are overlaid as node attributes in the topology visualization.

The result is a Layer-2 topology map that can be correlated with the Layer-3 Path Visualization graph, bridging the gap between logical forwarding paths and physical device infrastructure.

---

## V. Graph Simplification and Visualization Techniques

Raw network graphs — especially those spanning the global Internet — can contain hundreds of nodes and thousands of edges. ThousandEyes employs several graph-reduction and visual-encoding techniques to make these graphs interpretable:

### 5.1 Interface Grouping

A single physical router may have dozens of IP addresses (one per interface). In a raw path trace, each interface appears as a separate node, inflating the graph and obscuring the actual topology. ThousandEyes' interface grouping collapses multiple IPs belonging to the same device into a single node, producing a graph that more accurately represents the physical network. Grouping is configurable:

- **By IP address**: No grouping; each IP is a distinct node (maximum detail).
- **By device**: IPs on the same device are merged (inferred from rDNS and WHOIS data).
- **By network**: All IPs within the same AS/network are merged into a single node.
- **By network + location**: Network-level grouping further subdivided by geographic location.
- **By geography**: All nodes in the same geographic area are merged.

### 5.2 Complexity Controls

A slider control allows users to progressively hide intermediate hops. At maximum complexity, every discovered hop is visible. As the slider is reduced, core-Internet hops — those deep within transit provider backbones — are collapsed into dotted lines annotated with the number of hidden hops. This focuses attention on the edges of the path: the agent's local network and the destination's network, which are most likely to contain the root cause of a problem.

### 5.3 Performance Color Encoding

The graph's visual weight is driven by metric values:

| Color | Meaning |
|---|---|
| **Dark green** | Healthy — 0% loss, low latency |
| **Yellow/Orange** | Degraded — moderate loss or elevated latency |
| **Red** | Critical — high loss, extreme latency, or link failure |
| **Red circle** around a node | Packet loss detected at this hop |
| **Red link** | High delay on this segment |
| **Red loop** around a node | Routing loop detected |

This encoding transforms the graph into a heat map: a healthy network appears as a green flow from left to right, while problems appear as red "hot spots" that an engineer can immediately zoom into.

### 5.4 Split-Path and Collapsed-Path Rendering

- **Split paths**: When ECMP or policy routing causes traffic to take multiple routes, the graph branches at the divergence point. Each branch's line thickness is proportional to the number of traces that traversed it, indicating the load distribution across paths.
- **Collapsed paths**: When complexity controls hide intermediate hops, the hidden segment is rendered as a dotted line with a numeric annotation (e.g., "5 hops hidden"), preserving awareness of the path's true length without cluttering the visualization.

### 5.5 Cloud Provider Annotation

For paths traversing AWS, Azure, or GCP infrastructure, ThousandEyes replaces raw IP nodes with enriched labels showing the cloud service name, region, and availability zone. Nodes display the cloud provider's icon, and a verified-information badge indicates that the enrichment data was confirmed by the cloud provider. This transforms opaque IP addresses into meaningful infrastructure context directly within the graph.

---

## VI. Network Resilience and Fault Analysis Through Graph Theory

The graph models constructed by ThousandEyes enable several categories of resilience analysis that map directly to classical graph-theoretic problems:

### 6.1 Outage Impact Assessment

When a provider announces an outage — or Internet Insights detects one — the immediate question is: *how does this affect my services?* This is a **graph reachability** problem: given that node *X* (the failed PoP) is removed from the graph, which agent-to-destination paths are severed? ThousandEyes answers this by correlating Internet Insights outage data with customer test data, automatically identifying which tests traverse the affected nodes.

### 6.2 Cascade Analysis

A failure in one part of the network graph can propagate. If a Tier-1 transit provider's backbone link fails, traffic is rerouted through alternative paths, potentially overloading those paths and causing secondary failures. ThousandEyes' temporal path visualization captures these cascades: engineers can observe the graph structure before, during, and after a failure event, watching paths shift, latency increase on alternative routes, and — in severe cases — loss appear on previously healthy paths.

### 6.3 Redundancy Validation

A resilient network architecture requires **edge-disjoint paths** — multiple independent routes between critical endpoints. ThousandEyes' multi-path discovery verifies this: if all traces from an agent converge on a single intermediate hop, that hop is a single point of failure regardless of how many ISPs the organization has contracted. The path graph makes this visible immediately, enabling engineers to validate that their multi-homed or multi-cloud architecture actually provides the expected redundancy.

### 6.4 DDoS Mitigation Validation

During a DDoS attack, traffic is typically rerouted through a scrubbing center via BGP announcements. ThousandEyes provides two graph-level views of this process:
1. **BGP Route Visualization** shows the AS-path change as the scrubbing center's AS is inserted into the path.
2. **Path Visualization** shows the actual forwarding-plane change: traffic now routes through the scrubbing center's IP infrastructure.

By monitoring both graphs during an attack, engineers can verify that mitigation is active, measure the latency overhead introduced by scrubbing, and confirm that clean traffic is being properly re-injected to the origin.

### 6.5 SLA Enforcement and Vendor Comparison

Internet Insights tracks outage history per provider, building a longitudinal graph of reliability data. This enables:
- **SLA enforcement**: Quantifying a provider's actual availability against contractual commitments, backed by concrete telemetry rather than the provider's own reporting.
- **Vendor evaluation**: Comparing the outage frequency, duration, and scope of competing providers using graph-derived metrics, supporting data-driven procurement decisions.

---

## VII. Data Access and Programmatic Graph Analysis

### 7.1 ThousandEyes API

The ThousandEyes REST API exposes the platform's graph data programmatically, enabling custom analysis, integration, and automation:

**Path trace endpoints.** The API returns detailed path trace data for each test round, including the ordered sequence of hops (nodes), their IP addresses, network ownership, geographic location, and per-hop metrics (loss, latency, delay, DSCP, MTU). This data can be consumed as a node-and-edge list for reconstruction in external graph analysis tools.

**Network end-to-end endpoints.** Aggregate metrics — agent-to-target loss, latency, jitter, and bandwidth — are available as time-series data, enabling trend analysis and long-term performance tracking.

**BGP endpoints.** The API provides AS-path data, reachability metrics, update counts, and RPKI validation status for each monitored prefix, enabling programmatic BGP graph construction and analysis.

**Export formats.** API responses are JSON-structured, with node and edge data that maps directly to adjacency-list representations suitable for import into graph analysis libraries.

### 7.2 Integration with Observability Platforms

ThousandEyes' graph data feeds into broader observability ecosystems:

- **Splunk**: The Cisco ThousandEyes App for Splunk streams test data, outage events, and activity logs into Splunk dashboards. This enables correlation of ThousandEyes graph data with logs, metrics, and traces from other sources, providing a unified view of infrastructure health.
- **OpenTelemetry**: ThousandEyes supports streaming BGP metrics via the OpenTelemetry protocol, allowing integration with any OTel-compatible backend (Grafana, Datadog, New Relic, etc.).
- **Webhooks and ServiceNow**: Alert-driven integrations push graph events (outages, path changes, loss thresholds) to incident management systems, triggering automated workflows.
- **Splunk AppDynamics**: Combining application performance monitoring with ThousandEyes' network graph provides end-to-end visibility from application code to network path.

### 7.3 Custom Graph Analysis Workflows

Engineers who need analysis beyond the built-in visualizations can leverage the API to build custom workflows:

- **Graph library import**: Export path trace data and import into Python's NetworkX or R's igraph for advanced graph-theoretic computations — centrality measures, community detection, minimum cut analysis, etc.
- **Topology diffing**: By querying the API at regular intervals and comparing successive graph snapshots, engineers can detect structural changes — new hops appearing, existing hops disappearing, path lengths changing — and trigger automated alerts on topology drift.
- **Custom dashboards**: API data can feed into Grafana, Tableau, or custom web applications for tailored graph visualizations that match specific operational requirements.

---

## VIII. AI-Powered Graph Intelligence

Beginning in 2025, Cisco is layering AI capabilities on top of ThousandEyes' graph-derived telemetry:

### 8.1 Cisco AI Assistant

The Cisco AI Assistant, integrated into the ThousandEyes interface, is trained on network telemetry data and test configurations. It can:
- Analyze path visualization data in real time and provide natural-language root-cause summaries.
- Identify which graph nodes are contributing to degradation without requiring the engineer to manually inspect each hop.
- Correlate graph anomalies across multiple tests and time windows, surfacing patterns that might not be apparent from a single graph view.

### 8.2 WAN Insights

WAN Insights applies statistical models to SD-WAN telemetry graphs, producing **predictive routing recommendations**. By analyzing historical patterns in the overlay/underlay graph — latency trends, loss patterns, path utilization — the system can forecast future degradation and recommend proactive path changes before users are affected. This is a form of **predictive graph analytics**: using temporal patterns in a dynamic graph to anticipate structural changes.

### 8.3 AgenticOps

Cisco's AgenticOps vision extends AI from advisory to autonomous action. Specialized AI agents continuously:
1. **Sense**: Ingest real-time graph telemetry from ThousandEyes agents.
2. **Reason**: Apply graph analysis and anomaly detection to identify emerging issues.
3. **Act**: Execute corrective actions — rerouting traffic, adjusting SD-WAN policies, escalating to incident management.
4. **Validate**: Re-measure the graph after action to confirm the issue is resolved.

This closes the loop from graph observation to graph-informed remediation, moving toward autonomous network operations.

### 8.4 Machine Learning on Historical Graph Patterns

ThousandEyes' longitudinal graph data — capturing path structures, performance metrics, and outage events over months and years — provides a rich training dataset for anomaly detection models. These models learn the "normal" graph structure for a given test and flag deviations: unexpected new hops, abnormal latency distributions, path changes that correlate with past outage patterns. This transforms the graph from a diagnostic tool into a predictive one.

---

## IX. Real-World Application Domains

### 9.1 Enterprise SaaS Monitoring

For enterprises dependent on SaaS applications — Microsoft 365, Salesforce, ServiceNow, Webex, Zoom — ThousandEyes constructs path graphs from office and remote-worker locations to each application's endpoints. This reveals which ISPs, transit providers, and CDN nodes are in the critical path, enabling targeted escalation when performance degrades. Internet Insights adds a macro view: if Salesforce is experiencing a widespread outage, the enterprise can immediately confirm the issue is external and redirect support resources accordingly.

### 9.2 Multi-Cloud Assurance

Organizations operating across AWS, Azure, and GCP face the challenge of monitoring interconnections between cloud providers — inter-region and inter-cloud traffic traverses networks outside the customer's control. ThousandEyes' cloud-enriched path graphs map these interconnections, identifying performance bottlenecks at cloud-provider handoff points and enabling data-driven multi-cloud architecture decisions.

### 9.3 SD-WAN Optimization

Cisco SD-WAN and Meraki MX deployments benefit from ThousandEyes' dual-layer graph model. When an SD-WAN tunnel shows degradation, the overlay/underlay correlation pinpoints whether the issue is in the overlay policy (tunnel misconfiguration, incorrect SLA class assignment) or the underlay transport (ISP congestion, backbone failure). WAN Insights extends this with predictive recommendations, suggesting proactive path changes based on graph telemetry trends.

### 9.4 Hybrid Workforce

With employees working from home, coffee shops, and co-working spaces, the "last mile" to the corporate network is no longer a managed LAN segment — it's an uncontrolled path through consumer ISPs and public Internet. Endpoint Agents on employee devices construct path graphs from each location to corporate applications, identifying ISP-specific issues (a particular residential ISP's peering point is congested) and enabling IT to provide targeted guidance or escalate to the ISP with concrete evidence.

### 9.5 Industrial IoT (IIoT)

The 2025 extension of ThousandEyes to Cisco Industrial Ethernet switches and Industrial Routers brings graph-based visibility to operational technology (OT) environments. Enterprise Agents deployed on industrial networking equipment construct path graphs from factory floors and remote sites to cloud-hosted SCADA, MES, and ERP systems, enabling IT/OT teams to collaboratively troubleshoot connectivity issues that affect production.

### 9.6 Incident Response

During a major incident, the combination of Internet Insights (macro-scale outage graph), Path Visualization (hop-level diagnostic graph), BGP Route Visualization (control-plane routing graph), and Device Layer (internal infrastructure graph) provides a multi-layer graph model that spans the full incident domain. Engineers can start at the highest level — *is this a global outage?* — and drill down through successively more detailed graphs to isolate the root cause, all within a single platform.

---

## X. Conclusion

Cisco ThousandEyes is, at its core, a large-scale, distributed implementation of applied graph theory. Its agents collect raw network data — ICMP responses, TCP timings, BGP advertisements, SNMP neighbor tables — and assemble it into interconnected graph models that span from individual device interfaces to the global Internet topology. The platform's diagnostic power comes from the graph operations it performs on these models: path discovery, anomaly clustering, centrality analysis, reachability computation, and temporal graph comparison.

The trajectory is clear. The platform is evolving from a system where humans interpret graph visualizations toward one where AI agents autonomously sense, reason, and act on graph-derived telemetry. WAN Insights already demonstrates predictive graph analytics; AgenticOps extends this to closed-loop remediation. As networks grow more complex — 5G edge deployments, multi-cloud architectures, IoT at scale — the graph models will expand accordingly, but the fundamental abstractions remain the same: nodes, edges, weights, paths, and the algorithms that operate on them.

For network engineers, understanding the graph-theoretic foundations of ThousandEyes is not merely academic. It sharpens the interpretation of every visualization the platform produces: recognizing a cut vertex as a single point of failure, reading edge weights as latency contributions, understanding that an Internet Insights outage alert is the result of spatial cluster analysis on a global measurement graph. The graph is the network. ThousandEyes makes it visible.

---

## Appendix

### A. ThousandEyes Test Types and Their Graph Models

| Test Type | Primary Graph Model | Node Type | Edge Type | Key Metrics |
|---|---|---|---|---|
| Agent-to-Server | Path Visualization (directed, weighted) | IP hops | Network segments | Loss, latency, jitter, delay, MTU |
| Agent-to-Agent | Bidirectional Path Visualization | IP hops | Network segments | Loss, latency, jitter, throughput |
| HTTP Server | Path Visualization + HTTP layer | IP hops + server | Network segments | Loss, latency, response time, availability |
| Page Load | Path Visualization + DOM graph | IP hops + page components | Network + resource dependencies | Loss, latency, page load time, DOM load |
| API Test | Path Visualization per API step | IP hops per endpoint | Network segments per call | Loss, latency, API response time, completion |
| DNS Server | Path Visualization to DNS server | IP hops + DNS resolver | Network segments | Loss, latency, resolution time, mappings |
| BGP | AS-level directed graph | Autonomous Systems | Peering/transit relationships | Path changes, reachability, updates, RPKI |
| Device Layer | Undirected adjacency graph | Network devices | Layer-2 links | CPU, memory, interface errors, bandwidth |

### B. Key Metrics Glossary

| Metric | Unit | Description |
|---|---|---|
| **Loss** | % | Percentage of probes that did not receive a response from the target hop |
| **Latency** | ms | Round-trip time from agent to target or to a specific hop |
| **Jitter** | ms | Standard deviation of latency measurements; indicates path stability |
| **Link Delay** | ms | Estimated one-way transmission delay across a single link |
| **DSCP** | Numeric | Differentiated Services Code Point observed in returned packets |
| **MTU** | Bytes | Minimum Maximum Transmission Unit along the path |
| **Reachability** | % | Percentage of BGP monitors that can see a valid route to a prefix |
| **Path Changes** | Count | Number of AS-path modifications observed in a time window |
| **Updates** | Count | Number of BGP UPDATE messages received for a prefix |
| **Throughput** | Mbps | Measured bandwidth capacity (Agent-to-Agent tests) |

### C. ThousandEyes API Endpoints for Graph Data

| Endpoint Category | Data Returned | Use Case |
|---|---|---|
| `/net/path-vis/{testId}` | Path trace nodes, links, per-hop metrics | Reconstruct path graph externally |
| `/net/metrics/{testId}` | End-to-end loss, latency, jitter time series | Trend analysis, SLA reporting |
| `/net/bgp-metrics/{testId}` | AS-paths, reachability, updates, RPKI status | BGP graph construction, hijack detection |
| `/internet-insights/outages` | Outage events with scope, scale, affected providers | Correlation with internal tests |
| `/endpoint-data/network-topology` | Endpoint agent path and network data | Hybrid workforce path analysis |

### D. Recommended Further Reading

1. **ThousandEyes Documentation**: [docs.thousandeyes.com](https://docs.thousandeyes.com) — Comprehensive product documentation including Path Visualization, BGP tests, Device Layer, and API reference.
2. **ThousandEyes Blog**: [thousandeyes.com/blog](https://www.thousandeyes.com/blog) — Technical articles on network monitoring methodology, product updates, and Internet outage analyses.
3. **ThousandEyes API Developer Guide**: [developer.cisco.com/docs/thousandeyes](https://developer.cisco.com/docs/thousandeyes/) — API reference and getting-started guides for programmatic data access.
4. **Cisco Live Sessions**: Annual presentations covering ThousandEyes architecture, new capabilities, and customer case studies.
5. **"Internet Insights: Detecting and Solving Internet Outages with Collective Intelligence"**: ThousandEyes webinar on the algorithms behind Internet Insights outage detection.
6. **RIPE-RIS and RouteViews**: Public BGP data sources that feed ThousandEyes' BGP monitoring — [ris.ripe.net](https://ris.ripe.net) and [routeviews.org](http://www.routeviews.org).
