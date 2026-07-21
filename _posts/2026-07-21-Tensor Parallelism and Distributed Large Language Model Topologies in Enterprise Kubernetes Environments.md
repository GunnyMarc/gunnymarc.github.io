---
title: "Tensor Parallelism and Distributed Large Language Model Topologies in Enterprise Kubernetes Environments"
date: 2026-07-21
permalink: /posts/2026/07/tensor-parallelism-distributed-llm-topologies-enterprise-kubernetes/
tags:
  - tensor parallelism
  - LLMs
  - Kubernetes
  - distributed AI
  - 3D parallelism
  - enterprise
---
#### Summary

Deploying distributed artificial intelligence (AI) workloads—specifically large language models (LLMs) with tens or hundreds of billions of parameters—requires a fundamental departure from traditional high-performance computing (HPC) and standard cloud-native application architectures. When orchestrating these workloads inside an enterprise container platform like Red Hat OpenShift, infrastructure architects and machine learning engineers must carefully align physical hardware topologies, low-latency node networks, containerised resource scheduling, and the underlying mathematical partitions of distributed deep learning frameworks.

By analysing how multi-GPU dense servers (using hardware platforms such as Cisco UCS C885A M8 or Dell PowerEdge XE9680 merely as industry reference archetypes) handle multi-node execution, this article explores the deep technical mechanics of Tensor Parallelism (TP). We contrast this with Pipeline Parallelism (PP) and Data Parallelism (DP), detail the mathematical and network-level communication profiles of these architectures, and map out how to construct a robust, deterministic, and highly performant distributed orchestration layer on Kubernetes.

---

### Understanding the Three Dimensions of 3D Parallelism

Modern neural network architectures, particularly the Decoder-only Transformer models that dominate the LLM landscape, have grown far too large to fit into the memory envelope of a single accelerator. High-bandwidth memory (HBM3/HBM3e) capacities on modern enterprise GPUs typically range from 80 GB to 141 GB per GPU. However, an 8-bit or 16-bit precision training run or serving configuration for a several-hundred-billion-parameter model can easily require hundreds of gigabytes or even terabytes of active memory, excluding the massive footprint needed for optimizer states, gradients, activations, and Key-Value (KV) caches.

To solve this, developers employ **3D Parallelism**, which combines three distinct techniques: Tensor Parallelism, Pipeline Parallelism, and Data Parallelism.

```
                  +----------------------------------------------+
                  |                 Total Job                    |
                  |                (World Size)                  |
                  +----------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
+------------------+                                            +------------------+
|  Data Parallel  |                                            |  Model Parallel  |
|      Group       |                                            |      Group       |
+------------------+                                            +------------------+
                                                                         |
                                                 +-----------------------+-----------------------+
                                                 |                                               |
                                      +--------------------+                          +--------------------+
                                      | Tensor Parallelism |                          |Pipeline Parallelism|
                                      |    (Intra-Node)    |                          |    (Inter-Node)    |
                                      +--------------------+                          +--------------------+
```

#### 1. Tensor Parallelism (TP)

Tensor Parallelism is an intra-layer parallelization technique. Instead of dividing the layers of a network sequentially across multiple accelerators, TP splits individual parameter matrices (such as the Query-Key-Value projection matrices or the multi-layer perceptron blocks in a Transformer layer) across multiple devices.

In a standard Megatron-LM style implementation, Tensor Parallelism is divided into:

- **Column-Parallel Linear Layers:** The weight matrix W of a linear projection (e.g., the input-to-hidden projection in the MLP block) is split vertically along its columns:

W=[W1​∣W2​∣⋯∣Wn​]

Each GPU i receives a slice Wi​ and computes its portion of the output Yi​=XWi​ independently.

- **Row-Parallel Linear Layers:** The weight matrix W of a subsequent projection (e.g., the hidden-to-output projection) is split horizontally along its rows:

W=​W1​W2​⋮Wn​​​

Each GPU must compute a partial result Yi​=Xi​Wi​. To obtain the final output Y=∑Yi​, the GPUs must perform an `all-reduce` operation across the entire TP communication group.

This structural split means that every single forward pass requires an `all-gather` (for column-parallel layers) or an `all-reduce` (for row-parallel layers), and every backward pass requires the inverse operation. Because these collectives occur multiple times within a _single_ Transformer layer, Tensor Parallelism is extremely latency-sensitive. It is structurally bounded by communication bandwidth, which is why it is almost exclusively restricted to high-speed local GPU fabrics like NVLink or Infinity Fabric.

#### 2. Pipeline Parallelism (PP)

Pipeline Parallelism partitions a neural network horizontally on a layer-by-layer basis. For instance, in a 96-layer Transformer, Pipeline Parallelism of degree 3 (PP=3) would assign layers 1–32 to Stage 0 (Server 1), layers 33–64 to Stage 1 (Server 2), and layers 65–96 to Stage 2 (Server 3).

The forward pass progresses sequentially: Stage 0 processes a micro-batch of data and passes the boundary activations to Stage 1, which processes them and passes its activations to Stage 2. The backward pass travels in the exact reverse direction. To prevent the accelerators from remaining idle while waiting for other stages to finish—a phenomenon known as the **pipeline bubble**—advanced scheduling algorithms such as the "1F1B" (One Forward, One Backward) schedule are used.

Compared to TP, PP is much less intensive on communication frequency. It only exchanges activation tensors at the boundaries between stages (e.g., once every 32 layers). Consequently, PP is highly suited for inter-node communication over high-speed networks, where latency is higher than local NVLink buses but throughput is still sufficient for batch transfers.

#### 3. Data Parallelism (DP)

Data Parallelism replicates the entire model (or a specific model segment defined by TP and PP) across multiple parallel worker groups. Each group receives a unique shard of the training micro-batch. During the backward pass, the gradients calculated by each worker are synchronized using an `all-reduce` or `reduce-scatter` operation (or via ZeRO-style distributed optimizer techniques) so that all replicas update their model weights identically.

Because DP synchronization occurs only once per backward step, it can scale across thousands of nodes, provided there is enough network throughput to prevent the optimizer step from bottlenecking.

The product of these three factors defines the overall scale of a running job:

WORLD_SIZE=TP×PP×DP

---

### Comparative Evaluation of 3D Parallelism Topologies

To illustrate the practical tradeoffs of these paradigms, let us analyze a cluster composed of three dense accelerator servers, each containing 8 high-performance GPUs (yielding a absolute pool of 24 GPUs). Hardware like Cisco UCS C885A M8 or Dell PowerEdge XE9680 servers utilize an internal 8-way GPU baseboard where every GPU is directly interconnected via a high-bandwidth proprietary fabric (such as NVIDIA HGX NVLink or AMD Infinity Fabric).

Inter-node connectivity is established through specialized PCI Express or OCP mezzanine Network Interface Cards (NICs), such as 400 Gbps NVIDIA ConnectX-7 or BlueField-3 adapters. This creates two very distinct communication regimes:

1. **Intra-node (Local):** Extreme bandwidth (~900 GB/s bidirectional per GPU), ultra-low latency (< 1 microsecond).
2. **Inter-node (Network):** High bandwidth (typically 400 Gbps / ~50 GB/s per NIC over RoCEv2/IB), slightly higher latency (several microseconds, depending on network hops).

#### Option A: Localized Tensor Parallelism with Cross-Node Pipelining (TP=8,PP=3,DP=1)

This is universally recognized as the optimal topological design for executing a single large model training job on this type of cluster.

```
[ Node 1 / Server 1 ]       [ Node 2 / Server 2 ]       [ Node 3 / Server 3 ]
+-------------------+       +-------------------+       +-------------------+
| GPUs 0 - 7        |       | GPUs 0 - 7        |       | GPUs 0 - 7        |
| TP Group 0        |=====> | TP Group 1        |=====> | TP Group 2        |
| (Pipeline Stg 0)  | RDMA  | (Pipeline Stg 1)  | RDMA  | (Pipeline Stg 2)  |
+-------------------+       +-------------------+       +-------------------+
```

##### Communication Profile

Inside each node, the high-frequency, latency-sensitive TP collectives (`all-reduce` and `all-gather` for the attention and MLP layers) run entirely within the local GPU fabric. These operations never touch the external cluster network.

Between the three nodes, the pipeline boundary activations are moved sequentially from Node 1 to Node 2, and then from Node 2 to Node 3. Since this pipeline-stage activation transport is a point-to-point transfer that occurs only at layer boundaries, the physical network is never saturated by the massive broadcast overhead of high-frequency Tensor Parallelism.

##### Structural Pros

- Highly efficient; minimizes the impact of inter-node latency on active computation steps.
- Robust scaling because the network requirements are heavily throttled by the pipeline schedule.
- Maximizes the utilization of local GPU fabric capabilities.

##### Structural Cons

- Subject to pipeline bubble overhead; some GPUs will inevitably spend clock cycles waiting to receive activations from upstream nodes, though this can be mitigated using interleaved 1F1B scheduling.

---

#### Option B: Extended Cluster-Wide Tensor Parallelism (TP=24,PP=1,DP=1)

In this configuration, all 24 GPUs across all three physical systems are bound together into a singular, massive Tensor Parallelism group.

```
+-----------------------------------------------------------------------------------------+
|                                    Single TP-24 Group                                   |
|                                                                                         |
|   [ Node 1 / Server 1 ]         [ Node 2 / Server 2 ]         [ Node 3 / Server 3 ]     |
|   +-------------------+         +-------------------+         +-------------------+     |
|   | GPUs 0 - 7        | <=====> | GPUs 8 - 15       | <=====> | GPUs 16 - 23      |     |
|   +-------------------+  RoCE   +-------------------+  RoCE   +-------------------+     |
+-----------------------------------------------------------------------------------------+
```

##### Communication Profile

To execute standard layer operations, `all-reduce` and `all-gather` collectives must be broadcast among all 24 ranks. In this scenario, every single matrix operation in the network's forward and backward passes spawns massive packets that must traverse the external inter-node network switches.

While local communication (e.g., between GPUS 0–7) remains on the local fabric, any communication involving the remaining 16 target ranks forces the collective libraries (NCCL or RCCL) to break up the transfers. Ranks are forced to route data out of the server’s physical NICs, across the network switch, and into the adjacent servers.

##### Structural Pros

- Allows a single enormously wide model instantiation without the scheduling baggage, memory caching, or mathematical complexity of Pipeline Parallelism.
- Can be useful if a single layer’s weight matrix or activation tensors are so large that they cannot physically fit into the 8-GPU memory pool of a single node.

##### Structural Cons

- Severe performance degradation in most network classes. External inter-node switch latency is orders of magnitude higher than local GPU interconnects. Every linear layer step bottlenecks as it waits for the network to resolve `all-reduce` passes across the nodes.
- Highly vulnerable to packet loss, packet spray issues, and port congestion on the physical network switches.

---

#### Option C: Replicated Localized Tensor Parallelism (TP=8,PP=1,DP=3)

For high-throughput serving and online inference deployment patterns, this topology provides the most robust and operationally resilient architectural framework.

```
       [ Node 1 / Server 1 ]                  [ Node 2 / Server 2 ]                  [ Node 3 / Server 3 ]
       +-------------------+                  +-------------------+                  +-------------------+
       |    GPUs 0 - 7     |                  |    GPUs 0 - 7     |                  |    GPUs 0 - 7     |
       |  Inference Rep 1  |                  |  Inference Rep 2  |                  |  Inference Rep 3  |
       +-------------------+                  +-------------------+                  +-------------------+
                 ^                                      ^                                      ^
                 |                                      |                                      |
                 +--------------------------------------+--------------------------------------+
                                                        |
                                              [ Load Balancer Pod ]
```

##### Communication Profile

Each server operates as a completely independent, self-contained model replica. There is absolutely zero inter-node communication required for active token generation during the inference cycle. The only network activity is the initial model weight loading phase and incoming query routing from a Kubernetes load balancer.

##### Structural Pros

- Perfect linear scaling of aggregate throughput (queries/tokens processed per second) as servers are added.
- Complete fault isolation: if one node suffers a hardware panic or a GPU failure, the remaining two nodes continue processing user queries without interruption.
- Extremely low query response latencies since no inter-node communication overhead is introduced during the attention phases.

##### Structural Cons

- Limited by the physical memory of a single server. If the model size exceeds the composite HBM capacity of an 8-GPU node, this option becomes unviable without introducing quantization or pipeline partitioning.

---

### Low-Level Network Mechanics and Collective Communication

To understand why custom topology designs are necessary, we must examine what happens at the network layer during a collective operation. When multiple nodes participate in an `all-reduce` or `all-gather` over an Ethernet-based infrastructure, they cannot rely on standard TCP/IP networking. TCP's kernel-space overhead, interrupt processing, and packet acknowledgement retries impose latency penalties that completely stall distributed AI steps.

Instead, distributed training and inference rely on **Remote Direct Memory Access (RDMA)** over Converged Ethernet (RoCEv2) or InfiniBand.

```
+-------------------------------------------------------------------------+
|                               Local Host                                |
|   +---------------+     +---------------+                               |
|   |  GPU Memory   |     |  System RAM   |                               |
|   |  (HBM Space)  |     | (Kernel bypassed)                            |
|   +---------------+     +---------------+                               |
|           |                     |                                       |
|           +----------+----------+                                       |
|                      | (GPUDirect RDMA / Direct PCIe Access)            |
|                      v                                                  |
|            +--------------------+                                       |
|            |  Network Adapter   |                                       |
|            | (RoCEv2 / IB NIC)  |                                       |
+------------+--------------------+---------------------------------------+
                       |
                       |  (RoCEv2 Packets via UDP/PFC/ECN)
                       v
            +--------------------+
            | Network Switch L2  |
            +--------------------+
```

#### The Role of GPUDirect RDMA

In a typical network pipeline, moving data from a GPU on Node A to a GPU on Node B requires copying the data from GPU memory (HBM) to system memory (RAM), handing it over to the CPU kernel network stack, pushing it over the network card, and then reversing the entire copy sequence on the receiving end.

GPUDirect RDMA bypasses the host CPU and system RAM entirely. The Network Interface Card (NIC) directly reads and writes to the local HBM via the PCIe bus. When coordinated by communication libraries such as NVIDIA's NCCL or AMD's RCCL, multiple physical network adapters can be paired directly with physical GPUs in a 1:1 mapping. In high-end design layouts, a node with 8 GPUs will feature 8 dedicated single-port NICs, ensuring that each GPU has a dedicated, non-blocking path out to the network fabric.

#### Network Congestion and Lossless Ethernet

Since RoCEv2 is encapsulated inside standard UDP/IP packets, it operates over standard Ethernet switches. However, RDMA is fundamentally designed on the assumption of a lossless network layer. If a switch drops a single packet due to buffer exhaustion (e.g., during a synchronized `all-reduce` incast), the entire collective communication must be retransmitted, causing severe performance drops.

To keep the network lossless and performant, several low-level switch mechanisms must be configured:

- **Priority Flow Control (PFC - IEEE 802.1Qbb):** This link-level flow control mechanism operates on specific Class of Service (CoS) values. When a switch port's receive buffer exceeds a designated watermark, it broadcasts a "pause frame" back to the transmitter on that specific traffic class, pausing only the RDMA traffic while letting standard control plane traffic flow normally.
- **Explicit Congestion Notification (ECN - RFC 3168):** Switches mark the IP headers of packets when congestion watermarks are breached. The receiving NIC detects these marks and sends a Congestion Notification Packet (CNP) back to the sender, prompting the sending GPU/NIC to throttle its output rate before buffers overflow and trigger PFC pauses.

---

### Architecting Distributed AI on Red Hat OpenShift

Executing these massive 3D parallel workloads inside a Kubernetes-native framework like Red Hat OpenShift requires advanced customization of both the control plane and the compute node configurations. OpenShift must be tailored to ensure that the physical hardware attributes are exposed directly to the container runtime, and that workloads are scheduled with absolute topological awareness.

#### 1. Node Isolation and Custom Tuning Profiles

Dense GPU systems must be placed on bare-metal worker nodes using specialized OpenShift CoreOS (RHCOS) configurations to eliminate hypervisor-induced latency overhead. These systems are isolated using specialized Kubernetes node labels and taints:

```yaml
apiVersion: v1
kind: Node
metadata:
  labels:
    node-role.kubernetes.io/gpu-worker: ""
    feature.node.kubernetes.io/pci-neonetwork.present: "true"
spec:
  taints:
  - effect: NoSchedule
    key: nvidia.com/gpu
    value: "present"
```

To ensure optimal packet processing and host-side execution times, the **Node Tuning Operator (NTO)** is deployed to configure kernel parameters. This includes disabling CPU frequency scaling (enabling performance mode), configuring hugepages (typically 2MB page sizes for local memory optimization), and setting host network buffers:

```ini
# Custom sysctl configurations applied by OpenShift NTO
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
```

#### 2. GPU and Network Device Enablement

Instead of manually installing container runtimes and hardware drivers, OpenShift leverages automated operators to maintain state and perform orchestration tasks:

- **GPU Operator:** Automatically provisions the host's specialized driver modules, registers the Kubernetes device plugins so containers can specifically request resource counts like `nvidia.com/gpu: 8`, configures the container runtime hook to pass GPU context, and runs DCGM (Data Center GPU Manager) monitoring daemons.
- **Network Operator & SR-IOV Operator:** To make GPUDirect RDMA functional inside a container, the network cards must bypass the standard Kubernetes overlay network (e.g., OVN-Kubernetes). The **SR-IOV Network Operator** is configured to create virtual functions (VFs) from the physical host NIC ports, or the **Mellanox Network Operator** is used to pass physical RoCEv2 networks directly into the container namespace via the Multus CNI. This ensures the container directly talks to the host NIC without an intervening abstract network layer.

#### 3. Topology-Aware Scheduling and Gang Placement

Two major issues often disrupt distributed AI jobs in traditional Kubernetes environments: scheduling fragmentation and alignment failures.

##### Gang Scheduling

If a user launches a 3-node, 24-GPU model-parallel training run but OpenShift only has 2 nodes free, a standard Kubernetes scheduler will launch 16 pods on the free machines and stick the remaining 8 in a `Pending` state. The first two stages will spin up, block waiting to establish TCP/RDMA control plane handshakes, and hang indefinitely—wasting active GPU capacity.

To prevent this, OpenShift utilizes integrated queueing systems like **Kueue** or scheduler plug-ins like **Coscheduling** (often orchestrated via the Kubeflow Training Operator). These schedule the pods as an indivisible group (a gang). If all 24 GPU slots are not simultaneously available, the job is held in a queue, keeping the running cluster free for other workloads.

##### Topology-Aware Scheduling (NUMA & PCIe Alignment)

Modern multi-socket CPU systems have complex internal layouts. A server may align physical PCIe Slots 1–4 directly to CPU Socket 0, while physical slots 5–8 align to CPU Socket 1. If a containerized process is scheduled onto CPU cores on Socket 0 but requests access to a GPU or a NIC wired to Socket 1, every memory access must cross the slow inter-socket Interconnect (Ultra Path Interconnect / UPI).

The **NUMA-Aware Scheduler** or the **Topology Manager** on OpenShift must be set to a strict `Single-NUMA-Node` policy to guarantee that the containerized CPU cores, allocated memory, virtual GPU adapters, and high-performance network interfaces are physically co-located on the exact same PCIe root complex.

---

### Concrete Kubernetes Workload Configurations

To instantiate a practical multi-node execution job on the cluster, developers use distributed training controllers. The example below details an standard **MPIJob** custom resource schema configured for our recommended topology **Option A (TP=8,PP=3)**.

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: distributed-llm-training
  namespace: ai-workloads
spec:
  slotsPerWorker: 8
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: mpi-launcher
            image: quay.io/ai-enterprise/distributed-training:latest
            command:
            - mpirun
            - -np
            - "24"
            - --allow-run-as-root
            - -x
            - NCCL_DEBUG=INFO
            - -x
            - NCCL_IB_DISABLE=0
            - -x
            - NCCL_NET_GDR_LEVEL=5
            - python
            - train_llm.py
            - --model_path=/mnt/shared/models/llama3-70b
            - --tensor_parallel_size=8
            - --pipeline_parallel_size=3
    Worker:
      replicas: 3
      template:
        metadata:
          labels:
            ai-job: llama3-70b-training
        spec:
          containers:
          - name: mpi-worker
            image: quay.io/ai-enterprise/distributed-training:latest
            resources:
              limits:
                nvidia.com/gpu: "8"
                memory: "512Gi"
                cpu: "64"
                mellanox.com/mlnx_sriov_rdma: "8"
              requests:
                nvidia.com/gpu: "8"
                memory: "512Gi"
                cpu: "64"
                mellanox.com/mlnx_sriov_rdma: "8"
            securityContext:
              capabilities:
                add: ["IPC_LOCK"]
            volumeMounts:
            - name: model-storage
              mountPath: /mnt/shared
            - name: dshm
              mountPath: /dev/shm
          volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: shared-storage-pvc
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "256Gi"
```

In this manifest:

- `slotsPerWorker: 8` and a worker replica count of `3` maps directly to our 3 physical servers.
- Each worker container asks for `nvidia.com/gpu: 8` and `mellanox.com/mlnx_sriov_rdma: 8` to map each GPU to its corresponding non-blocking host RDMA network path.
- `NCCL_NET_GDR_LEVEL=5` configures NCCL to use the highest level of GPUDirect RDMA, allowing it to bypass system memory and route packets directly across the PCIe switch to the NIC.
- `IPC_LOCK` is requested in the container's security context to allow the training libraries to lock physical memory pages, preventing operating system page-swapping from interrupting execution loops.
- `/dev/shm` is mounted using an `emptyDir` backed by RAM memory to allow fast host-level process communication (high-speed shared memory) between the 8 local GPU container worker processes.

---

### Step-by-Step Deployment Roadmap

Deploying a complex, multi-node configuration requires a structured verification process to pinpoint network bottle-necks or scheduling misalignments before launching production jobs.

#### Step 1: Establish Single-Node Baselines

Before introducing network-based complexity, validate that each individual server is operating at peak physical capacity. Run localized micro-benchmarks inside the containers to confirm raw hardware execution times.

- Run a standard matrix multiplication benchmark to measure raw Tensor Core performance.
- Run a local NCCL or RCCL test (`all_reduce_perf`) strictly within the node to confirm that the intra-node GPU fabric (NVLink/Infinity Fabric) is achieving expected speeds (e.g., ~900 GB/s on H100 architectures):
    
    ```bash
    ./all_reduce_perf -b 8 -e 256M -f 2 -g 8
    ```
    

#### Step 2: Configure and Verify Lossless Networking

Work with your network operations team to implement Priority Flow Control (PFC) and Explicit Congestion Notification (ECN) on the physical L2/L3 switch infrastructure. Once the switches are configured, confirm their operational status:

- Query the host operating system to confirm the RDMA driver detects the correct link rate (e.g., 400 Gbps).
- Inspect raw switch ports and interface counters during initial data transfers to ensure that no packets are dropped and that pause frames are registering correctly when congestion thresholds are tested.

#### Step 3: Run Multi-Node Collective Benchmarks

With physical networking verified, launch a synthetic Kubernetes benchmark across two, and then three, nodes. By running multi-node NCCL tests, you can measure exactly how performance changes as communication scales out.

- Launch a multi-node MPI test, comparing `all_reduce_perf` across 16 GPUs (2 nodes) and 24 GPUs (3 nodes).
- Verify that the bandwidth matches your network design. For a single 400 Gbps link, the inter-node out-of-box transfer rate should reliably yield around 40–45 GB/s of active data transport.

#### Step 4: Run the Final Application Workload

Only when all physical benchmarks and collective benchmarks show nominal latency and throughput should you deploy your distributed model architecture (e.g., Option A).

- Enable detailed logging metrics in your training framework to capture training steps and execution metrics.
- Monitor GPU utilization (using DCGM metrics in OpenShift) and verify that "GPU Active" percentages remain high (>80%), while "communication wait time" or "network throttled time" metrics remain low. This confirms that your 3D parallelism topology matches the physical capabilities of your cluster.

---
The primary equipment referenced in this article consists of high-density AI servers—such as the Cisco UCS C885A M8 or Dell PowerEdge XE9680—configured with an internal 8-way GPU baseboard. These servers leverage high-bandwidth proprietary interconnects, like NVIDIA HGX NVLink to enable ultra-fast, direct communication between all local accelerators.

---
#TensorParallelism #DistributedAI #LargeLanguageModels #DeepLearning #ModelParallelism #PipelineParallelism #3DParallelism #RedHatOpenShift #Kubernetes #CloudNativeAI #BareMetal #GPUOperator #ContainerOrchestration #AIInfrastructure #NVIDIAnvlink #HGX #AMDInfinityFabric #CiscoUCS #DellPowerEdge #AcceleratedComputing #GPUDirectRDMA #RoCEv2 #InfiniBand #LosslessEthernet #HighPerformanceComputing

