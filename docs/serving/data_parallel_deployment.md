# Data Parallel Deployment

vLLM supports Data Parallel deployment, where model weights are replicated across separate instances/GPUs to process independent batches of requests.

This will work with both dense and MoE models.

For MoE models, particularly those like DeepSeek that employ MLA (Multi-head Latent Attention), it can be advantageous to use data parallel for the attention layers and expert or tensor parallel (EP or TP) for the expert layers.

In these cases, the data parallel ranks are not completely independent. Forward passes must be aligned, and expert layers across all ranks are required to synchronize during every forward pass, even when there are fewer requests to be processed than DP ranks.

By default, expert layers form a tensor parallel group of size `DP × TP`. To use expert parallelism instead, include the `--enable-expert-parallel` CLI arg (on all nodes in the multi-node case). See [Expert Parallel Deployment](expert_parallel_deployment.md) for details on how attention and expert layers behave differently with EP enabled.

In vLLM, each DP rank is deployed as a separate "core engine" process that communicates with front-end process(es) via ZMQ sockets. Data Parallel attention can be combined with Tensor Parallel attention, in which case each DP engine owns a number of per-GPU worker processes equal to the configured TP size.

For MoE models, when any requests are in progress in any rank, we must ensure that empty "dummy" forward passes are performed in all ranks that don't currently have any requests scheduled. This is handled via a separate DP Coordinator process that communicates with all ranks, and a collective operation performed every N steps to determine when all ranks become idle and can be paused. When TP is used in conjunction with DP, expert layers form a group of size `DP × TP` (using either tensor parallelism by default, or expert parallelism if `--enable-expert-parallel` is set).

### Independent MoE ranks

With `--data-parallel-independent-moe`, the data parallel ranks of an MoE model run as those of a dense model do: each rank holds every expert (its MoE layers are tensor parallel within the rank only), runs its own engine loop, never performs dummy forward passes and never takes part in a collective with another rank. There is no wave coordination, and the DP Coordinator runs only where the internal load balancer needs its statistics. This suits a model that fits one rank's GPUs, served on nodes whose GPUs cannot reach each other quickly (or at all), where a per-layer all-gather across ranks would cost more than it saves.

```bash
vllm serve $MODEL --data-parallel-size 2 --data-parallel-independent-moe
```

It cannot be combined with `--enable-expert-parallel`, EPLB, elastic EP, fault tolerance or a non-default `--all2all-backend`, all of which shard or coordinate experts across ranks.

Co-located ranks still share what they can on the host: an Engram/PLE table offloaded with `dp_shared_memory` (see [Engram](../features/engram.md)) is mapped once per node. Because the ranks have no DP group, the co-located ranks form a CPU-only group (a loopback TCPStore, no torch process group) while the table loads, the first of them loading it, and close it once the model is loaded. The ranks a launcher starts on a node share a table, and so do the external-LB ranks that `--nnodes` places on one node (`--data-parallel-size / --nnodes` consecutive ranks per node, so `--data-parallel-multi-port-external-lb` on one node shares one table between its children). Ray-placed ranks keep a table each. The loading group listens on the loopback ports after `--data-parallel-rpc-port` (one per worker of a rank; move them with that option when two deployments share a host). Co-located ranks wait for each other at most five minutes; a peer that does not come (a rank restarted while the others serve, or `--nnodes` not describing an external-LB deployment) leaves each rank its own table.

Independent ranks work with every load balancing mode below. With internal load balancing, a router can still pin a session to a rank by setting the `X-data-parallel-rank` request header; with external load balancing, each rank's server starts and serves on its own, without contacting rank 0.

In all cases, it is beneficial to load-balance requests between DP ranks. For online deployments, this balancing can be optimized by taking into account the state of each DP engine - in particular its currently scheduled and waiting (queued) requests, and KV cache state. Each DP engine has an independent KV cache, and the benefit of prefix caching can be maximized by directing prompts intelligently.

This document focuses on online deployments (with the API server). DP + EP is also supported for offline usage (via the LLM class), for an example see [examples/features/data_parallel/data_parallel_offline.py](../../examples/features/data_parallel/data_parallel_offline.py).

There are two distinct modes supported for online deployments - self-contained with internal load balancing, or externally per-rank process deployment and load balancing.

## Internal Load Balancing

vLLM supports "self-contained" data parallel deployments that expose a single API endpoint.

It can be configured by simply including e.g. `--data-parallel-size=4` in the vllm serve command line arguments. This will require 4 GPUs. It can be combined with tensor parallel, for example `--data-parallel-size=4 --tensor-parallel-size=2`, which would require 8 GPUs. When sizing DP deployments, remember that `--max-num-seqs` applies per DP rank.

By contrast, the admission control limits `--max-num-queued-reqs` and `--max-num-queued-tokens` apply to the whole server, not per DP rank: each API server process counts in-flight requests and prefill backlog across all DP ranks it routes to. For example, with `--data-parallel-size=4 --max-num-seqs=256 --max-num-queued-reqs=256`, the deployment rejects new requests once 256 are in-flight in total, even though ranks could jointly run 1024. To keep ranks saturating, size the cap as roughly `data-parallel-size * max-num-seqs` plus the desired queue depth.

Running a single data parallel deployment across multiple nodes requires a different `vllm serve` to be run on each node, specifying which DP ranks should run on that node. In this case, there will still be a single HTTP entrypoint - the API server(s) will run only on one node, but it doesn't necessarily need to be co-located with the DP ranks.

This will run DP=4, TP=2 on a single 8-GPU node:

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

This will run DP=4 with DP ranks 0 and 1 on the head node and ranks 2 and 3 on the second node:

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

This will run DP=4 with only the API server on the first node and all engines on the second node:

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

This DP mode can also be used with Ray by specifying `--data-parallel-backend=ray`:

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

There are several notable differences when using Ray:

- A single launch command (on any node) is needed to start all local and remote DP ranks, therefore it is more convenient compared to launching on each node
- There is no need to specify `--data-parallel-address`, and the node where the command is run is used as `--data-parallel-address`
- There is no need to specify `--data-parallel-rpc-port`
- When a single DP group requires multiple nodes, *e.g.* in case a single model replica needs to run on at least two nodes, make sure to set `VLLM_RAY_DP_PACK_STRATEGY="span"` in which case `--data-parallel-size-local` is ignored and will be automatically determined
- Remote DP ranks will be allocated based on node resources of the Ray cluster

Currently, the internal DP load balancing is done within the API server process(es) and is based on the running and waiting queues in each of the engines. This could be made more sophisticated in future by incorporating KV cache aware logic.

When deploying large DP sizes using this method, the API server process can become a bottleneck. In this case, the orthogonal `--api-server-count` command line option can be used to scale this out (for example `--api-server-count=4`). This is transparent to users - a single HTTP endpoint / port is still exposed. Note that this API server scale-out is "internal" and still confined to the "head" node.

<figure markdown="1">
![DP Internal LB Diagram](../assets/deployment/dp_internal_lb.png)
</figure>

## Hybrid Load Balancing

Hybrid load balancing sits between the internal and external approaches. Each node runs its own API server(s) that only queue requests to the data-parallel engines colocated on that node. An upstream load balancer (for example, an ingress controller or traffic router) spreads user requests across those per-node endpoints.

Enable this mode with `--data-parallel-hybrid-lb` while still launching every node with the global data-parallel size. The key differences from internal load balancing are:

- You must provide `--data-parallel-size-local` and `--data-parallel-start-rank` so each node knows which ranks it owns.
- Not compatible with `--headless` since every node exposes an API endpoint.
- Scale `--api-server-count` per node based on the number of local ranks

In this configuration, each node keeps scheduling decisions local, which reduces cross-node traffic and avoids single node bottlenecks at larger DP sizes.

## External Load Balancing

For larger scale deployments especially, it can make sense to handle the orchestration and load balancing of data parallel ranks externally.

In this case, it's more convenient to treat each DP rank like a separate vLLM deployment, with its own endpoint, and have an external router balance HTTP requests between them, making use of appropriate real-time telemetry from each server for routing decisions.

This can already be done trivially for non-MoE models, since each deployed server is fully independent. In that case, launch independent vLLM instances without any `--data-parallel-*` arguments; external DP CLI options are only supported for MoE deployments.

We support an equivalent topology for MoE DP+EP which can be configured via the following CLI arguments.

If DP ranks are co-located (same node / ip address), a default RPC port is used, but a different HTTP server port must be specified for each rank:

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

For multi-node cases, the address/port of rank 0 must also be specified:

```bash
# Rank 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

When using `--data-parallel-external-lb` with `--nnodes`, the data parallel rank
can be inferred from `--node-rank` only when `--nnodes` is evenly divisible by
`--data-parallel-size`. If it is not evenly divisible, automatic rank inference
is rejected and `--data-parallel-rank` must be set explicitly. When multiple
external-LB ranks are co-located on a node, assign each process a disjoint
device set with `CUDA_VISIBLE_DEVICES` or `--device-ids`; devices are not
partitioned automatically between processes.

The coordinator process also runs in this scenario, co-located with the DP rank 0 engine.

<figure markdown="1">
![DP External LB Diagram](../assets/deployment/dp_external_lb.png)
</figure>

In the above diagram, each of the dotted boxes corresponds to a separate launch of `vllm serve` - these could be separate Kubernetes pods, for example.
