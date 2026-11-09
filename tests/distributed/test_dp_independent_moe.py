# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data parallel MoE ranks that run independently, without expert parallelism.

With `--data-parallel-independent-moe` each DP rank of an MoE model holds every
expert and runs as dense DP ranks run: its own engine loop, no dummy batches,
no all-gather or reduce-scatter across ranks in the MoE layers. Co-located
ranks still map one CPU-offloaded PLE table, over a CPU-only group they form
while the table loads and close afterwards.
"""

import mmap
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import EngramConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.distributed import cleanup_dist_env_and_memory, parallel_state
from vllm.distributed.parallel_state import (
    NodeLocalLoadGroup,
    get_engram_table_group,
    init_distributed_environment,
    release_engram_load_group,
)
from vllm.model_executor.layers.fused_moe import all2all_utils
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.models.deepseek_v41.common import engram as engram_ops
from vllm.utils.network_utils import get_open_port, get_open_ports_list
from vllm.utils.system_utils import update_environment_variables

QWEN = "Qwen4ExpForCausalLM"


def _independent(**kwargs) -> ParallelConfig:
    kwargs.setdefault("data_parallel_size", 2)
    kwargs.setdefault("data_parallel_size_local", kwargs["data_parallel_size"])
    kwargs.setdefault("distributed_executor_backend", "mp")
    return ParallelConfig(
        data_parallel_independent_moe=True, is_moe_model=True, **kwargs
    )


# --- Configuration -----------------------------------------------------------


@pytest.mark.parametrize(
    "option,match",
    [
        ({"enable_expert_parallel": True}, "--enable-expert-parallel"),
        ({"all2all_backend": "deepep_low_latency"}, "--all2all-backend"),
        ({"all2all_backend": "alltoall_batched"}, "--all2all-backend"),
        ({"enable_fault_tolerance": True}, "--enable-fault-tolerance"),
    ],
)
def test_independent_moe_refuses_expert_parallel_layouts(option, match):
    with pytest.raises(ValueError, match=match):
        _independent(**option)


def test_independent_moe_names_every_conflict():
    with pytest.raises(ValueError) as exc_info:
        _independent(enable_expert_parallel=True, all2all_backend="mori")
    message = str(exc_info.value)
    assert "--enable-expert-parallel" in message
    assert "--all2all-backend mori" in message
    assert "Drop --data-parallel-independent-moe" in message


@pytest.mark.parametrize(
    "independent,is_moe,expected",
    [
        (False, True, True),
        (False, None, True),
        (False, False, False),
        (True, True, False),
        (True, None, False),
        (True, False, False),
    ],
)
def test_dp_lockstep(independent, is_moe, expected):
    config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_independent_moe=independent,
        distributed_executor_backend="mp",
    )
    assert config.uses_dp_lockstep(is_moe) is expected


def test_loading_ports_follow_the_rpc_port_whatever_vllm_port_says(monkeypatch):
    """A probed port can be the one the executor binds next (VLLM_PORT)."""
    monkeypatch.setenv("VLLM_PORT", "40000")
    config = _independent(data_parallel_rpc_port=31000)
    assert config._independent_dp_group_ports == [31001]
    assert config._data_parallel_master_port_list[0] >= 40000


def test_external_lb_loading_ports_follow_the_rpc_port():
    """External-LB ranks are separate servers: their ports need no exchange."""
    configs = [
        _independent(
            data_parallel_size_local=1,
            data_parallel_rank=rank,
            data_parallel_external_lb=True,
            data_parallel_rpc_port=31000,
        )
        for rank in range(2)
    ]
    assert [c._independent_dp_group_ports for c in configs] == [[31001], [31001]]


def test_no_loading_ports_without_the_flag():
    config = ParallelConfig(data_parallel_size=2, distributed_executor_backend="mp")
    assert config._independent_dp_group_ports == []


def test_reconfigure_keeps_the_node_group_of_an_internal_lb_rank():
    config = _independent(data_parallel_size=4, data_parallel_size_local=2)
    config.data_parallel_rank_local = 1
    config.reconfigure_for_independent_dp_rank()
    assert config._independent_dp_node_group == (1, 2)
    assert config.data_parallel_size == 1
    assert config.data_parallel_size_local == 1
    assert config.data_parallel_rank == 0
    # No MoE call gathers other ranks' tokens any more.
    assert config.elastic_ep_max_dp_size == 1


@pytest.mark.parametrize(
    "dp_size,nnodes,rank,expected",
    [
        (2, 1, 1, (1, 2)),  # One node: every rank is co-located.
        (4, 2, 3, (1, 2)),  # Two per node, in order.
        (2, 2, 1, None),  # One per node: nothing to share with.
        (3, 2, 0, None),  # Uneven: co-location is not knowable.
    ],
)
def test_reconfigure_infers_the_node_group_of_an_external_lb_rank(
    dp_size, nnodes, rank, expected
):
    config = _independent(
        data_parallel_size=dp_size,
        data_parallel_size_local=1,
        data_parallel_rank=rank,
        data_parallel_external_lb=True,
        nnodes=nnodes,
    )
    config.reconfigure_for_independent_dp_rank()
    assert config._independent_dp_node_group == expected


def test_reconfigure_leaves_ray_and_dense_ranks_ungrouped():
    ray = _independent(data_parallel_backend="ray")
    ray.data_parallel_rank_local = 1
    ray.reconfigure_for_independent_dp_rank()
    assert ray._independent_dp_node_group is None

    dense = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        distributed_executor_backend="mp",
    )
    dense.data_parallel_rank_local = 1
    dense.reconfigure_for_independent_dp_rank()
    assert dense._independent_dp_node_group is None


def _vllm_config_view(parallel_config, is_moe=True):
    return SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(is_moe=is_moe),
    )


@pytest.mark.parametrize(
    "independent,external_lb,expected",
    [
        (False, False, True),
        (False, True, True),  # Lockstep: the coordinator runs the waves.
        (True, False, True),  # Internal LB still balances on its stats.
        (True, True, False),  # Nothing to coordinate between servers.
    ],
)
def test_needs_dp_coordinator(independent, external_lb, expected):
    parallel_config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=1 if external_lb else 2,
        data_parallel_external_lb=external_lb,
        data_parallel_independent_moe=independent,
        distributed_executor_backend="mp",
    )
    view = _vllm_config_view(parallel_config)
    assert VllmConfig.needs_dp_coordinator.fget(view) is expected


def test_engram_config_still_verifies_on_a_reset_independent_rank():
    """The engine re-runs VllmConfig.__post_init__ after its DP reset; a rank
    that shares the table with co-located peers must pass it."""
    config = _independent(data_parallel_size=2, data_parallel_size_local=2)
    engram = EngramConfig(cpu_offload=True, dp_shared_memory=True)
    engram.verify_parallel_config(config)
    config.data_parallel_rank_local = 1
    config.reconfigure_for_independent_dp_rank()
    assert config.data_parallel_size == 1
    engram.verify_parallel_config(config)
    # Without a co-located peer, DP=1 still refuses it.
    alone = ParallelConfig(distributed_executor_backend="mp")
    with pytest.raises(ValueError, match="data_parallel_size > 1"):
        engram.verify_parallel_config(alone)


# --- Engine core and frontend selection --------------------------------------


class _Chosen(Exception):
    pass


def _fake_engine(name):
    class Fake:
        def __init__(self, *args, **kwargs):
            raise _Chosen(name, kwargs.get("engine_index"))

    return Fake


@pytest.mark.parametrize(
    "independent,expected",
    [(False, ("lockstep", None)), (True, ("independent", 1))],
)
def test_engine_core_path(monkeypatch, independent, expected):
    from vllm.v1.engine import core

    run_engine_core = core.EngineCoreProc.run_engine_core
    monkeypatch.setattr(core, "DPEngineCoreProc", _fake_engine("lockstep"))
    monkeypatch.setattr(core, "EngineCoreProc", _fake_engine("independent"))
    parallel_config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_independent_moe=independent,
        is_moe_model=True,
        distributed_executor_backend="mp",
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(is_moe=True),
        kv_transfer_config=None,
    )
    with pytest.raises(_Chosen) as exc_info:
        run_engine_core(vllm_config=vllm_config, dp_rank=1, local_dp_rank=1)
    assert exc_info.value.args == expected
    if independent:
        assert parallel_config.data_parallel_size == 1
        assert parallel_config.data_parallel_index == 1
        assert parallel_config._independent_dp_node_group == (1, 2)
    else:
        assert parallel_config.data_parallel_size == 2


@pytest.mark.parametrize(
    "independent,external_lb,expected",
    [
        (False, True, "DPAsyncMPClient"),
        (True, True, "AsyncMPClient"),
        (False, False, "DPLBAsyncMPClient"),
        (True, False, "DPLBAsyncMPClient"),
    ],
)
def test_frontend_client(monkeypatch, independent, external_lb, expected):
    from vllm.v1.engine import core_client

    for name in ("AsyncMPClient", "DPAsyncMPClient", "DPLBAsyncMPClient"):
        monkeypatch.setattr(core_client, name, lambda *a, _n=name, **k: _n)
    parallel_config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=1 if external_lb else 2,
        data_parallel_external_lb=external_lb,
        data_parallel_independent_moe=independent,
        distributed_executor_backend="mp",
    )
    client = core_client.EngineCoreClient.make_async_mp_client(
        _vllm_config_view(parallel_config), executor_class=None, log_stats=False
    )
    assert client == expected


# --- MoE layers --------------------------------------------------------------


@pytest.mark.parametrize("independent", [False, True])
def test_moe_prepare_finalize_skips_the_dp_gather(monkeypatch, independent):
    """Lockstep DP without EP all-gathers tokens across ranks in every MoE
    layer; an independent rank's layers see DP=1 and keep their tokens."""
    parallel_config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_independent_moe=independent,
        is_moe_model=True,
        distributed_executor_backend="mp",
    )
    if independent:
        parallel_config.reconfigure_for_independent_dp_rank()
    # The worker's DP group has the configured size.
    dp_size = parallel_config.data_parallel_size
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.config.get_dp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=dp_size),
    )
    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=1,
        pcp_size_=1,
        dp_size_=dp_size,
        sp_size_=1,
        vllm_parallel_config=parallel_config,
    )
    monkeypatch.setattr(
        all2all_utils,
        "make_moe_prepare_and_finalize_naive_dp_ep",
        lambda **kwargs: "allgather_reducescatter",
    )
    monkeypatch.setattr(
        all2all_utils,
        "make_moe_prepare_and_finalize_no_dp_ep",
        lambda use_monolithic: "local",
    )
    prepare_finalize = all2all_utils.maybe_make_prepare_finalize(
        SimpleNamespace(moe_parallel_config=moe_parallel_config, moe_backend="auto"),
        quant_config=None,
        allow_new_interface=True,
        all2all_manager=SimpleNamespace(world_size=2),
    )
    assert moe_parallel_config.dp_size == (1 if independent else 2)
    assert prepare_finalize == ("local" if independent else "allgather_reducescatter")


# --- The table-loading group, with real processes ----------------------------


def _engram_view(parallel_config, architecture=QWEN, **engram_options):
    engram_options.setdefault("cpu_offload", True)
    engram_options.setdefault("dp_shared_memory", True)
    return SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(architecture=architecture, is_moe=True),
        engram_config=EngramConfig(**engram_options),
    )


@pytest.mark.parametrize(
    "independent,architecture,options,node_group",
    [
        (False, QWEN, {}, (0, 2)),  # Lockstep ranks use the Engram DP group.
        (True, "DeepseekV41ForCausalLM", {}, (0, 2)),  # Shards at runtime.
        (True, QWEN, {"embedding_across_dp": True}, (0, 2)),  # Spread over DP.
        (True, QWEN, {"dp_shared_memory": False}, (0, 2)),
        (True, QWEN, {}, None),  # No co-located peer.
    ],
)
def test_no_loading_group_where_it_does_not_apply(
    monkeypatch, independent, architecture, options, node_group
):
    monkeypatch.setattr(parallel_state, "_ENGRAM_DP", None)
    monkeypatch.setattr(parallel_state, "_ENGRAM_LOAD", None)
    monkeypatch.setattr(parallel_state, "_ENGRAM_LOAD_TRIED", False)
    monkeypatch.setattr(
        parallel_state.StatelessProcessGroup,
        "create",
        lambda **kwargs: pytest.fail("no group may be formed"),
    )
    parallel_config = ParallelConfig(
        data_parallel_independent_moe=independent, distributed_executor_backend="mp"
    )
    parallel_config._independent_dp_node_group = node_group
    parallel_config._independent_dp_group_ports = [get_open_port()]
    with set_current_vllm_config(
        _engram_view(parallel_config, architecture, **options)
    ):
        assert get_engram_table_group() is None


def _rank_config(index: int, size: int, ports: list[int], timeout: int):
    parallel_config = ParallelConfig(
        data_parallel_independent_moe=True,
        distributed_executor_backend="mp",
        cpu_distributed_timeout_seconds=timeout,
    )
    parallel_config._independent_dp_node_group = (index, size)
    parallel_config._independent_dp_group_ports = ports
    return _engram_view(parallel_config)


def _init_rank_world(world_port: int) -> None:
    """Each independent rank is a world of its own, as after the DP reset."""
    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(world_port),
        }
    )
    init_distributed_environment(backend="gloo")


def _patch_cuda_registration(monkeypatch):
    """Let `share_shm_table` run on CPU: registration succeeds and pins."""
    ok = SimpleNamespace(value=0)
    runtime = SimpleNamespace(
        cudaHostRegister=lambda *args: ok, cudaHostUnregister=lambda *args: ok
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(
        engram_ops, "get_accelerator_view_from_cpu_tensor", lambda tensor: tensor
    )


def _loading_worker(index: int, size: int, ports, world_ports, shm_dir) -> None:
    # Never undone: the mappings' finalizers run as the process exits.
    monkeypatch = pytest.MonkeyPatch()
    _patch_cuda_registration(monkeypatch)
    monkeypatch.setattr(engram_ops, "SHM_PATH", shm_dir)
    try:
        _init_rank_world(world_ports[index])
        with set_current_vllm_config(_rank_config(index, size, ports, 60)):
            group = get_engram_table_group()
        # CPU-only: no torch process group, so no NCCL communicator either.
        assert isinstance(group, NodeLocalLoadGroup)
        assert not hasattr(group, "device_group")
        assert (group.rank_in_group, group.world_size) == (index, size)
        # Formed under a short timeout, it then waits as long as a load takes.
        assert group._group.store.timeout == timedelta(seconds=60)
        # Formed once, then reused by every table.
        with set_current_vllm_config(_rank_config(index, size, ports, 60)):
            assert get_engram_table_group() is group
        assert torch.distributed.get_world_size() == 1

        # The leader writes the table; after the fence every rank reads the
        # leader's rows through its own mapping of the same file.
        num_bytes = 4096
        table, view = engram_ops.DPSharedEngramStorage.share_shm_table(num_bytes, group)
        assert view is table
        if index == 0:
            table.copy_(torch.arange(num_bytes, dtype=torch.int64).to(torch.uint8))
        group.barrier()
        expected = torch.arange(num_bytes, dtype=torch.int64).to(torch.uint8)
        assert torch.equal(table, expected)
        group.barrier()

        # The leader cannot create the file: every rank is told, so none waits.
        monkeypatch.setattr(engram_ops, "_shm_shortfall", lambda n: "too small")
        with pytest.raises(RuntimeError, match="too small"):
            engram_ops.DPSharedEngramStorage.share_shm_table(num_bytes, group)
        group.barrier()

        # Released after loading: nothing more may run on it.
        release_engram_load_group()
        assert group.released
        with pytest.raises(RuntimeError, match="released after loading"):
            group.barrier()
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("size", [2, 3])
def test_loading_group_shares_one_table_across_independent_ranks(size):
    ports = get_open_ports_list(1)
    world_ports = get_open_ports_list(size)
    with tempfile.TemporaryDirectory() as shm_dir:
        context = mp.get_context("spawn")
        procs = [
            context.Process(
                target=_loading_worker, args=(i, size, ports, world_ports, shm_dir)
            )
            for i in range(size)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=120)
        assert [proc.exitcode for proc in procs] == [0] * size


def _lonely_worker(world_port: int, ports, result) -> None:
    # The leader is not coming: give up on it quickly rather than in 300 s.
    parallel_state.ENGRAM_LOAD_GROUP_FORM_TIMEOUT_S = 2
    try:
        _init_rank_world(world_port)
        with set_current_vllm_config(_rank_config(1, 2, ports, 2)):
            result.put(get_engram_table_group() is None)
    finally:
        cleanup_dist_env_and_memory()


def test_unreachable_leader_falls_back_to_a_table_per_rank():
    """A rank whose co-located leader never comes keeps its own table."""
    context = mp.get_context("spawn")
    result = context.Queue()
    proc = context.Process(
        target=_lonely_worker,
        args=(get_open_port(), get_open_ports_list(1), result),
    )
    proc.start()
    proc.join(timeout=60)
    assert proc.exitcode == 0
    assert result.get(timeout=1) is True


def test_shared_mapping_is_one_file(tmp_path):
    """Sanity check of the mechanism the group relies on: two MAP_SHARED
    mappings of one file see each other's writes."""
    path = tmp_path / "table"
    path.write_bytes(b"\0" * 64)
    with open(path, "r+b") as a, open(path, "r+b") as b:
        first = mmap.mmap(a.fileno(), 64, flags=mmap.MAP_SHARED)
        second = mmap.mmap(b.fileno(), 64, flags=mmap.MAP_SHARED)
        np.frombuffer(first, dtype=np.uint8)[:] = 7
        assert np.frombuffer(second, dtype=np.uint8).tolist() == [7] * 64
        first.close()
        second.close()


def test_cli_flag_reaches_the_parallel_config():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        ["--data-parallel-size", "2", "--data-parallel-independent-moe"]
    )
    assert EngineArgs.from_cli_args(args).data_parallel_independent_moe is True
    args = parser.parse_args(["--data-parallel-size", "2"])
    assert EngineArgs.from_cli_args(args).data_parallel_independent_moe is False
