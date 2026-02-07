"""
多种子稳态基准测试脚本

用途：
- 以固定随机种子、固定参数、固定流程评估多种子吞吐
- 用于每次优化前后做可比对的性能回归
"""

import random
import statistics
import time
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F

# 基准脚本属于测试场景：启用测试模式（仅此类场景限制显存占用）
os.environ.setdefault("SCC_TEST_MODE", "1")

from src.main import (
    clear_count_only_stats,
    clear_results,
    detect_slime_chunk,
    get_count_only_summary,
    process_seed,
    set_count_only_mode,
    set_verbose_output,
    warmup_cudagraphs,
)
from src.config import BLOCK_SIZE, PATTERN, SPAWN_RADIUS, device


# ===== 基准参数（与历史对比一致） =====
RNG_SEED = 20260206
# 约 1 分钟配置（按当前约 460~480 seeds/s 估算）
BENCH_SEEDS = 5000
BURN_IN_SEEDS = 2000
RUNS = 5

# ===== 测试配置 =====
# - TEST_BLOCK_SIZE: 使用配置中按显卡参数公式计算的最优分块大小
# - SEED_PACING_MS: 每个 seed 后短暂让出调度，进一步平滑 GPU 争用（默认关闭）
TEST_BLOCK_SIZE = BLOCK_SIZE
SEED_PACING_MS = 0

THRESHOLD = 170
CHUNK_RADIUS = 507

# ===== 稳定性参考案例（已知正确） =====
REFERENCE_SEED = 7981483398353467015
REFERENCE_SLIME_CHUNKS = 50
REFERENCE_AFK_X = -68
REFERENCE_AFK_Z = 437


def _validate_reference_seed_result(chunk_radius: int, use_compiled: bool = True) -> None:
    """校验参考种子在指定位置的已知结果，确保结果稳定。"""
    seed_tensor = torch.tensor(REFERENCE_SEED, dtype=torch.int64, device=device)
    found_match = False

    for x_start, z_start, chunk_tensor in detect_slime_chunk(
        seed_tensor,
        chunk_radius,
        use_compiled=use_compiled,
    ):
        chunk_float = chunk_tensor[None, None].float()
        conv_result = F.conv2d(chunk_float, PATTERN.float())

        pattern_h, pattern_w = PATTERN.shape[-2], PATTERN.shape[-1]
        valid_h = conv_result.shape[-2] - (pattern_h - 1)
        valid_w = conv_result.shape[-1] - (pattern_w - 1)

        if valid_h <= 0 or valid_w <= 0:
            continue

        local_h = REFERENCE_AFK_Z - SPAWN_RADIUS - z_start
        local_w = REFERENCE_AFK_X - SPAWN_RADIUS - x_start
        if 0 <= local_h < valid_h and 0 <= local_w < valid_w:
            value = int(conv_result[0, 0, local_h, local_w].item())
            if value == REFERENCE_SLIME_CHUNKS:
                found_match = True
                break

    if not found_match:
        raise RuntimeError(
            "参考案例校验失败: "
            f"seed={REFERENCE_SEED}, expected_count={REFERENCE_SLIME_CHUNKS}, "
            f"expected_pos=({REFERENCE_AFK_X}, {REFERENCE_AFK_Z})"
        )

    print(
        "reference_case=ok "
        f"seed={REFERENCE_SEED}, count={REFERENCE_SLIME_CHUNKS}, "
        f"pos=({REFERENCE_AFK_X}, {REFERENCE_AFK_Z})"
    )


def _prepare_seeds(n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    seeds = [rng.randint(-(2**63), 2**63 - 1) for _ in range(n)]
    # 固定把首个样本替换为参考种子，确保每次基准都覆盖稳定性案例
    if n > 0:
        seeds[0] = REFERENCE_SEED
    return seeds


def benchmark_multi_seed(
    seeds: List[int],
    threshold: int,
    chunk_radius: int,
    runs: int,
    burn_in: int,
    block_size: int = TEST_BLOCK_SIZE,
    seed_pacing_ms: int = SEED_PACING_MS,
    use_compiled: bool = True,
    count_only: bool = False,
) -> Tuple[Tuple[float, float], Tuple[float, float], List[float]]:
    """
    执行多种子稳态基准。

    Returns:
        (time_range_seconds, throughput_range_seeds_per_sec, raw_times)
    """
    set_verbose_output(False)
    set_count_only_mode(count_only)

    pacing_seconds = max(seed_pacing_ms, 0) / 1000.0

    # 预热编译与 CUDA 图
    warmup_cudagraphs(chunk_radius=chunk_radius, full_pipeline=False)

    # 先做一次稳定性校验，保证基准结果可信
    _validate_reference_seed_result(chunk_radius=chunk_radius, use_compiled=use_compiled)

    # burn-in（排除冷启动残留）
    clear_results()
    if count_only:
        clear_count_only_stats()

    for s in seeds[:burn_in]:
        process_seed(
            s,
            threshold,
            chunk_radius,
            block_size=block_size,
            use_compiled=use_compiled,
        )
        if not count_only:
            clear_results()
        if pacing_seconds > 0:
            time.sleep(pacing_seconds)

    times: List[float] = []
    throughputs: List[float] = []
    for i in range(runs):
        if not count_only:
            clear_results()
        t0 = time.perf_counter()

        for s in seeds:
            process_seed(
                s,
                threshold,
                chunk_radius,
                block_size=block_size,
                use_compiled=use_compiled,
            )
            if not count_only:
                clear_results()
            if pacing_seconds > 0:
                time.sleep(pacing_seconds)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0
        rate = len(seeds) / dt
        times.append(dt)
        throughputs.append(rate)
        print(f"run{i+1}={dt:.4f}s, seeds_per_sec={rate:.2f}")

    min_s, max_s = min(times), max(times)
    min_rate, max_rate = min(throughputs), max(throughputs)

    print("time_samples=[" + ", ".join(f"{t:.4f}" for t in times) + "]")
    print(f"time_range={min_s:.4f}s~{max_s:.4f}s")
    print(f"throughput_range={min_rate:.2f}~{max_rate:.2f} seeds/s")

    # 仅作参考，不作为主比较指标
    print(f"time_median(ref)={statistics.median(times):.4f}s")

    if count_only:
        print(f"count_only_summary={get_count_only_summary()}")

    return (min_s, max_s), (min_rate, max_rate), times


if __name__ == "__main__":
    print("=" * 56)
    print("SlimeChunkCounter 多种子稳态基准")
    print("=" * 56)
    print(
        f"params: threshold={THRESHOLD}, chunk_radius={CHUNK_RADIUS}, "
        f"bench_seeds={BENCH_SEEDS}, burn_in={BURN_IN_SEEDS}, runs={RUNS}, rng_seed={RNG_SEED}, "
        f"test_block_size={TEST_BLOCK_SIZE}, seed_pacing_ms={SEED_PACING_MS}"
    )
    print(
        f"reference: seed={REFERENCE_SEED}, count={REFERENCE_SLIME_CHUNKS}, "
        f"pos=({REFERENCE_AFK_X}, {REFERENCE_AFK_Z})"
    )

    seeds = _prepare_seeds(BENCH_SEEDS, RNG_SEED)

    print("\n[normal mode]")
    benchmark_multi_seed(
        seeds=seeds,
        threshold=THRESHOLD,
        chunk_radius=CHUNK_RADIUS,
        runs=RUNS,
        burn_in=BURN_IN_SEEDS,
        block_size=TEST_BLOCK_SIZE,
        seed_pacing_ms=SEED_PACING_MS,
        use_compiled=True,
        count_only=False,
    )
