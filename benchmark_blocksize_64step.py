import os
import csv
import time
import random
import statistics
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from src.main import (
    process_seed,
    set_verbose_output,
    set_count_only_mode,
    clear_count_only_stats,
    clear_results,
)
from src.config import BLOCK_SIZE as AUTO_BLOCK_SIZE, SPAWN_RADIUS


if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，无法做显卡 block_size 基准测试")

# ===== 基准参数 =====
THRESHOLD = 170
CHUNK_RADIUS = 500 + SPAWN_RADIUS
SEEDS_PER_REPEAT = 140
REPEATS = 2
WARMUP_SEEDS = 16
RNG_SEED = 20260207

# 按要求：从 128 到 8192，每 128 一档
CANDIDATES = list(range(128, 8192 + 1, 128))

props = torch.cuda.get_device_properties(0)
gpu_name = props.name

rng = random.Random(RNG_SEED)
total_needed = WARMUP_SEEDS + SEEDS_PER_REPEAT * REPEATS
seeds = [rng.randint(-(2**63), 2**63 - 1) for _ in range(total_needed)]

set_verbose_output(False)
set_count_only_mode(True)

results = []

print(f"GPU={gpu_name}")
print(f"AUTO_BLOCK_SIZE={AUTO_BLOCK_SIZE}")
print(f"chunk_radius={CHUNK_RADIUS}, threshold={THRESHOLD}, repeats={REPEATS}, seeds_per_repeat={SEEDS_PER_REPEAT}")
print(f"candidates={len(CANDIDATES)} -> {CANDIDATES[0]}..{CANDIDATES[-1]} step=64")

for idx, block_size in enumerate(CANDIDATES, 1):
    try:
        clear_results()
        clear_count_only_stats()

        # warmup
        for s in seeds[:WARMUP_SEEDS]:
            process_seed(s, THRESHOLD, CHUNK_RADIUS, block_size=block_size, use_compiled=True)

        repeat_rates = []

        for r in range(REPEATS):
            start = WARMUP_SEEDS + r * SEEDS_PER_REPEAT
            end = start + SEEDS_PER_REPEAT
            batch = seeds[start:end]

            clear_results()
            clear_count_only_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for s in batch:
                process_seed(s, THRESHOLD, CHUNK_RADIUS, block_size=block_size, use_compiled=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            repeat_rates.append(len(batch) / dt)

        med_rate = statistics.median(repeat_rates)
        avg_rate = statistics.mean(repeat_rates)
        std_rate = statistics.pstdev(repeat_rates) if len(repeat_rates) > 1 else 0.0

        results.append(
            {
                "block_size": block_size,
                "median_seeds_per_sec": med_rate,
                "mean_seeds_per_sec": avg_rate,
                "std_seeds_per_sec": std_rate,
                "repeat1": repeat_rates[0],
                "repeat2": repeat_rates[1] if len(repeat_rates) > 1 else repeat_rates[0],
                "status": "ok",
            }
        )
        print(f"[{idx:02d}/{len(CANDIDATES)}] block={block_size:4d} | median={med_rate:7.2f} seeds/s | mean={avg_rate:7.2f} | std={std_rate:6.2f}")

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        results.append(
            {
                "block_size": block_size,
                "median_seeds_per_sec": 0.0,
                "mean_seeds_per_sec": 0.0,
                "std_seeds_per_sec": 0.0,
                "repeat1": 0.0,
                "repeat2": 0.0,
                "status": "oom",
            }
        )
        print(f"[{idx:02d}/{len(CANDIDATES)}] block={block_size:4d} | OOM")

ok_rows = [r for r in results if r["status"] == "ok"]
if not ok_rows:
    raise RuntimeError("没有成功的 block_size")

best = max(ok_rows, key=lambda x: x["median_seeds_per_sec"])

os.makedirs("results", exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
csv_path = os.path.join("results", f"blocksize_benchmark_64step_{ts}.csv")
png_path = os.path.join("results", f"blocksize_benchmark_64step_{ts}.png")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "block_size",
            "median_seeds_per_sec",
            "mean_seeds_per_sec",
            "std_seeds_per_sec",
            "repeat1",
            "repeat2",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

ok_rows_sorted = sorted(ok_rows, key=lambda x: x["block_size"])
x = [r["block_size"] for r in ok_rows_sorted]
y = [r["median_seeds_per_sec"] for r in ok_rows_sorted]

plt.figure(figsize=(11, 6), dpi=150)
plt.plot(x, y, marker="o", linewidth=1.6, markersize=3.5, label="Median throughput")
plt.axvline(AUTO_BLOCK_SIZE, linestyle="--", linewidth=1.2, label=f"Auto block_size={AUTO_BLOCK_SIZE}")
plt.scatter([best["block_size"]], [best["median_seeds_per_sec"]], s=70, zorder=3, label=f"Best={best['block_size']}")
plt.title(f"Block Size Benchmark (64-step) on {gpu_name}")
plt.xlabel("block_size")
plt.ylabel("seeds / sec (median)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(png_path)
plt.close()

print("\n===== SUMMARY =====")
print(f"best_block_size={best['block_size']}")
print(f"best_median_seeds_per_sec={best['median_seeds_per_sec']:.4f}")
print(f"auto_block_size={AUTO_BLOCK_SIZE}")
print(f"csv={csv_path}")
print(f"png={png_path}")

ranked = sorted(ok_rows, key=lambda x: x["median_seeds_per_sec"], reverse=True)
print("top10=")
for i, row in enumerate(ranked[:10], 1):
    print(f"  {i}. block={row['block_size']:4d}, median={row['median_seeds_per_sec']:.2f}, mean={row['mean_seeds_per_sec']:.2f}, std={row['std_seeds_per_sec']:.2f}")
