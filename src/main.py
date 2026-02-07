from datetime import datetime
import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import argparse
import csv
import threading
from typing import Generator, Tuple, Union, Optional, List, Dict
import time
import random

import glob


from src.config import (
    LOG_LEVEL,
    DEFAULT_MODE,
    DEFAULT_RADIUS,
    DEFAULT_THRESHOLD,
    SPAWN_RADIUS,
    MAX_SLIME_CHUNKS,
    MAX_RADIUS,
    MIN_SEED,
    MAX_SEED,
    device,
    BLOCK_SIZE,
    PATTERN,
    PATTERN_FP16,
    USE_FP16,
    v1,
    v2,
    v3,
    v4,
    scrambler,
    multiplier,
    addend,
    mask,
)


MAX_LOG_FILES = 10  # 最大保留日志文件数
RESULTS_DIR = "results"  # 结果输出目录
CHECKPOINT_FILE = "checkpoint.txt"  # 断点文件
CHECKPOINT_FLUSH_INTERVAL = 100  # 检查点批量写入间隔

# 线程安全的结果收集器
_results_lock = threading.Lock()
_results: List[Dict] = []
_processed_seeds: set = set()
_pending_checkpoints: List[int] = []
_verbose_output: bool = True
_count_only_mode: bool = False

# 计数模式统计（不保存明细结果）
_count_only_stats_lock = threading.Lock()
_count_only_total_seeds: int = 0
_count_only_hit_seeds: int = 0
_count_only_best: Dict[int, int] = {}

# 单块坐标缓存（chunk_radius 固定时可复用，减少每个 seed 的 arange 分配开销）
_single_block_coord_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _device_cache_key(dev: torch.device) -> str:
    """生成设备缓存键。"""
    return f"{dev.type}:{dev.index}"


def _get_single_block_coords(
    chunk_radius: int,
    device: torch.device,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """获取单块快速路径所需的坐标张量（带缓存）。"""
    key = (chunk_radius, _device_cache_key(device))
    cached = _single_block_coord_cache.get(key)
    if cached is not None:
        x_coords, z_coords = cached
        return -chunk_radius, -chunk_radius, x_coords, z_coords

    x_start = -chunk_radius
    z_start = -chunk_radius
    x_end = chunk_radius + 1
    z_end = chunk_radius + 1

    x_block = torch.arange(x_start, x_end, dtype=torch.int32, device=device)
    z_block = torch.arange(z_start, z_end, dtype=torch.int32, device=device)
    x_coords = x_block.unsqueeze(0)
    z_coords = z_block.unsqueeze(1)

    _single_block_coord_cache[key] = (x_coords, z_coords)
    return x_start, z_start, x_coords, z_coords


def load_checkpoint() -> set:
    """
    加载已处理的种子检查点

    Returns:
        set: 已处理的种子集合
    """
    checkpoint_path = os.path.join(RESULTS_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return set(int(line.strip()) for line in f if line.strip())
        except Exception:
            pass
    return set()


def save_checkpoint(seed: int) -> None:
    """
    保存已处理的种子到检查点文件

    Args:
        seed: 已处理的种子
    """
    with _results_lock:
        _processed_seeds.add(seed)
        _pending_checkpoints.append(seed)


def flush_checkpoints(force: bool = False) -> None:
    """
    批量写入检查点，减少频繁 I/O

    Args:
        force: 是否强制写入（忽略批量阈值）
    """
    with _results_lock:
        if not _pending_checkpoints:
            return
        if not force and len(_pending_checkpoints) < CHECKPOINT_FLUSH_INTERVAL:
            return

        to_write = _pending_checkpoints.copy()
        _pending_checkpoints.clear()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(RESULTS_DIR, CHECKPOINT_FILE)
    with open(checkpoint_path, "a") as f:
        f.writelines(f"{seed}\n" for seed in to_write)


def set_verbose_output(enabled: bool) -> None:
    """设置是否输出详细命中日志"""
    global _verbose_output
    _verbose_output = enabled


def is_verbose_output() -> bool:
    """获取是否输出详细命中日志"""
    return _verbose_output


def set_count_only_mode(enabled: bool) -> None:
    """设置是否启用仅计数模式（不保存每个命中明细）"""
    global _count_only_mode
    _count_only_mode = enabled


def is_count_only_mode() -> bool:
    """获取是否启用仅计数模式"""
    return _count_only_mode


def clear_count_only_stats() -> None:
    """清空仅计数模式统计数据"""
    global _count_only_total_seeds, _count_only_hit_seeds
    with _count_only_stats_lock:
        _count_only_total_seeds = 0
        _count_only_hit_seeds = 0
        _count_only_best.clear()


def update_count_only_seed_best(seed: int, count: int) -> None:
    """更新某个种子的最佳命中计数"""
    with _count_only_stats_lock:
        prev = _count_only_best.get(seed)
        if prev is None or count > prev:
            _count_only_best[seed] = count


def finalize_count_only_seed(seed: int) -> None:
    """在种子处理结束后汇总统计"""
    global _count_only_total_seeds, _count_only_hit_seeds
    with _count_only_stats_lock:
        _count_only_total_seeds += 1
        if seed in _count_only_best:
            _count_only_hit_seeds += 1


def get_count_only_summary() -> Dict[str, Union[int, float]]:
    """获取仅计数模式统计摘要"""
    with _count_only_stats_lock:
        total = _count_only_total_seeds
        hit = _count_only_hit_seeds
        hit_rate = (hit / total) if total > 0 else 0.0
        best = max(_count_only_best.values()) if _count_only_best else 0
        return {
            "processed_seeds": total,
            "hit_seeds": hit,
            "hit_rate": hit_rate,
            "best_count": best,
        }


def _is_single_seed_mode(args: argparse.Namespace) -> bool:
    """根据命令行参数判断是否单种子模式"""
    if args.interactive:
        return False
    return (args.seed is not None) and (not args.multiple)


def should_warmup(args: argparse.Namespace) -> bool:
    """
    自动预热策略：
    - 明确 --no-warmup: 不预热
    - 单种子模式默认不预热（优先首结果响应）
    - 多种子模式默认预热（优先稳态吞吐）
    """
    if args.no_warmup:
        return False
    return not _is_single_seed_mode(args)


def clear_checkpoint() -> None:
    """清除检查点文件"""
    checkpoint_path = os.path.join(RESULTS_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    with _results_lock:
        _processed_seeds.clear()
        _pending_checkpoints.clear()


def add_result(seed: int, count: int, x: int, z: int) -> None:
    """
    线程安全地添加结果（异步写入）

    Args:
        seed: 世界种子
        count: 史莱姆区块数
        x: 挂机点 X 坐标
        z: 挂机点 Z 坐标
    """
    result = {
        "seed": seed,
        "slime_chunks": count,
        "afk_x": x,
        "afk_z": z,
        "timestamp": datetime.now().isoformat()
    }
    with _results_lock:
        _results.append(result)


def add_results_batch(results: List[Dict]) -> None:
    """
    线程安全地批量添加结果

    Args:
        results: 结果字典列表
    """
    if not results:
        return
    with _results_lock:
        _results.extend(results)


def save_results_to_csv(filename: Optional[str] = None) -> str:
    """
    保存结果到 CSV 文件

    Args:
        filename: 文件名，如果为 None 则自动生成

    Returns:
        str: 保存的文件路径
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if filename is None:
        filename = f"results_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"

    filepath = os.path.join(RESULTS_DIR, filename)

    with _results_lock:
        if not _results:
            return ""

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "slime_chunks", "afk_x", "afk_z", "timestamp"])
            writer.writeheader()
            writer.writerows(_results)

    return filepath


def clear_results() -> None:
    """清空结果列表"""
    with _results_lock:
        _results.clear()


def get_top_results(n: int = 10, deduplicate: bool = True) -> List[Dict]:
    """
    获取排名前 N 的结果（按史莱姆区块数排序）

    Args:
        n: 返回结果数量
        deduplicate: 是否去重（同一位置只保留最高分）

    Returns:
        List[Dict]: 排序后的结果列表
    """
    with _results_lock:
        if not _results:
            return []

        if deduplicate:
            # 按 (seed, afk_x, afk_z) 去重，保留最高分
            unique_results = {}
            for r in _results:
                key = (r['seed'], r['afk_x'], r['afk_z'])
                if key not in unique_results or r['slime_chunks'] > unique_results[key]['slime_chunks']:
                    unique_results[key] = r
            sorted_results = sorted(unique_results.values(), key=lambda x: x['slime_chunks'], reverse=True)
        else:
            sorted_results = sorted(_results, key=lambda x: x['slime_chunks'], reverse=True)

        return sorted_results[:n]


def get_results_summary() -> Dict:
    """
    获取结果统计摘要

    Returns:
        Dict: 包含统计信息的字典
    """
    with _results_lock:
        if not _results:
            return {'count': 0, 'max': 0, 'min': 0, 'avg': 0}

        counts = [r['slime_chunks'] for r in _results]
        return {
            'count': len(_results),
            'max': max(counts),
            'min': min(counts),
            'avg': sum(counts) / len(counts),
            'unique_seeds': len(set(r['seed'] for r in _results)),
        }


def cleanup_old_logs(log_dir: str = "log", max_files: int = MAX_LOG_FILES) -> None:
    """
    清理旧的日志文件，只保留最新的 max_files 个

    Args:
        log_dir: 日志目录
        max_files: 最大保留文件数
    """
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if len(log_files) > max_files:
        # 按修改时间排序，删除最旧的
        log_files.sort(key=os.path.getmtime)
        for old_file in log_files[:-max_files]:
            try:
                os.remove(old_file)
            except OSError:
                pass  # 忽略删除失败


def init_logging() -> None:
    """
    初始化日志设置及目录，并清理旧日志
    """
    os.makedirs("log", exist_ok=True)
    cleanup_old_logs()
    logging.basicConfig(
        filename=f"log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s:\t\t%(message)s",
        encoding="UTF-8",
    )


def log_and_print(message: str) -> None:
    print(message)
    logging.info(message)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Minecraft 史莱姆区块计数器 - 寻找最佳史莱姆农场位置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                          # 交互式模式
  python run.py -s 12345                 # 指定种子
  python run.py -s 12345 -r 2048 -t 55   # 指定所有参数
  python run.py -m                       # 多种子模式（随机搜索）
        """,
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="世界种子"
    )
    parser.add_argument(
        "-r", "--radius",
        type=int,
        default=None,
        help=f"区块检测半径 (默认: {DEFAULT_RADIUS}, 最大: {MAX_RADIUS})"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=None,
        help=f"计数阈值 (默认: {DEFAULT_THRESHOLD}, 最大: {MAX_SLIME_CHUNKS})"
    )
    parser.add_argument(
        "-m", "--multiple",
        action="store_true",
        help="多种子模式，随机搜索种子"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="强制使用交互式模式"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续（多种子模式）"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="清除检查点，重新开始"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过启动预热（首次运行可能更慢）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式：不打印每个命中，只输出摘要"
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="高吞吐模式：仅统计每个种子是否命中及最佳计数，不保存明细结果"
    )
    return parser.parse_args()


def get_user_inputs(args: Optional[argparse.Namespace] = None) -> Tuple[Union[str, int], int, int]:
    """
    获取用户输入的运行模式, 检测半径和计数阈值
    支持命令行参数和交互式输入

    Args:
        args: 命令行参数，如果为 None 则使用交互式输入

    Returns:
        tuple: 模式, 检测半径, 计数阈值

    Raises:
        ValueError: 当输入值无效时
    """
    # 如果提供了命令行参数且不是交互式模式
    if args and not args.interactive:
        # 确定模式
        if args.multiple:
            mode = DEFAULT_MODE
        elif args.seed is not None:
            if not (MIN_SEED <= args.seed <= MAX_SEED):
                raise ValueError(f"种子值必须在 -2^63 到 2^63-1 之间，当前值: {args.seed}")
            mode = args.seed
        else:
            # 没有指定种子也没有指定多种子模式，使用交互式
            return get_user_inputs_interactive()

        # 获取半径
        if args.radius is not None:
            if args.radius <= 0:
                raise ValueError(f"检测半径必须为正整数，当前值: {args.radius}")
            if args.radius > MAX_RADIUS:
                raise ValueError(f"检测半径过大，最大支持 {MAX_RADIUS}，当前值: {args.radius}")
            radius = args.radius
        else:
            radius = DEFAULT_RADIUS

        # 获取阈值
        if args.threshold is not None:
            if args.threshold <= 0:
                raise ValueError(f"计数阈值必须为正整数，当前值: {args.threshold}")
            if args.threshold > MAX_SLIME_CHUNKS:
                raise ValueError(f"计数阈值过大，最大有效值为 {MAX_SLIME_CHUNKS}，当前值: {args.threshold}")
            threshold = args.threshold
        else:
            threshold = DEFAULT_THRESHOLD

        return mode, radius, threshold

    # 交互式模式
    return get_user_inputs_interactive()


def get_user_inputs_interactive() -> Tuple[Union[str, int], int, int]:
    """
    交互式获取用户输入

    Returns:
        tuple: 模式, 检测半径, 计数阈值

    Raises:
        ValueError: 当输入值无效时
    """
    mode_input = (
        input(
            f"运行模式, 计算所有种子(multiple seeds)或单个种子(single seed) ([{DEFAULT_MODE}]ultiple seeds/种子值):"
        )
        .strip()
        .upper()
    )

    if not mode_input or mode_input.startswith(DEFAULT_MODE):
        mode = DEFAULT_MODE
    else:
        try:
            mode = int(mode_input)
            if not (MIN_SEED <= mode <= MAX_SEED):
                raise ValueError(f"种子值必须在 -2^63 到 2^63-1 之间，当前值: {mode}")
        except ValueError as e:
            if "种子值必须" in str(e):
                raise
            raise ValueError(f"无效的种子值: {mode_input}，请输入整数或 'M'")

    radius_input = input(f"区块检测半径 [{DEFAULT_RADIUS}]:")
    if radius_input:
        try:
            radius = int(radius_input)
            if radius <= 0:
                raise ValueError(f"检测半径必须为正整数，当前值: {radius}")
            if radius > MAX_RADIUS:
                raise ValueError(f"检测半径过大，最大支持 {MAX_RADIUS}，当前值: {radius}")
        except ValueError as e:
            if "检测半径" in str(e):
                raise
            raise ValueError(f"无效的检测半径: {radius_input}，请输入正整数")
    else:
        radius = DEFAULT_RADIUS

    threshold_input = input(f"计数阈值 [{DEFAULT_THRESHOLD}]:")
    if threshold_input:
        try:
            threshold = int(threshold_input)
            if threshold <= 0:
                raise ValueError(f"计数阈值必须为正整数，当前值: {threshold}")
            if threshold > MAX_SLIME_CHUNKS:
                raise ValueError(f"计数阈值过大，最大有效值为 {MAX_SLIME_CHUNKS}，当前值: {threshold}")
        except ValueError as e:
            if "计数阈值" in str(e):
                raise
            raise ValueError(f"无效的计数阈值: {threshold_input}，请输入正整数")
    else:
        threshold = DEFAULT_THRESHOLD

    return mode, radius, threshold


def generate_seeds(mode: Union[str, int]) -> Generator[torch.Tensor, None, None]:
    """
    生成由模式控制的种子, 如果是multiple seeds模式(str)则随机生成, 否则使用模式指定的种子(种子值)

    Args:
        mode: 模式

    Yields:
        torch.Tensor: 种子值
    """
    if mode == DEFAULT_MODE:
        while True:
            # 使用 Python RNG 生成，再转 Tensor，避免高频 torch.randint 调用开销
            yield torch.tensor(
                random.randint(-(2**63), 2**63 - 1),
                dtype=torch.int64,
                device=device,
            )
    else:
        yield torch.tensor(mode, dtype=torch.int64, device=device)


def generate_seed_values(mode: Union[str, int]) -> Generator[int, None, None]:
    """
    生成 Python int 类型的种子值（用于多种子高频调度，减少张量构造开销）

    Args:
        mode: 模式

    Yields:
        int: 种子值
    """
    if mode == DEFAULT_MODE:
        while True:
            yield random.randint(-(2**63), 2**63 - 1)
    else:
        yield int(mode)


def get_random_seed(
    worldSeed: torch.Tensor, chunkX: torch.Tensor, chunkZ: torch.Tensor
) -> torch.Tensor:
    """
    通过世界种子和区块坐标计算随机数生成种子

    Args:
        worldSeed: 世界种子
        chunkX: 区块X坐标
        chunkZ: 区块Z坐标

    Returns:
        torch.Tensor: 随机数种子
    """
    return (
        worldSeed
        + (chunkX * chunkX * v1).to(dtype=torch.int64)
        + (chunkX * v2).to(dtype=torch.int64)
        + (chunkZ * chunkZ).to(dtype=torch.int64) * v3
        + (chunkZ * v4).to(dtype=torch.int64)
        ^ scrambler
    )


def next_int(seed: torch.Tensor) -> torch.Tensor:
    """
    模拟 Java Random.nextInt(10) 的行为（优化版本）

    重试概率约 2 亿分之一 (10 / 2^31)，但为保证正确性必须处理。
    使用纯张量操作，避免 Python 条件分支，支持 torch.compile 优化。

    Args:
        seed: 随机数种子张量

    Returns:
        torch.Tensor: 0-9 之间的随机整数
    """
    seed = (seed ^ multiplier) & mask

    # 第一次迭代（99.9999995% 的情况下有效）
    s1 = (seed * multiplier + addend) & mask
    u1 = (s1 >> 17).to(dtype=torch.int32)
    # 处理有符号整数：如果最高位为 1，则为负数
    u1 = torch.where((u1 & (1 << 31)).bool(), u1 - (1 << 32), u1)
    r1 = u1 % 10
    valid1 = (u1 - r1 + 9) >= 0

    # 第二次迭代（处理极少数无效情况）
    # 始终计算，但只在需要时使用结果（避免条件分支）
    s2 = (s1 * multiplier + addend) & mask
    u2 = (s2 >> 17).to(dtype=torch.int32)
    u2 = torch.where((u2 & (1 << 31)).bool(), u2 - (1 << 32), u2)
    r2 = u2 % 10

    # 使用 torch.where 合并结果（无条件分支）
    return torch.where(valid1, r1, r2)


# 检查是否可以使用 torch.compile
def _can_use_torch_compile() -> bool:
    """检查是否可以使用 torch.compile with inductor"""
    if not torch.cuda.is_available():
        return False
    try:
        import triton
        # Triton 3.x+ 使用新的 API
        return hasattr(triton, 'compiler') and hasattr(triton.compiler, 'compile')
    except ImportError:
        return False


def _can_use_cudagraphs() -> bool:
    """检查是否可以使用 CUDA Graphs"""
    return torch.cuda.is_available()


_USE_TORCH_COMPILE = _can_use_torch_compile()
_USE_CUDAGRAPHS = _can_use_cudagraphs() and not _USE_TORCH_COMPILE


def _compute_slime_chunks_batch_impl(
    seed: torch.Tensor,
    x_coords: torch.Tensor,
    z_coords: torch.Tensor,
) -> torch.Tensor:
    """
    批量计算史莱姆区块

    Args:
        seed: 世界种子
        x_coords: X 坐标张量
        z_coords: Z 坐标张量

    Returns:
        torch.Tensor: 是否为史莱姆区块的布尔张量
    """
    seeds = get_random_seed(seed, x_coords, z_coords)
    return next_int(seeds) == 0


# 使用 torch.compile 优化（如果可用）
# 注意：不使用 fullgraph=True，因为内部函数使用了 @torch.compiler.disable
if _USE_TORCH_COMPILE:
    _compute_slime_chunks_batch = torch.compile(
        _compute_slime_chunks_batch_impl,
        mode="reduce-overhead",
    )
elif _USE_CUDAGRAPHS:
    # Windows 上使用 cudagraphs 后端（不依赖 Triton）
    _compute_slime_chunks_batch = torch.compile(
        _compute_slime_chunks_batch_impl,
        backend="cudagraphs",
    )
else:
    _compute_slime_chunks_batch = _compute_slime_chunks_batch_impl


@torch.no_grad()
def detect_slime_chunk(
    seed: Union[int, torch.Tensor],
    chunk_radius: int,
    device: torch.device = device,
    block_size: int = BLOCK_SIZE,
    use_compiled: bool = True,
) -> Generator[Tuple[int, int, torch.Tensor], None, None]:
    """
    分块计算史莱姆区块，带重叠，避免 OOM 且保证卷积结果正确

    Args:
        seed: 世界种子
        chunk_radius: 检测半径
        device: 计算设备
        block_size: 每个分块的有效大小

    Yields:
        Tuple[int, int, torch.Tensor]: (x_offset, z_offset, chunk_tensor) 分块的史莱姆区块数据
    """
    # 确保 seed 是 Tensor
    if not isinstance(seed, torch.Tensor):
        seed = torch.tensor(seed, dtype=torch.int64, device=device)

    compute_fn = _compute_slime_chunks_batch if use_compiled else _compute_slime_chunks_batch_impl

    overlap = 15 - 1
    total_size = 2 * chunk_radius + 1

    # 单块快速路径：当总区域只需一个分块时，避免双层循环与边界计算开销
    if total_size <= block_size:
        x_start, z_start, x_coords, z_coords = _get_single_block_coords(chunk_radius, device)
        chunks = compute_fn(seed, x_coords, z_coords)
        yield x_start, z_start, chunks
        return

    # 使用 Python 计算坐标范围，避免 GPU 同步
    for i in range(0, total_size, block_size):
        for j in range(0, total_size, block_size):
            # 直接用 Python 计算起始坐标
            x_start = -chunk_radius + i
            z_start = -chunk_radius + j
            
            # 计算块的实际大小（包含重叠）
            x_end = min(x_start + block_size + overlap, chunk_radius + 1)
            z_end = min(z_start + block_size + overlap, chunk_radius + 1)
            
            # 在 GPU 上创建坐标张量
            x_block = torch.arange(x_start, x_end, dtype=torch.int32, device=device)
            z_block = torch.arange(z_start, z_end, dtype=torch.int32, device=device)

            # 使用广播坐标，避免 meshgrid + flatten + reshape 的额外开销
            # 结果形状为 [len(z_block), len(x_block)]，与原实现保持一致
            x_coords = x_block.unsqueeze(0)
            z_coords = z_block.unsqueeze(1)
            chunks = compute_fn(seed, x_coords, z_coords)

            yield x_start, z_start, chunks


def detect_and_log_matches(
    chunk_tensor: torch.Tensor,
    threshold: int,
    x_start: int,
    z_start: int,
    seed: torch.Tensor,
    verbose: bool = True,
) -> None:
    """
    对输入的 chunk_tensor 进行卷积匹配，若匹配值 >= threshold，则打印匹配位置和数值。

    Args:
        chunk_tensor: [H, W] 的布尔或整数张量，表示当前分块的史莱姆区块
        threshold: 匹配阈值
        x_start: 当前块在全局 X 方向的起始索引偏移
        z_start: 当前块在全局 Z 方向的起始索引偏移
        seed: 当前世界种子（用于打印）
        verbose: 是否输出详细日志
    """
    with torch.inference_mode():
        threshold_f = float(threshold)
        chunk_tensor = chunk_tensor[None, None]

        # 使用预计算的 FP16 pattern（如果可用）
        if USE_FP16:
            chunk_tensor = chunk_tensor.half()
            conv_result = F.conv2d(chunk_tensor, PATTERN_FP16).float()
        else:
            chunk_tensor = chunk_tensor.float()
            conv_result = F.conv2d(chunk_tensor, PATTERN)

        # 计算有效区域，避免边界问题
        pattern_h, pattern_w = PATTERN.shape[-2], PATTERN.shape[-1]
        valid_h = conv_result.shape[-2] - (pattern_h - 1)
        valid_w = conv_result.shape[-1] - (pattern_w - 1)

        # 如果有效区域为空，直接返回
        if valid_h <= 0 or valid_w <= 0:
            return

        valid_result = conv_result[:, :, :valid_h, :valid_w]

        # 先做块级最大值筛选，未达阈值时直接跳过后续 argwhere 与 GPU->CPU 传输
        max_value = float(valid_result.max().item())
        if max_value < threshold_f:
            return

        count_only = is_count_only_mode()
        if count_only:
            # 仅计数模式：最大值已知且 >= threshold，直接记录避免构造 match_mask 与回传
            seed_val = int(seed.item())
            update_count_only_seed_best(seed_val, int(max_value))
            return

        match_mask = valid_result >= threshold_f

        # 获取所有匹配位置和值（使用 argwhere 比 nonzero 快 30+ 倍）
        positions = torch.argwhere(match_mask[0, 0])  # [N, 2] tensor
        values = valid_result[0, 0][match_mask[0, 0]]

        # 一次性转移到 CPU
        positions_np = positions.cpu().numpy()
        values_np = values.cpu().numpy()
        seed_val = int(seed.item())
        spawn_radius = SPAWN_RADIUS
        ts_now = datetime.now().isoformat

        batched_results: List[Dict] = []
        log_messages: List[str] = []

        for (h, w), value in zip(positions_np, values_np):
            x = x_start + int(w) + spawn_radius
            z = z_start + int(h) + spawn_radius
            count = int(value)

            batched_results.append(
                {
                    "seed": seed_val,
                    "slime_chunks": count,
                    "afk_x": x,
                    "afk_z": z,
                    "timestamp": ts_now(),
                }
            )
            if verbose:
                log_messages.append(
                    f"史莱姆区块数: {count}, 种子: {seed_val}, 挂机点区块位置: ({x}, {z})"
                )

        # 结果批量入库，减少锁竞争
        add_results_batch(batched_results)

        # 日志批量输出
        if log_messages:
            for msg in log_messages:
                log_and_print(msg)


def process_seed(
    seed: Union[int, torch.Tensor],
    threshold: int,
    chunk_radius: int,
    block_size: int = BLOCK_SIZE,
    use_compiled: bool = True,
) -> None:
    """
    处理单个种子的史莱姆区块检测

    Args:
        seed: 世界种子
        threshold: 匹配阈值
        chunk_radius: 检测半径
        block_size: 分块大小（用于控制单次 CUDA 工作负载）
    """
    if not isinstance(seed, torch.Tensor):
        seed = torch.tensor(seed, dtype=torch.int64, device=device)
    else:
        seed = seed.to(device, dtype=torch.int64)

    # 📌 优化：缓存 verbose 标志，避免每个块都调用 is_verbose_output() 函数
    verbose = is_verbose_output()

    for x_start, z_start, chunk_tensor in detect_slime_chunk(
        seed, chunk_radius, block_size=block_size, use_compiled=use_compiled
    ):
        detect_and_log_matches(
            chunk_tensor,
            threshold,
            x_start,
            z_start,
            seed,
            verbose=verbose,
        )

    if is_count_only_mode():
        finalize_count_only_seed(int(seed.item()))


def warmup_cudagraphs(seed: int = 12345, chunk_radius: int = 10, full_pipeline: bool = False) -> None:
    """
    预热 CUDA Graphs，触发 torch.compile 编译缓存。
    
    这会执行一次完整的检测流程，确保所有 CUDA 内核都被编译并缓存,
    避免后续正式运行时的 20+ 秒编译开销。
    
    Args:
        seed: 用于预热的种子
        chunk_radius: 用于预热的半径
    """
    print("🔄 正在预热 CUDA Graphs...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    seed_tensor = torch.tensor(seed, dtype=torch.int64, device=device)

    # 执行一次流程触发编译：默认只预热计算路径，避免额外卷积开销
    block_count = 0
    for x_start, z_start, chunk_tensor in detect_slime_chunk(seed_tensor, chunk_radius):
        if full_pipeline:
            detect_and_log_matches(
                chunk_tensor,
                50,
                x_start,
                z_start,
                seed_tensor,
                verbose=False,
            )
        block_count += 1
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mode = "完整流水线" if full_pipeline else "计算路径"
    print(f"✅ 预热完成（{mode}）！({elapsed:.2f}s, {block_count} blocks)")
    if full_pipeline:
        clear_results()  # 清除预热产生的临时结果


def run(mode: Union[str, int], radius: int, threshold: int, resume: bool = False) -> None:
    chunk_radius = radius + SPAWN_RADIUS

    # 加载检查点（如果启用断点续传）
    processed_seeds = load_checkpoint() if resume else set()
    if resume and processed_seeds:
        log_and_print(f"📂 从检查点恢复，已跳过 {len(processed_seeds)} 个种子")

    # 单种子模式：同步执行
    if mode != DEFAULT_MODE:
        # 单种子模式通常只执行一次卷积，关闭 benchmark 避免首次算法搜索开销
        cudnn_benchmark_prev = torch.backends.cudnn.benchmark
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False

        verbose = is_verbose_output()

        for seed in generate_seeds(mode):
            try:
                # 预计算总块数
                total_chunks = (
                    (2 * chunk_radius + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
                ) ** 2

                # 单种子单块任务：跳过 torch.compile 首次编译开销（通常更快）
                use_compiled = total_chunks > 1

                # 创建 tqdm 进度条，显示完整进度
                with tqdm(
                    total=total_chunks,
                    desc=f"Processing seed {seed.item()}",
                    dynamic_ncols=True,
                    bar_format="{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} blocks | {rate_fmt} | ETA: {remaining}",
                    leave=True,
                ) as pbar:
                    for x_start, z_start, chunk_tensor in detect_slime_chunk(
                        seed, chunk_radius, use_compiled=use_compiled
                    ):
                        detect_and_log_matches(
                            chunk_tensor,
                            threshold,
                            x_start,
                            z_start,
                            seed,
                            verbose=verbose,
                        )
                        pbar.update(1)  # 手动更新进度
            except Exception:
                logging.exception("Error processing single seed")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = cudnn_benchmark_prev
        return

    # 多种子模式：GPU 串行处理（避免 torch.compile + 多线程导致崩溃/退化）
    # 说明：在当前 PyTorch + CUDA Graphs 环境下，线程并发会触发 TLS 断言错误，
    # 且吞吐显著下降，因此改为稳定的串行调度。
    log_and_print("⚙️ 多种子调度模式: GPU 串行")

    with tqdm(
        desc="Processing seeds",
        dynamic_ncols=True,
        bar_format="{desc} | {rate_fmt} | Total: {n_fmt}",
    ) as pbar:
        skipped_count = 0
        for seed_val in generate_seed_values(mode):
            # 跳过已处理的种子
            if seed_val in processed_seeds:
                skipped_count += 1
                if skipped_count % 1000 == 0:
                    pbar.set_postfix_str(f"skipped={skipped_count}")
                continue

            try:
                process_seed(
                    seed_val,
                    threshold,
                    chunk_radius,
                    use_compiled=True,
                )
                # 保存检查点
                save_checkpoint(seed_val)
                flush_checkpoints()
            except torch.cuda.OutOfMemoryError:
                logging.error(f"GPU 内存不足处理种子 {seed_val}，尝试清理缓存后继续")
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    logging.error(f"CUDA 错误处理种子 {seed_val}: {e}")
                    torch.cuda.empty_cache()
                else:
                    logging.exception(f"运行时错误处理种子 {seed_val}")
            except Exception:
                logging.exception(f"未知错误处理种子 {seed_val}")
            finally:
                pbar.update(1)


def main() -> None:
    init_logging()

    log_and_print(f"Torch use device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024
        log_and_print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # 解析命令行参数
    args = parse_args()
    set_verbose_output(not args.quiet)
    set_count_only_mode(args.count_only)
    if args.count_only:
        clear_count_only_stats()

    try:
        mode, radius, threshold = get_user_inputs(args)
    except ValueError as e:
        print(f"❌ 输入错误: {e}")
        sys.exit(1)

    log_and_print(
        f"mode or single seed number = {'multiple seeds' if mode == DEFAULT_MODE else mode}\nradius = {radius}\nthreshold = {threshold}"
    )

    # 处理检查点选项
    if args.clear_checkpoint:
        clear_checkpoint()
        log_and_print("🗑️ 检查点已清除")

    if torch.cuda.is_available() and should_warmup(args):
        # 默认只预热计算路径，更快；完整流水线在基准脚本中单独预热
        try:
            warmup_cudagraphs(full_pipeline=False)
        except Exception:
            logging.exception("CUDA warmup failed, continue without warmup")

    try:
        run(mode, radius, threshold, resume=args.resume)
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断，正在安全退出...")
        logging.info("Program interrupted by user.")
    except Exception:
        logging.exception("Unexpected error in main")
    finally:
        # 强制刷写检查点，保证中断/异常也不丢进度
        flush_checkpoints(force=True)

        # 显示结果摘要
        summary = get_results_summary()
        if summary['count'] > 0:
            log_and_print(f"\n📈 结果统计:")
            log_and_print(f"   总结果数: {summary['count']}")
            log_and_print(f"   最高史莱姆区块数: {summary['max']}")
            log_and_print(f"   平均史莱姆区块数: {summary['avg']:.1f}")
            log_and_print(f"   涉及种子数: {summary['unique_seeds']}")

            # 显示 Top 5 结果
            top_results = get_top_results(5)
            if top_results:
                log_and_print(f"\n🏆 Top 5 结果:")
                for i, r in enumerate(top_results, 1):
                    log_and_print(f"   {i}. 种子 {r['seed']}: {r['slime_chunks']} 区块 @ ({r['afk_x']}, {r['afk_z']})")

        if is_count_only_mode():
            count_summary = get_count_only_summary()
            log_and_print("\n⚡ 仅计数模式统计:")
            log_and_print(f"   已处理种子数: {count_summary['processed_seeds']}")
            log_and_print(f"   命中种子数: {count_summary['hit_seeds']}")
            log_and_print(f"   命中率: {count_summary['hit_rate'] * 100:.2f}%")
            log_and_print(f"   最佳命中计数: {count_summary['best_count']}")

        # 保存结果到 CSV（count-only 模式不保存明细）
        if not is_count_only_mode():
            csv_path = save_results_to_csv()
            if csv_path:
                log_and_print(f"\n📊 结果已保存到: {csv_path}")
        logging.shutdown()


if __name__ == "__main__":
    main()
