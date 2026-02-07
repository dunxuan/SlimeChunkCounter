"""
SlimeChunkCounter 配置文件
包含所有可配置的常量和默认值
"""

import logging
import math
import os
import sys
import torch

# ==================== 日志配置 ====================
LOG_LEVEL = logging.INFO

# ==================== 默认参数 ====================
DEFAULT_MODE = "M"  # 默认运行模式：M = 多种子模式
DEFAULT_RADIUS = 1024  # 默认检测半径（区块数）
DEFAULT_THRESHOLD = 50  # 默认计数阈值
DEFAULT_BLOCK_SIZE = 1024  # CPU 模式下的默认块大小
MIN_BLOCK_SIZE = 256
BLOCK_SIZE_ALIGNMENT = 128
EST_BYTES_PER_CELL = 20  # 每个网格点估算显存占用（含中间张量）
BASE_BLOCK_SCALE = 64
DEFAULT_BLOCK_TUNING_FACTOR = 0.80  # 经验校正：1650Ti 实测最优更接近 1024

# ==================== 游戏常量 ====================
SPAWN_RADIUS = 7  # 史莱姆生成半径（区块）
MAX_SLIME_CHUNKS = 177  # 15x15 圆形区域最大史莱姆区块数

# ==================== 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_test_mode() -> bool:
    """是否为测试/基准模式。"""
    return (
        ("pytest" in sys.modules)
        or (os.environ.get("PYTEST_CURRENT_TEST") is not None)
        or (os.environ.get("SCC_TEST_MODE") == "1")
    )


IS_TEST_MODE = _is_test_mode()


def _get_gpu_memory_fraction() -> float:
    """
    获取显存占用比例。

    优先级：
    1) 环境变量 SCC_GPU_MEMORY_FRACTION
    2) 测试/基准模式默认 0.50
    3) 常规运行默认 1.00（不限制）
    """
    env_value = os.environ.get("SCC_GPU_MEMORY_FRACTION")
    if env_value is not None:
        try:
            value = float(env_value)
            if 0 < value <= 1:
                return value
        except (TypeError, ValueError):
            pass

    return 0.50 if IS_TEST_MODE else 1.00


def _get_block_tuning_factor() -> float:
    """
    获取 block size 经验校正系数。

    可通过环境变量 SCC_BLOCK_TUNING_FACTOR 覆盖，默认 0.80。
    """
    env_value = os.environ.get("SCC_BLOCK_TUNING_FACTOR")
    if env_value is not None:
        try:
            value = float(env_value)
            if 0.1 <= value <= 2.0:
                return value
        except (TypeError, ValueError):
            pass
    return DEFAULT_BLOCK_TUNING_FACTOR


# 常规 run.py 默认不限制显存；测试/基准默认 50%
GPU_MEMORY_FRACTION = _get_gpu_memory_fraction()
BLOCK_TUNING_FACTOR = _get_block_tuning_factor()

# 启用 cuDNN benchmark 模式（对固定尺寸输入自动选择最快算法）
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # 仅在测试/基准模式限制显存占用，常规运行默认不限制
    if GPU_MEMORY_FRACTION < 1.0:
        torch.cuda.memory.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

# 根据设备类型设置块大小
if torch.cuda.is_available():
    # 根据显卡参数给出经验最优分块大小（单一公式）
    # 公式：block_size = 64 * tuning * sqrt(有效显存GiB * SM数量 * 计算能力)
    # - 有效显存GiB: total_memory * GPU_MEMORY_FRACTION
    # - 计算能力: major + minor/10
    # - tuning: 经验校正系数（默认 0.80）
    # 同时用“显存估算上界”动态约束，避免固定上限导致不同显卡利用不足。
    # 最终按 BLOCK_SIZE_ALIGNMENT 对齐。
    props = torch.cuda.get_device_properties(0)
    effective_mem_bytes = int(props.total_memory * GPU_MEMORY_FRACTION)
    total_mem_gib = props.total_memory / (1024 ** 3)
    effective_mem_gib = total_mem_gib * GPU_MEMORY_FRACTION
    sm_count = max(int(props.multi_processor_count), 1)
    compute_capability = max(props.major + props.minor / 10.0, 1.0)

    # 经验公式给出候选值
    raw_block_size = int(
        BASE_BLOCK_SCALE
        * BLOCK_TUNING_FACTOR
        * math.sqrt(effective_mem_gib * sm_count * compute_capability)
    )

    # 动态显存上界：sqrt(可用元素数)，并留 50% 余量给卷积/缓存/临时张量
    max_elements = max(effective_mem_bytes // EST_BYTES_PER_CELL // 2, 1)
    mem_bound_size = int(math.sqrt(max_elements))

    # 最终值取候选值与动态显存上界的较小者，再做对齐
    unconstrained = min(raw_block_size, mem_bound_size)
    aligned_block_size = (unconstrained // BLOCK_SIZE_ALIGNMENT) * BLOCK_SIZE_ALIGNMENT
    BLOCK_SIZE = max(MIN_BLOCK_SIZE, aligned_block_size)
else:
    BLOCK_SIZE = DEFAULT_BLOCK_SIZE

# ==================== 史莱姆区块检测模式 ====================
# 15x15 的圆形检测模式，中心 3x3 为玩家站立位置（不计入）
PATTERN = (
    torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float,
        device=device,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

# 预计算 FP16 版本的 PATTERN（如果 GPU 支持）
USE_FP16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
PATTERN_FP16 = PATTERN.half() if USE_FP16 else PATTERN

# ==================== 随机数生成常量 ====================
# Minecraft Java 版史莱姆区块判定算法使用的常量
v1 = torch.tensor(4987142, dtype=torch.int32, device=device)
v2 = torch.tensor(5947611, dtype=torch.int32, device=device)
v3 = torch.tensor(4392871, dtype=torch.int64, device=device)
v4 = torch.tensor(389711, dtype=torch.int32, device=device)
scrambler = torch.tensor(987234911, dtype=torch.int64, device=device)

# Java Random 类使用的常量
multiplier = torch.tensor(0x5DEECE66D, dtype=torch.int64, device=device)
addend = torch.tensor(0xB, dtype=torch.int64, device=device)
mask = torch.tensor((1 << 48) - 1, dtype=torch.int64, device=device)

# ==================== 输入验证限制 ====================
MAX_RADIUS = 100000  # 最大检测半径
MIN_SEED = -(2**63)  # 最小种子值
MAX_SEED = 2**63 - 1  # 最大种子值
