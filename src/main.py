from datetime import datetime
import logging
import os
import torch
import torch.nn.functional as F
import concurrent.futures
from tqdm import tqdm
import sys

LOG_LEVEL = logging.INFO
DEFAULT_MODE = "M"
DEFAULT_RADIUS = 1024
DEFAULT_THRESHOLD = 50
SPAWN_RADIUS = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCK_SIZE = int(torch.cuda.mem_get_info()[1] / 1024 / 1024 * 1)
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
# get_random_seed的变量
v1 = torch.tensor(4987142, dtype=torch.int32, device=device)
v2 = torch.tensor(5947611, dtype=torch.int32, device=device)
v3 = torch.tensor(4392871, dtype=torch.int64, device=device)
v4 = torch.tensor(389711, dtype=torch.int32, device=device)
scrambler = torch.tensor(987234911, dtype=torch.int64, device=device)
multiplier = torch.tensor(0x5DEECE66D, dtype=torch.int64, device=device)
addend = torch.tensor(0xB, dtype=torch.int64, device=device)
mask = torch.tensor((1 << 48) - 1, dtype=torch.int64, device=device)


def init_logging():
    """
    初始化日志设置及目录
    """
    os.makedirs("log", exist_ok=True)
    logging.basicConfig(
        filename=f"log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s:\t\t%(message)s",
        encoding="UTF-8",
    )


def log_and_print(message):
    print(message)
    logging.info(message)


def get_user_inputs():
    """
    获取用户输入的运行模式, 检测半径和计数阈值

    Returns:
        tuple: 模式, 检测半径, 计数阈值
    """
    mode = (
        input(
            f"运行模式, 计算所有种子(multiple seeds)或单个种子(single seed) ([{DEFAULT_MODE}]ultiple seeds/种子值):"
        )
        .strip()
        .upper()
    )
    mode = DEFAULT_MODE if not mode or mode.startswith(DEFAULT_MODE) else int(mode)

    radius = input(f"区块检测半径 [{DEFAULT_RADIUS}]:")
    radius = int(radius) if radius else DEFAULT_RADIUS

    threshold = input(f"计数阈值 [{DEFAULT_THRESHOLD}]:")
    threshold = int(threshold) if threshold else DEFAULT_THRESHOLD

    return mode, radius, threshold


def generate_seeds(mode):
    """
    生成由模式控制的种子, 如果是multiple seeds模式(str)则随机生成, 否则使用模式指定的种子(种子值)

    Args:
        mode (str or int): 模式

    Yields:
        int: 种子值
    """
    if mode == DEFAULT_MODE:
        while True:
            yield torch.randint(-(2**63), 2**63 - 1, (), device=device)
    else:
        yield torch.tensor(mode, dtype=torch.int64, device=device)


@torch.compiler.disable
def get_random_seed(worldSeed, chunkX, chunkZ):
    """
    通过世界种子和区块坐标计算随机数生成种子

    Args:
        worldSeed (torch.int64): 世界种子
        chunkX (torch.int32): 区块X坐标
        chunkZ (torch.int32): 区块Z坐标
        device (torch.device): 计算设备

    Returns:
        torch.int64: 随机数种子
    """
    return (
        worldSeed
        + (chunkX * chunkX * v1).to(dtype=torch.int64)
        + (chunkX * v2).to(dtype=torch.int64)
        + (chunkZ * chunkZ).to(dtype=torch.int64) * v3
        + (chunkZ * v4).to(dtype=torch.int64)
        ^ scrambler
    )


@torch.compiler.disable
def next_int(seed: torch.Tensor) -> torch.Tensor:
    seed = (seed ^ multiplier) & mask

    def _next(s):
        s = (s * multiplier + addend) & mask
        s = s >> 17
        s = s.to(dtype=torch.int32)
        s = torch.where((s & (1 << 31)).bool(), s - (1 << 32), s)
        return s

    u = _next(seed)
    r = u % 10

    attempts = 0
    max_attempts = 10
    while (invalid := u - r + 9 < 0).any() and attempts < max_attempts:
        new_seed = (seed * multiplier + addend) & mask
        new_u = _next(new_seed)
        new_r = new_u % 10
        u = torch.where(invalid, new_u, u)
        r = torch.where(invalid, new_r, r)
        seed = torch.where(invalid, new_seed, seed)
        attempts += 1

    return r


def detect_slime_chunk(seed, chunk_radius, device=device, block_size=BLOCK_SIZE):
    """
    分块计算史莱姆区块，带重叠，避免 OOM 且保证卷积结果正确

    Args:
        seed (torch.int64): 世界种子
        chunk_radius (int): 检测半径
        block_size (int): 每个分块的有效大小

    Yields:
        (x_offset, z_offset, chunk_tensor): 分块的史莱姆区块数据
    """
    overlap = 15 - 1
    coords = torch.arange(
        -chunk_radius, chunk_radius + 1, dtype=torch.int32, device=device
    )

    for i in range(0, len(coords), block_size):
        for j in range(0, len(coords), block_size):
            x_block = coords[i : i + block_size + overlap]
            z_block = coords[j : j + block_size + overlap]

            x_coords, z_coords = torch.meshgrid(x_block, z_block, indexing="xy")
            seeds = get_random_seed(seed, x_coords.flatten(), z_coords.flatten())
            is_slime_chunk_results = next_int(seeds) % 10 == 0
            chunks = is_slime_chunk_results.reshape(x_coords.shape)

            yield x_block[0].item(), z_block[0].item(), chunks


def detect_and_log_matches(
    chunk_tensor, pattern_tensor, threshold, x_start, z_start, chunk_radius, seed
):
    """
    对输入的 chunk_tensor 进行卷积匹配，若匹配值 >= threshold，则打印匹配位置和数值。

    Args:
        chunk_tensor (torch.Tensor): [H, W] 的布尔或整数张量，表示当前分块的史莱姆区块
        pattern_tensor (torch.Tensor): [1, 1, H_p, W_p] 的卷积核
        threshold (int): 匹配阈值
        i (int): 当前块在全局 Y 方向的起始索引偏移
        j (int): 当前块在全局 X 方向的起始索引偏移
        chunk_radius (int): 全局检测半径（用于坐标还原）
        seed (torch.Tensor): 当前世界种子（用于打印）
    """
    chunk_tensor = chunk_tensor[None, None].float()
    conv_result = F.conv2d(chunk_tensor, pattern_tensor)
    valid_result = conv_result[
        :, :, : -(PATTERN.shape[-2] - 1), : -(PATTERN.shape[-1] - 1)
    ]

    mask = valid_result >= threshold
    if mask.any():
        positions = torch.nonzero(mask, as_tuple=False)
        values = valid_result[mask]
        for pos, value in zip(positions, values):
            h, w = pos[-2:].tolist()
            x = x_start + w + SPAWN_RADIUS
            z = z_start + h + SPAWN_RADIUS
            log_and_print(
                f"\n史莱姆区块数: {value.item():.0f}, 种子: {seed.item()}, 挂机点区块位置: ({x}, {z})"
            )


@torch.compile(mode="reduce-overhead", dynamic=False)
def process_seed(seed, threshold, chunk_radius, pattern_tensor):
    if not isinstance(seed, torch.Tensor):
        seed = torch.tensor(seed, dtype=torch.int64, device=device)
    else:
        seed = seed.to(device, dtype=torch.int64)

    for x_start, z_start, chunk_tensor in detect_slime_chunk(seed, chunk_radius):
        detect_and_log_matches(
            chunk_tensor,
            pattern_tensor,
            threshold,
            x_start,
            z_start,
            chunk_radius,
            seed,
        )


def run(mode, radius, threshold):
    chunk_radius = radius + SPAWN_RADIUS
    pattern_tensor = PATTERN.float()

    # 单种子模式：同步执行
    if mode != DEFAULT_MODE:
        for seed in generate_seeds(mode):
            try:
                # 预计算总块数
                total_chunks = (
                    (2 * chunk_radius + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
                ) ** 2

                # 创建 tqdm 进度条，显示完整进度
                with tqdm(
                    total=total_chunks,
                    desc=f"Processing seed {seed.item()}",
                    dynamic_ncols=True,
                    bar_format="{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} blocks | {rate_fmt} | ETA: {remaining}",
                    leave=True,
                ) as pbar:
                    for x_start, z_start, chunk_tensor in detect_slime_chunk(
                        seed, chunk_radius
                    ):
                        detect_and_log_matches(
                            chunk_tensor,
                            pattern_tensor,
                            threshold,
                            x_start,
                            z_start,
                            chunk_radius,
                            seed,
                        )
                        pbar.update(1)  # 手动更新进度
            except Exception:
                logging.exception("Error processing single seed")
        return

    # 多种子模式：异步线程池
    with tqdm(
        desc="Processing seeds",
        dynamic_ncols=True,
        bar_format="{desc} | {rate_fmt} | Total: {n_fmt}",
    ) as pbar:

        def wrapped_process_seed(seed):
            try:
                process_seed(seed, threshold, chunk_radius, pattern_tensor)
            except Exception:
                logging.exception(f"Error processing seed {seed.item()}")
            finally:
                pbar.update(1)  # 任务完成才更新，确保速率准确

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for seed in generate_seeds(mode):
                future = executor.submit(wrapped_process_seed, seed)
                futures.append(future)
                # 不在这里 update，等任务完成再 update（速率更真实）

                # 防止任务堆积，定期清理已完成
                if len(futures) > 100:
                    done, not_done = concurrent.futures.wait(
                        futures,
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    futures = list(not_done)


def main():
    init_logging()

    log_and_print(f"Torch use device: {device}")

    mode, radius, threshold = get_user_inputs()
    log_and_print(
        f"mode or single seed number = {'multiple seeds' if mode == DEFAULT_MODE else mode}\nradius = {radius}\nthreshold = {threshold}"
    )

    try:
        run(mode, radius, threshold)
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断，正在安全退出...")
        logging.info("Program interrupted by user.")
        logging.shutdown()
        sys.exit(0)
    except Exception:
        logging.exception("Unexpected error in main")
        logging.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
