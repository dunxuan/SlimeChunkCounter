from datetime import datetime
import logging
import os
import torch
import torch.nn.functional as F

DEBUG = False
LOG_LEVEL = logging.INFO if not DEBUG else logging.DEBUG
DEFAULT_MODE = "M"
DEFAULT_RADIUS = 500
DEFAULT_THRESHOLD = 50
SPAWN_RADIUS = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fmt: off
PATTERN = torch.tensor([
    [False, False,  False,  False,  False,  True,   True,   True,   True,   True,   False,  False,  False,  False,  False],
    [False, False,  False,  True,   True,   True,   True,   True,   True,   True,   True,   True,   False,  False,  False],
    [False, False,  True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   False,  False],
    [False, True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   False],
    [False, True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   False],
    [True,  True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True],
    [True,  True,   True,   True,   True,   True,   False,  False,  False,  True,   True,   True,   True,   True,   True],
    [True,  True,   True,   True,   True,   True,   False,  False,  False,  True,   True,   True,   True,   True,   True],
    [True,  True,   True,   True,   True,   True,   False,  False,  False,  True,   True,   True,   True,   True,   True],
    [True,  True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True],
    [False, True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   False],
    [False, True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   False],
    [False, False,  True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   True,   False,  False],
    [False, False,  False,  True,   True,   True,   True,   True,   True,   True,   True,   True,   False,  False,  False],
    [False, False,  False,  False,  False,  True,   True,   True,   True,   True,   False,  False,  False,  False,  False]
], dtype=torch.bool, device=device)
# fmt: on
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
    while mode == DEFAULT_MODE:
        yield torch.randint(-(2**63), 2**63 - 1, (1,), device=device).item()
    yield mode


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


def next_int(seed):
    """
    生成区块随机数

    Args:
        seed (torch.int64): 种子张量

    Returns:
        torch.int32: 随机整数张量
    """
    seed = (seed ^ multiplier) & mask

    def next():
        nonlocal seed
        seed = (seed * multiplier + addend) & mask
        return (seed >> 17).to(dtype=torch.int32)

    u = next()
    r = u % 10
    while torch.any(u - r + 9 < 0):
        u = next()
        r = u % 10

    return r.to(dtype=torch.int32)


def detect_slime_chunk(seed, chunk_radius, device=device):
    """
    获取用 seed 生成的世界的在 chunk_radius 半径范围内的区块表

    Args:
        seed (int): 世界种子
        chunk_radius (int): 检测半径（区块）
        device (torch.device): 运算设备

    Returns:
        torch.Tensor: 检测完成的区块表
    """
    chunk_range = torch.arange(
        -chunk_radius, chunk_radius + 1, dtype=torch.int32, device=device
    )
    x_coords, z_coords = torch.meshgrid(chunk_range, chunk_range, indexing="xy")

    chunkX = x_coords.flatten()
    chunkZ = z_coords.flatten()

    worldSeed = torch.tensor(seed, dtype=torch.int64, device=device)
    seeds = get_random_seed(worldSeed, chunkX, chunkZ)

    is_slime_chunk_results = next_int(seeds) % 10 == 0
    chunks = is_slime_chunk_results.reshape(x_coords.shape)

    return chunks


def run(mode, radius, threshold, device=device):
    """
    循环获取随机世界种子或使用用户给定种子值, 计算该世界在 radius 半径里刷怪范围内的 阈值>=threshold 的史莱姆区块数

    Args:
        mode (str or int): 运行模式值
        radius (int): 检测半径
        threshold (int): 计数阈值
        device (torch.device): 默认即可, 参数在此处作为调试使用
    """
    chunk_radius = radius + SPAWN_RADIUS

    pattern_tensor = PATTERN.float().unsqueeze(0).unsqueeze(0)
    logging.debug(f"pattern_tensor = {pattern_tensor}")

    for seed in generate_seeds(mode):
        try:
            logging.debug(f"seed = {seed}")

            detected_chunks = detect_slime_chunk(seed, chunk_radius)
            logging.debug(f"detected_chunks = {detected_chunks}")

            chunk_tensor = detected_chunks.float().unsqueeze(0).unsqueeze(0)
            logging.debug(f"chunk_tensor = {chunk_tensor}")

            conv_result = F.conv2d(chunk_tensor, pattern_tensor)
            logging.debug(f"conv_result= {conv_result}")

            mask = conv_result >= threshold
            if mask.sum().item() > 0:
                positions = torch.nonzero(mask, as_tuple=False)
                values = conv_result[mask]

                for pos, value in zip(positions, values):
                    h, w = pos[-2:].tolist()
                    x = h - chunk_radius + 7
                    z = w - chunk_radius + 7
                    message = f"史莱姆区块数: {value.item():.0f}, 种子: {seed}, 挂机点区块位置: ({x}, {z})"
                    log_and_print(message)
            else:
                logging.debug(
                    f"This World isn't have exceed the threshold value: seed = {seed}"
                )

            if mode != DEFAULT_MODE:
                break

        except KeyboardInterrupt:
            log_and_print("检测到用户中断 (Ctrl+C)，程序终止。")
            exit(0)


def main():
    init_logging()

    mode, radius, threshold = get_user_inputs()
    log_and_print(
        f"mode or single seed number = {'multiple seeds' if mode == DEFAULT_MODE else mode}\nradius = {radius}\nthreshold = {threshold}"
    )

    log_and_print(f"Torch use device: {device}")

    run(mode, radius, threshold, device=device)


if __name__ == "__main__":
    main()
