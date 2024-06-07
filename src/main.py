from datetime import datetime
from multiprocessing import Pool
import random
import logging
import os
import numpy as np
from javarandom import Random
import torch
import torch.nn.functional as F

DEBUG = False
LOG_LEVEL = logging.INFO if not DEBUG else logging.DEBUG
DEFAULT_MODE = "M"
DEFAULT_RADIUS = 16000
DEFAULT_THRESHOLD = 50
CHUNK_SIZE = 16
SPAWN_RADIUS = 7
# fmt: off
PATTERN = np.array([
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
])
# fmt: on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logging():
    """
    初始化日志设置及目录
    """
    if not os.path.exists("log"):
        os.mkdir("log")
    logging.basicConfig(
        filename=f"log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s:\t\t\t%(message)s",
    )


def get_mode():
    """
    用户输入运行模式, 是只计算单个种子还是随机种子循环运行

    Returns:
        str or int: 模式
    """
    mode = (
        input(
            f"运行模式, 计算所有种子(multiple seeds)或单个种子(singel seed) ([{DEFAULT_MODE}]ultiple seeds/种子值):"
        )
        .strip()
        .upper()
    )
    return DEFAULT_MODE if not mode or mode.startswith(DEFAULT_MODE) else int(mode)


def get_radius():
    """
    用户输入主世界检测范围的半径, 以挂机点为准

    Returns:
        int: 检测半径, 默认为DEFAULT_RADIUS
    """
    radius = input(f"检测半径 [{DEFAULT_RADIUS}]:")
    return int(radius) if radius else DEFAULT_RADIUS


def get_threshold():
    """
    用户输入刷怪范围内史莱姆区块阈值, 阈值及其以上保留

    Returns:
        int: 计数阈值, 默认为DEFAULT_THRESHOLD
    """
    threshold = input(f"计数阈值 [{DEFAULT_THRESHOLD}]:")
    return int(threshold) if threshold else DEFAULT_THRESHOLD


def is_slime_chunk(chunkX, chunkZ, worldSeed):
    """
    判断该区块是否是史莱姆区块

    Args:
        chunkX (np.int64): 区块X坐标
        chunkZ (np.int64): 区块Z坐标
        worldSeed (np.int64): 世界种子

    Returns:
        bool: 是否是史莱姆区块
    """
    seed = (
        worldSeed
        + chunkX * (chunkX * 4987142 + 5947611)
        + chunkZ * (chunkZ * 4392871 + 389711)
        ^ 987234911
    )
    return Random(seed).nextInt(10) == 0


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
            yield random.randint(-(2**32), 2**32 - 1)
    else:
        yield mode


def detect_slime_chunk(seed, chunk_radius):
    """
    获取用 seed 生成的世界的在 chunk_radius 半径范围内的区块表

    Args:
        seed (int): 世界种子
        chunk_radius (int): 检测半径（区块）

    Returns:
        np.ndarray: 检测完成的区块表
    """
    x_coords, y_coords = np.meshgrid(
        np.arange(-chunk_radius, chunk_radius + 1, dtype=np.int64),
        np.arange(-chunk_radius, chunk_radius + 1, dtype=np.int64),
        indexing="ij",
    )

    params = [(x, y, seed) for x, y in zip(x_coords.flatten(), y_coords.flatten())]

    with Pool() as pool:
        results = pool.starmap(is_slime_chunk, params)

    chunks = np.array(results).reshape(x_coords.shape)
    return chunks


def run(mode, radius, threshold, device=device):
    """
    循环获取随机世界种子, 计算该世界在 radius 半径里刷怪范围内的 阈值>=threshold 的史莱姆区块数

    Args:
        mode (str or int): 运行模式值
        radius (int): 检测半径
        threshold (int): 计数阈值
        device (tensor.device): 默认即可, 参数在此处作为调试使用
    """
    afk_radius = radius // CHUNK_SIZE
    chunk_radius = afk_radius + SPAWN_RADIUS

    pattern_tensor = (
        torch.tensor(PATTERN, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )
    logging.debug(f"pattern_tensor = {pattern_tensor}")

    for seed in generate_seeds(mode):
        logging.debug(f"seed = {seed}")

        detected_chunks = detect_slime_chunk(seed, chunk_radius)
        logging.debug(f"detected_chunks = {detected_chunks}")

        chunk_tensor = (
            torch.tensor(detected_chunks, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        logging.debug(f"chunk_tensor = {chunk_tensor}")

        conv_result = F.conv2d(chunk_tensor, pattern_tensor)
        logging.debug(f"conv_result= {conv_result}")

        if (conv_result >= threshold).sum().item() > 0:
            _, _, H_out, W_out = conv_result.shape

            for h, w in np.ndindex(H_out, W_out):
                value = conv_result[0, 0, h, w]
                x = h - chunk_radius + 7
                z = w - chunk_radius + 7
                message = f"史莱姆区块数: {value:.0f}, 种子: {seed}, 挂机点区块位置: ({x}, {z})"
                print(message)
                logging.info(message)
        else:
            logging.debug(
                f"This World isn't have exceed the threshold value: seed = {seed}"
            )

        if DEBUG or mode != DEFAULT_MODE:
            break


def main():
    # 初始化日志设置
    init_logging()

    # 获取运行模式，并记录日志
    mode = get_mode()
    logging.info(
        f"mode or singel seed number = {'multiple seeds' if mode == DEFAULT_MODE else mode}"
    )

    # 获取检测半径, 并记录日志
    radius = get_radius()
    logging.info(f"radius = {radius}")

    # 获取计数阈值, 并记录日志
    threshold = get_threshold()
    logging.info(f"threshold = {threshold}")

    # 记录使用设备日志, CPU还是CUDA
    logging.info(f"Torch use device: {device}")

    # 开始运行
    run(mode, radius, threshold, device)


if __name__ == "__main__":
    main()
