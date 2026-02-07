import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.main import detect_slime_chunk
from src.config import BLOCK_SIZE, SPAWN_RADIUS, PATTERN
import matplotlib.ticker as ticker
import os
import argparse

DEFAULT_RADIUS = 500
OUTPUT_DIR = "results"
SPAWN_RANGE = 8  # 刷怪范围（区块）


@torch.no_grad()
def get_full_chunk_map(seed: int, radius: int, device: torch.device) -> torch.Tensor:
    """
    将 detect_slime_chunk 生成器的结果合并为完整的区块地图

    Args:
        seed: 世界种子
        radius: 区块检测半径
        device: 计算设备

    Returns:
        torch.Tensor: 完整的史莱姆区块地图
    """
    size = 2 * radius + 1
    full_map = torch.zeros((size, size), dtype=torch.bool, device=device)

    for x_start, z_start, chunk_tensor in detect_slime_chunk(seed, radius, device):
        # 计算在完整地图中的位置
        x_idx = x_start + radius
        z_idx = z_start + radius

        # 获取有效区域大小（去除重叠部分）
        valid_h = min(BLOCK_SIZE, chunk_tensor.shape[0] - 14, size - z_idx)
        valid_w = min(BLOCK_SIZE, chunk_tensor.shape[1] - 14, size - x_idx)

        if valid_h > 0 and valid_w > 0:
            full_map[z_idx : z_idx + valid_h, x_idx : x_idx + valid_w] = chunk_tensor[
                :valid_h, :valid_w
            ]

    return full_map


@torch.no_grad()
def count_slime_chunks_at_position(chunk_map: torch.Tensor, chunk_x: int, chunk_z: int, radius: int) -> int:
    """
    计算指定位置的刷怪范围内的史莱姆区块数量

    Args:
        chunk_map: 完整的区块地图
        chunk_x: 挂机点 X 坐标（区块）
        chunk_z: 挂机点 Z 坐标（区块）
        radius: 地图半径

    Returns:
        int: 史莱姆区块数量
    """
    # 使用 PATTERN 进行卷积计算
    chunk_tensor = chunk_map[None, None].float()
    conv_result = F.conv2d(chunk_tensor, PATTERN.float())

    # 计算挂机点在卷积结果中的位置
    # 卷积后尺寸减少 (pattern_size - 1)
    pattern_offset = (PATTERN.shape[-1] - 1) // 2
    result_x = radius + chunk_x - pattern_offset
    result_z = radius + chunk_z - pattern_offset

    if 0 <= result_x < conv_result.shape[-1] and 0 <= result_z < conv_result.shape[-2]:
        return int(conv_result[0, 0, result_z, result_x].item())
    return 0


def generate_chunk_map_image(
    seed: int,
    radius: int,
    highlight_pos: tuple = None,
    output_path: str = None,
    show: bool = True,
    view_radius: int = None,
) -> str:
    """
    生成史莱姆区块分布图

    Args:
        seed: 世界种子
        radius: 区块检测半径
        highlight_pos: 高亮显示的位置 (x, z)，通常是挂机点（区块坐标）
        output_path: 输出文件路径，如果为 None 则自动生成
        show: 是否显示图片
        view_radius: 显示范围半径（区块），如果为 None 则显示全部

    Returns:
        str: 保存的文件路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detected_chunks = get_full_chunk_map(seed, radius, device)
    detected_chunks_numpy = detected_chunks.cpu().numpy()

    # 自动计算史莱姆区块数量
    slime_count = None
    if highlight_pos:
        chunk_x, chunk_z = highlight_pos
        slime_count = count_slime_chunks_at_position(detected_chunks, chunk_x, chunk_z, radius)

    # 如果指定了挂机点和显示范围，只显示周围区域
    if highlight_pos and view_radius:
        chunk_x, chunk_z = highlight_pos
        # 计算裁剪范围（相对于地图中心）
        center = radius
        x_min = max(0, center + chunk_x - view_radius)
        x_max = min(detected_chunks_numpy.shape[1], center + chunk_x + view_radius + 1)
        z_min = max(0, center + chunk_z - view_radius)
        z_max = min(detected_chunks_numpy.shape[0], center + chunk_z + view_radius + 1)

        detected_chunks_numpy = detected_chunks_numpy[z_min:z_max, x_min:x_max]

        # 更新坐标范围
        extent_x_min = chunk_x - view_radius - 0.5
        extent_x_max = chunk_x + view_radius + 0.5
        extent_z_min = chunk_z - view_radius - 0.5
        extent_z_max = chunk_z + view_radius + 0.5
    else:
        extent_x_min = -radius - 0.5
        extent_x_max = radius + 0.5
        extent_z_min = -radius - 0.5
        extent_z_max = radius + 0.5

    # 创建高分辨率图形
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)

    # 绘制史莱姆区块
    ax.imshow(
        detected_chunks_numpy,
        cmap="Greens",
        interpolation="none",
        extent=(extent_x_min, extent_x_max, extent_z_max, extent_z_min),
    )

    # 添加网格线（在区块边界上，即 x.5 的位置）
    display_range = view_radius if view_radius else radius
    if display_range <= 50:
        # 小范围时显示每个区块的网格线（在边界上）
        # 网格线位置：-0.5, 0.5, 1.5, 2.5 ... （区块边界）
        grid_x = [i - 0.5 for i in range(int(extent_x_min + 1), int(extent_x_max + 1) + 1)]
        grid_z = [i - 0.5 for i in range(int(extent_z_min + 1), int(extent_z_max + 1) + 1)]

        for x in grid_x:
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.7)
        for z in grid_z:
            ax.axhline(z, color='gray', linewidth=0.5, alpha=0.7)

        # 刻度标签在区块中心（整数位置）
        tick_x = list(range(int(extent_x_min + 0.5), int(extent_x_max + 0.5) + 1))
        tick_z = list(range(int(extent_z_min + 0.5), int(extent_z_max + 0.5) + 1))
        ax.set_xticks(tick_x)
        ax.set_yticks(tick_z)
    else:
        # 大范围时显示稀疏网格
        ax.grid(True, alpha=0.3)

    # 坐标轴原点线（在区块边界 -0.5 处，即原点区块的左/上边界）
    if extent_x_min < -0.5 < extent_x_max:
        ax.axvline(-0.5, color="black", linewidth=1.5)
    if extent_z_min < -0.5 < extent_z_max:
        ax.axhline(-0.5, color="black", linewidth=1.5)

    # 高亮显示挂机点
    if highlight_pos:
        chunk_x, chunk_z = highlight_pos
        # 绘制挂机点标记（小一点的星号）
        ax.plot(chunk_x, chunk_z, 'r*', markersize=8, label=f'AFK Chunk ({chunk_x}, {chunk_z})')

        # 刷怪范围圈（128格 = 8区块）
        spawn_radius = 8
        circle_spawn = plt.Circle((chunk_x, chunk_z), spawn_radius, fill=False, color='red', linewidth=1.5, linestyle='-', label='Spawn Range (128 blocks)')
        ax.add_patch(circle_spawn)

        # 不刷怪范围圈（24格 ≈ 1.5区块）
        no_spawn_radius = 1.5
        circle_no_spawn = plt.Circle((chunk_x, chunk_z), no_spawn_radius, fill=False, color='blue', linewidth=1.5, linestyle='--', label='No Spawn (24 blocks)')
        ax.add_patch(circle_no_spawn)

        ax.legend(loc='upper right', fontsize=9)

    # 刻度格式
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # 标题
    if highlight_pos:
        chunk_x, chunk_z = highlight_pos
        block_x = chunk_x * 16 + 8
        block_z = chunk_z * 16 + 8
        title = f"Slime Chunk Map - Seed: {seed}"
        if slime_count:
            title += f" - Slime Chunks: {slime_count}"
        title += f"\nAFK Point: ({block_x}, {block_z})"
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Slime Chunk Map - Seed: {seed}", fontsize=14)
    ax.set_xlabel("X (chunks)")
    ax.set_ylabel("Z (chunks)")

    # 保存图片
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"chunk_map_{seed}.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"📷 图片已保存到: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成史莱姆区块分布图")
    parser.add_argument("-s", "--seed", type=int, help="世界种子")
    parser.add_argument("-r", "--radius", type=int, default=DEFAULT_RADIUS, help=f"区块检测半径 (默认: {DEFAULT_RADIUS})")
    parser.add_argument("-x", "--afk-x", type=int, help="挂机点 X 坐标（区块）")
    parser.add_argument("-z", "--afk-z", type=int, help="挂机点 Z 坐标（区块）")
    parser.add_argument("-v", "--view-radius", type=int, default=10, help="显示范围半径（区块，默认: 10）")
    parser.add_argument("-o", "--output", type=str, help="输出文件路径")
    parser.add_argument("--no-show", action="store_true", help="不显示图片")
    parser.add_argument("--full", action="store_true", help="显示完整地图（不裁剪）")

    args = parser.parse_args()

    # 获取种子
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(input("种子 (-2^63 ~ 2^63 - 1): "))

    # 获取半径
    if args.radius:
        radius = args.radius
    else:
        radius_input = input(f"区块检测半径 [{DEFAULT_RADIUS}]: ")
        radius = int(radius_input) if radius_input else DEFAULT_RADIUS

    # 获取挂机点
    highlight_pos = None
    if args.afk_x is not None and args.afk_z is not None:
        highlight_pos = (args.afk_x, args.afk_z)

    # 确定显示范围
    view_radius = None if args.full else args.view_radius

    # 生成图片
    generate_chunk_map_image(
        seed=seed,
        radius=radius,
        highlight_pos=highlight_pos,
        output_path=args.output,
        show=not args.no_show,
        view_radius=view_radius if highlight_pos else None,
    )


if __name__ == "__main__":
    main()
