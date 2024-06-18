import torch
import matplotlib.pyplot as plt
from src.main import detect_slime_chunk
import matplotlib.ticker as ticker

DEFAULT_RADIUS = 500

# 输入种子和半径
seed = int(input("种子 (-2^63 ~ 2^63 - 1):"))
radius_input = input(f"区块检测半径 [{DEFAULT_RADIUS}]:")
radius = int(radius_input) if radius_input else DEFAULT_RADIUS

# 计算区块
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detected_chunks = detect_slime_chunk(seed, radius, device=device)
detected_chunks_numpy = detected_chunks.cpu().numpy()

plt.imshow(
    detected_chunks_numpy,
    cmap="Greens",
    interpolation="none",
    extent=(
        -radius - 0.5,
        radius + 0.5,
        radius + 0.5,
        -radius - 0.5,
    ),
)
plt.grid(True)

# 坐标系轴
plt.axhline(-0.5, color="black", linewidth=0.5)
plt.axvline(-0.5, color="black", linewidth=0.5)

# 刻度值格式
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

plt.show()
