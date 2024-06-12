import torch
import matplotlib.pyplot as plt
from src.main import detect_slime_chunk
import matplotlib.ticker as ticker

DEFAULT_RADIUS = 10

# seed = int(input("种子 (-2^32 ~ 2^32 - 1):"))
seed = 0

# radius_input = input(f"检测半径 [{DEFAULT_RADIUS}]:")
# radius = int(radius_input) if radius_input else DEFAULT_RADIUS
radius = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detected_chunks = detect_slime_chunk(seed, radius, device=device).t()

detected_chunks_numpy = detected_chunks.cpu().numpy()

plt.imshow(
    detected_chunks_numpy,
    cmap="Greens",
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

# 刻度
plt.gca().set_xticks([x - 0.5 for x in range(-radius, radius + 2)])
plt.gca().set_yticks([y - 0.5 for y in range(-radius, radius + 2)])

# 坐标值格式
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

plt.show()