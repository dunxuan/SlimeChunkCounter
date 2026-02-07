from unittest.mock import patch


def test_get_user_inputs(monkeypatch):
    from src.main import get_user_inputs
    from src.config import DEFAULT_MODE, DEFAULT_RADIUS, DEFAULT_THRESHOLD

    def mock_inputs(inputs):
        inputs_iter = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _: next(inputs_iter))

    test_cases = [
        (["10000", "10000", "100"], (10000, 10000, 100)),
        (["m", "16000", "50"], (DEFAULT_MODE, 16000, 50)),
        (["", "", ""], (DEFAULT_MODE, DEFAULT_RADIUS, DEFAULT_THRESHOLD)),
    ]

    for inputs, expected_output in test_cases:
        mock_inputs(inputs)
        assert get_user_inputs() == expected_output


def test_generate_seeds():
    from src.main import generate_seeds
    from src.config import DEFAULT_MODE

    for _ in range(100):
        seed_generator = generate_seeds(DEFAULT_MODE)
        seed = next(seed_generator)
        assert -(2**63) <= seed <= 2**63 - 1

    seed_generator = generate_seeds(0)
    seed = next(seed_generator)
    assert seed == 0


def test_detect_slime_chunk():
    import torch
    from src.main import detect_slime_chunk
    from src.config import device

    real_detected_chunks = torch.zeros((5, 5), dtype=torch.bool, device=device)
    real_detected_chunks[2, 0] = True
    real_detected_chunks[4, 4] = True

    # 收集生成器结果
    full_map = torch.zeros((5, 5), dtype=torch.bool, device=device)
    for x_start, z_start, chunk_tensor in detect_slime_chunk(0, 2):
        x_idx = x_start + 2
        z_idx = z_start + 2
        valid_h = min(chunk_tensor.shape[0], 5 - z_idx)
        valid_w = min(chunk_tensor.shape[1], 5 - x_idx)
        if valid_h > 0 and valid_w > 0:
            full_map[z_idx : z_idx + valid_h, x_idx : x_idx + valid_w] = chunk_tensor[
                :valid_h, :valid_w
            ]

    print("Detected Chunks:")
    print(full_map)

    print("Expected Chunks:")
    print(real_detected_chunks)

    assert torch.equal(full_map, real_detected_chunks)


@patch("torch.randint")
def test_run(mock_randint, capsys):
    from src.main import run
    import torch

    mock_randint.return_value = torch.tensor([0])
    # 使用 threshold=1 确保能匹配到结果
    run(0, 7, 1)
    captured = capsys.readouterr()
    # 验证输出包含预期的种子信息
    assert "种子: 0" in captured.out
    assert "史莱姆区块数:" in captured.out


def test_is_slime_chunk():
    from src.main import next_int, get_random_seed
    from src.config import device
    import torch

    def is_slime_chunk(worldSeed, chunkX, chunkZ, device=device):
        """
        检测具体某个区块是否是史莱姆区块

        Args:
            seed (int): 世界种子
            chunkX (int): 区块的 X 坐标
            chunkZ (int): 区块的 Z 坐标
            device (torch.device): 运算设备

        Returns:
            bool: 如果是史莱姆区块则返回 True, 否则返回 False
        """
        worldSeed = torch.tensor(worldSeed, dtype=torch.int64, device=device)
        chunkX = torch.tensor(chunkX, dtype=torch.int32, device=device)
        chunkZ = torch.tensor(chunkZ, dtype=torch.int32, device=device)

        seeds = get_random_seed(worldSeed, chunkX, chunkZ)
        # next_int 返回 u % 10，直接判断是否为 0
        is_slime_chunk_results = next_int(seeds) == 0

        return is_slime_chunk_results.item()

    assert is_slime_chunk(0, -50, -50) is False


def test_real_case_seed_7981483398353467015():
    """
    测试真实案例：种子 7981483398353467015 在位置 (-68, 437) 应该有 50 个史莱姆区块
    来源：log/2024-06-19-20-47-17.log
    """
    import torch
    import torch.nn.functional as F
    from src.main import detect_slime_chunk
    from src.config import device, PATTERN, SPAWN_RADIUS

    seed = 7981483398353467015
    # 预期结果：挂机点区块位置 (-68, 437)，史莱姆区块数 50
    expected_x = -68
    expected_z = 437
    expected_count = 50

    # 需要检测包含目标位置的范围
    # 原始搜索半径是 500，挂机点在 (-68, 437)
    # 检测半径 = 用户半径 + SPAWN_RADIUS
    check_radius = 500 + SPAWN_RADIUS

    # 收集检测结果并进行卷积
    found_match = False
    for x_start, z_start, chunk_tensor in detect_slime_chunk(seed, check_radius, device):
        chunk_float = chunk_tensor[None, None].float()
        conv_result = F.conv2d(chunk_float, PATTERN.float())

        # 检查是否有达到阈值的匹配
        pattern_h, pattern_w = PATTERN.shape[-2], PATTERN.shape[-1]
        valid_h = conv_result.shape[-2] - (pattern_h - 1)
        valid_w = conv_result.shape[-1] - (pattern_w - 1)

        if valid_h > 0 and valid_w > 0:
            valid_result = conv_result[:, :, :valid_h, :valid_w]
            match_mask = valid_result >= expected_count

            if match_mask.any():
                positions = torch.nonzero(match_mask, as_tuple=False)
                values = valid_result[match_mask]
                for pos, value in zip(positions, values):
                    h, w = pos[-2:].tolist()
                    x = x_start + w + SPAWN_RADIUS
                    z = z_start + h + SPAWN_RADIUS
                    if x == expected_x and z == expected_z and value.item() == expected_count:
                        found_match = True
                        print(f"找到匹配: 史莱姆区块数={value.item()}, 位置=({x}, {z})")
                        break
        if found_match:
            break

    assert found_match, f"未找到预期的匹配: 种子={seed}, 位置=({expected_x}, {expected_z}), 数量={expected_count}"
