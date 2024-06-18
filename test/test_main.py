from unittest.mock import patch


def test_get_user_inputs(monkeypatch):
    from src.main import (
        get_user_inputs,
        DEFAULT_MODE,
        DEFAULT_RADIUS,
        DEFAULT_THRESHOLD,
    )

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
    from src.main import generate_seeds, DEFAULT_MODE

    for _ in range(100):
        seed_generator = generate_seeds(DEFAULT_MODE)
        seed = next(seed_generator)
        assert -(2**63) <= seed <= 2**63 - 1

    seed_generator = generate_seeds(0)
    seed = next(seed_generator)
    assert seed == 0


def test_detect_slime_chunk():
    import torch
    from src.main import detect_slime_chunk, device

    real_detected_chunks = torch.zeros((5, 5), dtype=torch.bool, device=device)
    real_detected_chunks[2, 0] = True
    real_detected_chunks[4, 4] = True

    detected_chunks = detect_slime_chunk(0, 2)

    print("Detected Chunks:")
    print(detected_chunks)

    print("Expected Chunks:")
    print(real_detected_chunks)

    assert torch.equal(detected_chunks, real_detected_chunks)


@patch("torch.randint")
def test_run(mock_randint, capsys):
    from src.main import run
    import torch

    mock_randint.return_value = torch.tensor([0])
    run(0, 0, 0)
    captured = capsys.readouterr()
    assert captured.out == "史莱姆区块数: 12, 种子: 0, 挂机点区块位置: (0, 0)\n"


def test_is_slime_chunk():
    from src.main import next_int, get_random_seed, device
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
        is_slime_chunk_results = next_int(seeds) % 10 == 0

        return is_slime_chunk_results.item()

    assert is_slime_chunk(0, -50, -50) is False
