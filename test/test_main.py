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
        assert -(2**32) <= seed <= 2**32 - 1

    seed_generator = generate_seeds(0)
    seed = next(seed_generator)
    assert seed == 0


def test_detect_slime_chunk():
    import torch
    from src.main import detect_slime_chunk, device

    real_detected_chunks = torch.zeros((5, 5), dtype=torch.bool, device=device)
    real_detected_chunks[0, 2] = True
    real_detected_chunks[4, 4] = True

    detected_chunks = detect_slime_chunk(0, 2)

    print("Detected Chunks:")
    print(detected_chunks)

    print("Expected Chunks:")
    print(real_detected_chunks)

    assert torch.equal(detected_chunks, real_detected_chunks)


@patch("random.randint")
def test_run(mock_randint, capsys):
    from src.main import run

    mock_randint.return_value = 0
    run(0, 0, 0)
    captured = capsys.readouterr()
    assert captured.out == "史莱姆区块数: 12, 种子: 0, 挂机点区块位置: (0, 0)\n"
