from pytest import MonkeyPatch  # noqa: F401
from unittest.mock import patch
import numpy as np


def test_is_slime_chunk():
    from src.main import is_slime_chunk

    assert is_slime_chunk(-2, -2, 0) is False
    assert is_slime_chunk(-2, -1, 0) is False
    assert is_slime_chunk(-2, 0, 0) is True
    assert is_slime_chunk(-2, 1, 0) is False
    assert is_slime_chunk(-2, 2, 0) is False

    assert is_slime_chunk(-1, -2, 0) is False
    assert is_slime_chunk(-1, -1, 0) is False
    assert is_slime_chunk(-1, 0, 0) is False
    assert is_slime_chunk(-1, 1, 0) is False
    assert is_slime_chunk(-1, 2, 0) is False

    assert is_slime_chunk(0, -2, 0) is False
    assert is_slime_chunk(0, -1, 0) is False
    assert is_slime_chunk(0, 0, 0) is False
    assert is_slime_chunk(0, 1, 0) is False
    assert is_slime_chunk(0, 2, 0) is False

    assert is_slime_chunk(1, -2, 0) is False
    assert is_slime_chunk(1, -1, 0) is False
    assert is_slime_chunk(1, 0, 0) is False
    assert is_slime_chunk(1, 1, 0) is False
    assert is_slime_chunk(1, 2, 0) is False

    assert is_slime_chunk(2, -2, 0) is False
    assert is_slime_chunk(2, -1, 0) is False
    assert is_slime_chunk(2, 0, 0) is False
    assert is_slime_chunk(2, 1, 0) is False
    assert is_slime_chunk(2, 2, 0) is True


def test_get_radius(monkeypatch):
    from src.main import get_radius

    monkeypatch.setattr("builtins.input", lambda _: "10000")
    assert get_radius() == 10000


def test_get_threshold(monkeypatch):
    from src.main import get_threshold

    monkeypatch.setattr("builtins.input", lambda _: "100")
    assert get_threshold() == 100


def test_detect_slime_chunk():
    from src.main import detect_slime_chunk

    real_detected_chunks = np.zeros((5, 5), dtype=bool)
    real_detected_chunks[0, 2] = True
    real_detected_chunks[4, 4] = True

    detected_chunks = detect_slime_chunk(0, 2)

    print("Detected Chunks:")
    print(detected_chunks)

    print("Expected Chunks:")
    print(real_detected_chunks)

    assert np.array_equal(detected_chunks, real_detected_chunks)


@patch("src.main.DEBUG", True)
@patch("random.randint")
def test_run(mock_randint, capsys):
    from src.main import run

    mock_randint.return_value = 0
    run(0, 0)
    captured = capsys.readouterr()
    assert captured.out == "史莱姆区块数: 12, 种子: 0, 挂机点区块位置: (0, 0)\n"