from __future__ import annotations

import torch

from magnetic_model import build_model


def test_dummy_checkpoint_round_trip_and_forward(tmp_path) -> None:
    torch.manual_seed(7)
    source = build_model("A").eval()
    checkpoint_path = tmp_path / "dummy_A.pth"
    torch.save(
        {
            "model": source.state_dict(),
            "norm": {
                "B_mean": 0.0,
                "B_std": 1.0,
                "dB_mean": 0.0,
                "dB_std": 1.0,
                "T_mean": 25.0,
                "T_std": 1.0,
                "H_mean": 0.0,
                "H_std": 1.0,
            },
            "seq": 0.0,
            "ene": 0.0,
            "epoch": 0,
        },
        checkpoint_path,
    )

    loaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = build_model("A").eval()
    restored.load_state_dict(loaded_payload["model"], strict=True)

    inputs = torch.zeros((1, 1_000, 5), dtype=torch.float32)
    with torch.inference_mode():
        expected = source(inputs)
        actual = restored(inputs)

    assert actual.shape == (1, 1_000)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
