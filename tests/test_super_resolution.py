import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.super_resolution import SuperResolutionConfig, SuperResolutionEngine
from models.transformer.sr_transformer import SRTransformer


class SuperResolutionTests(unittest.TestCase):
    def test_sr_transformer_respects_scale_and_odd_sizes(self):
        model = SRTransformer(dim=12, depth=1, num_heads=3, scale_factor=2, window_size=8)
        model.eval()
        x = torch.rand(1, 3, 15, 17)
        with torch.inference_mode():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 3, 30, 34))

    def test_engine_bicubic_fallback_without_model(self):
        frame = np.zeros((17, 19, 3), dtype=np.uint8)
        frame[:, :, 0] = 128
        engine = SuperResolutionEngine(
            SuperResolutionConfig(backend="bicubic", scale_factor=2, device="cpu")
        )
        output, stats = engine.upscale_with_stats(frame)
        self.assertEqual(output.shape, (34, 38, 3))
        self.assertEqual(stats.backend, "bicubic")
        self.assertEqual(stats.output_resolution, (38, 34))
        self.assertGreaterEqual(stats.latency_ms, 0.0)

    def test_target_resolution_resize(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        engine = SuperResolutionEngine(
            SuperResolutionConfig(backend="bilinear", scale_factor=2, device="cpu")
        )
        output = engine.upscale_frame(frame, target_resolution=(40, 30))
        self.assertEqual(output.shape, (30, 40, 3))


if __name__ == "__main__":
    unittest.main()
