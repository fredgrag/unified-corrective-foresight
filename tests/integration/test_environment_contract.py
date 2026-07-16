from __future__ import annotations

from importlib import metadata
import shutil
import subprocess
import sys
import unittest


class EnvironmentContractTest(unittest.TestCase):
    def test_pinned_runtime_versions(self) -> None:
        import torch
        import torchcodec
        import torchvision

        self.assertEqual(sys.version_info[:2], (3, 12))
        self.assertEqual(torch.__version__.split("+")[0], "2.8.0")
        self.assertEqual(torchvision.__version__.split("+")[0], "0.23.0")
        self.assertEqual(metadata.version("torchcodec"), "0.7.0+cu128")
        self.assertEqual(metadata.version("lerobot"), "0.5.1")
        self.assertEqual(metadata.version("transformers"), "5.3.0")
        self.assertEqual(metadata.version("numpy"), "2.2.6")
        self.assertEqual(metadata.version("safetensors"), "0.7.0")
        self.assertEqual(metadata.version("PyYAML"), "6.0.3")
        self.assertIsNotNone(torchcodec)

    def test_cuda_and_video_runtime(self) -> None:
        import torch

        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 4)
        self.assertEqual(torch.version.cuda, "12.8")
        self.assertIsNotNone(shutil.which("ffmpeg"))
        result = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        )
        self.assertIn("ffmpeg version", result.stdout)

    def test_lerobot_v3_dataset_api_is_importable(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.assertTrue(callable(LeRobotDataset))


if __name__ == "__main__":
    unittest.main()
