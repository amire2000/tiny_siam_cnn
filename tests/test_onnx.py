"""Export and preprocessing parity checks, runnable without a GPU."""
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import cv2
import numpy as np
import torch

from tiny_siam_cnn import demo
from tiny_siam_cnn import onnx_demo
from tiny_siam_cnn.export_onnx import export_model


class OnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        torch.manual_seed(7)
        cls.model = demo.TinySiamese().eval()
        checkpoint = cls.root / 'model.pt'
        torch.save({'model': cls.model.state_dict()}, checkpoint)
        export_model(checkpoint, cls.root / 'model.onnx')
        cls.session = onnx_demo.create_session(cls.root / 'model.onnx', 'cpu')

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_encoder_and_pooling_parity(self):
        for count in (1, 70, 2250):
            patches = torch.rand(count, 1, 21, 21)
            patches[0] = 0.5
            with torch.inference_mode():
                expected = self.model(patches).numpy()
            actual = self.session.run(None, {'patches': patches.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

    def test_sampling_parity(self):
        rng = np.random.default_rng(3)
        frame = rng.random((61, 81), dtype=np.float32)
        centers = np.array([[35.25, 29.75], [1.2, 2.3]], dtype=np.float32)
        expected = demo.extract(torch.from_numpy(frame)[None, None], torch.from_numpy(centers), 31).numpy()
        actual = onnx_demo.extract(frame, centers, 31)
        np.testing.assert_allclose(actual, expected, atol=6e-6)
        angles = np.array([-30, 12], dtype=np.float32)
        rotated = onnx_demo.rotate_crop(actual, angles)
        expected = demo.rotate_crop(torch.from_numpy(actual), torch.from_numpy(angles)).numpy()
        np.testing.assert_allclose(rotated, expected, atol=6e-6)

    def test_known_translation(self):
        frames, _, mask = demo.scene(10007, 120)
        initial = demo.select_points(frames[0], mask)
        shifted = cv2.warpAffine(frames[0], np.float32([[1, 0, 3], [0, 1, -2]]), (480, 360))
        paths, _ = onnx_demo.track(self.session, [frames[0], shifted], initial)
        np.testing.assert_array_equal(paths[1], initial + [3, -2])

    def test_runtime_without_torch(self):
        code = """
import sys, importlib.abc
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('Runtime imported PyTorch')
sys.meta_path.insert(0, BlockTorch())
from tiny_siam_cnn import onnx_demo
import numpy as np
session = onnx_demo.create_session(sys.argv[1], 'cpu')
result = session.run(None, {'patches': np.zeros((1,1,21,21), dtype=np.float32)})
assert result[0].shape == (1,64)
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', code, str(self.root / 'model.onnx')], check=True)

    def test_invalid_reference_and_overwrite(self):
        frames, _, mask = demo.scene(10007, 120)
        initial = demo.select_points(frames[0], mask)
        paths = np.stack([initial, initial])
        demo.export(self.root, frames[:2], paths, paths, np.ones((2, 10)), 30)
        _, fps, actual, _ = onnx_demo.read_inputs(self.root / 'input.mp4', self.root / 'tracks.csv')
        self.assertEqual(fps, 30)
        np.testing.assert_array_equal(actual, initial)
        reference = self.root / 'tracks.csv'
        reference.write_text(reference.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'ten points'):
            onnx_demo.read_inputs(self.root / 'input.mp4', reference)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            export_model(self.root / 'model.pt', self.root / 'model.onnx')


if __name__ == '__main__':
    unittest.main()
