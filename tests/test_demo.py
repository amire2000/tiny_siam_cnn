"""Fast CPU checks; the full CUDA run is the end-to-end quality check."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from tiny_siam_cnn.demo import (PATCH, SOURCE, TinySiamese, extract, export, ground_truth,
                  rotate_crop, scene, select_points, track)


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patch_coordinates_and_rotation_center(self):
        image = torch.arange(81, dtype=torch.float32).reshape(1, 1, 9, 9)
        patch = extract(image, torch.tensor([[4., 4.]]), 3)
        torch.testing.assert_close(patch[0, 0], image[0, 0, 3:6, 3:6])
        source = torch.rand(2, 1, SOURCE, SOURCE)
        rotated = rotate_crop(source, torch.zeros(2))
        margin = (SOURCE - PATCH) // 2
        torch.testing.assert_close(rotated, source[:, :, margin:-margin, margin:-margin],
                                   atol=2e-6, rtol=1e-5)

    def test_scene_ground_truth_and_point_selection(self):
        frames, transforms, mask = scene(10007, 120)
        points = select_points(frames[0], mask)
        truth = ground_truth(points, transforms)
        self.assertEqual(truth.shape, (120, 10, 2))
        np.testing.assert_allclose(truth[0], points, atol=1e-5)
        np.testing.assert_allclose(truth[-1], points, atol=1e-5)
        distances = np.linalg.norm(truth[:, 0] - truth[:, 1], axis=1)
        np.testing.assert_allclose(distances, distances[0], atol=1e-4)
        self.assertLess(np.abs(np.diff(truth, axis=0)).max(), 7)

    def test_training_gradients_and_known_translation(self):
        torch.manual_seed(7)
        model = TinySiamese()
        descriptors = model(torch.rand(8, 1, PATCH, PATCH))
        torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(8))
        (-descriptors[0] @ descriptors[1]).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        frames, _, mask = scene(10007, 120)
        points = select_points(frames[0], mask)
        shifted = cv2.warpAffine(frames[0], np.float32([[1, 0, 3], [0, 1, -2]]), (480, 360))
        paths, scores = track(model, [frames[0], shifted], points, torch.device('cpu'))
        np.testing.assert_allclose(paths[1], points + [3, -2], atol=0.01)
        self.assertTrue(np.isfinite(scores).all())
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            export(output, [frames[0], shifted], paths, paths, scores, 30)
            capture = cv2.VideoCapture(str(output / 'tracked.mp4'))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            capture.release()
            self.assertTrue((output / 'paths.png').is_file())
            self.assertEqual(len((output / 'tracks.csv').read_text().splitlines()), 21)


if __name__ == '__main__':
    unittest.main()
