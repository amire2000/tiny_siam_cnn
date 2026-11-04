"""Frame-by-frame Siamese feature tracker."""
import numpy as np
import cv2
import torch

from .demo import PATCH, SOURCE, TinySiamese, extract, image_tensor, rotate_crop


class Tracker:
    def __init__(self, model, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.positions = None

    @torch.inference_mode()
    def initialize(self, frame, bbox):
        frame = np.asarray(frame)
        if frame.ndim != 2 or min(frame.shape) < PATCH:
            raise ValueError("Expected a grayscale frame at least 21 pixels wide and high")
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            raise ValueError("Box must be inside the frame")
        mask = np.zeros(frame.shape, np.uint8)
        mask[int(y):int(y+h), int(x):int(x+w)] = 255
        points = cv2.goodFeaturesToTrack(frame, 10, 0.01, 7, mask=mask)
        if points is None:
            raise ValueError("No usable features found inside the box")
        self.shape = frame.shape
        self.positions = torch.as_tensor(points[:, 0, :], device=self.device)
        angles = torch.arange(-30, 31, 10, device=self.device, dtype=torch.float32)
        sources = extract(image_tensor(frame, self.device), self.positions, SOURCE)
        self.bank = self.model(rotate_crop(sources.repeat_interleave(len(angles), 0), angles.repeat(len(points)))).reshape(len(points), len(angles), -1)
        yy, xx = torch.meshgrid(torch.arange(-7, 8, device=self.device), torch.arange(-7, 8, device=self.device), indexing="ij")
        self.offsets = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        return self.positions.cpu().numpy().copy()

    @torch.inference_mode()
    def update(self, frame):
        if self.positions is None:
            raise RuntimeError("Initialize the tracker first")
        if np.asarray(frame).shape != self.shape:
            raise ValueError("Frame shape must match the initial frame")
        candidates = self.positions[:, None, :] + self.offsets
        candidates[..., 0].clamp_(PATCH // 2, self.shape[1] - 1 - PATCH // 2)
        candidates[..., 1].clamp_(PATCH // 2, self.shape[0] - 1 - PATCH // 2)
        patches = extract(image_tensor(frame, self.device), candidates.reshape(-1, 2), PATCH)
        descriptors = self.model(patches).reshape(len(self.positions), len(self.offsets), -1)
        similarities = torch.bmm(descriptors, self.bank.transpose(1, 2)).amax(dim=2)
        best = similarities.argmax(dim=1)
        self.positions = candidates[torch.arange(len(self.positions), device=self.device), best]
        points = self.positions.cpu().numpy().copy()
        low, high = points.min(axis=0), points.max(axis=0)
        return points, similarities.max(dim=1).values.cpu().numpy(), (float(low[0]), float(low[1]), float(high[0]-low[0]), float(high[1]-low[1]))
