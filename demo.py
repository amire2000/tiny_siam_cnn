"""A small, inspectable Siamese patch tracker. Run `uv run demo.py --help`."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tracking_io import PATCH, SOURCE, POINTS, WIDTH, HEIGHT, export


class TinySiamese(nn.Module):
    """Both Siamese branches call this same encoder, sharing all weights."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)), nn.Flatten(), nn.Linear(400, 64),
        )

    def forward(self, patches):
        patches = patches - patches.mean(dim=(2, 3), keepdim=True)
        patches = patches / patches.std(dim=(2, 3), keepdim=True).clamp_min(0.05)
        return F.normalize(self.encoder(patches), dim=1)


def texture(rng, width, height):
    """Random multiscale texture: no painted tracking markers or point IDs."""
    coarse = rng.integers(35, 220, (height // 5, width // 5), dtype=np.uint8)
    result = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(100):
        center = tuple(rng.integers([0, 0], [width, height]).tolist())
        cv2.circle(result, center, int(rng.integers(1, 6)), int(rng.integers(20, 235)), -1)
    return cv2.GaussianBlur(result, (3, 3), 0.5)


def image_tensor(gray, device):
    return torch.as_tensor(gray, device=device, dtype=torch.float32)[None, None] / 255


def extract(image, centers, size):
    """Bilinear patches centered at floating-point (x, y) image coordinates."""
    offsets = torch.arange(size, device=image.device) - (size - 1) / 2
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    grid = centers[:, None, None, :] + torch.stack((xx, yy), dim=-1)
    scale = grid.new_tensor([image.shape[3] - 1, image.shape[2] - 1])
    grid = 2 * grid / scale - 1
    return F.grid_sample(image.expand(len(centers), -1, -1, -1), grid,
                         align_corners=True, padding_mode="border")


def rotate_crop(patches, degrees):
    """Rotate a 31px source and take its 21px center without padded corners."""
    angle = degrees * math.pi / 180
    matrices = patches.new_zeros((len(patches), 2, 3))
    ratio = PATCH / SOURCE
    matrices[:, 0, 0] = matrices[:, 1, 1] = angle.cos() * ratio
    matrices[:, 0, 1] = -angle.sin() * ratio
    matrices[:, 1, 0] = angle.sin() * ratio
    grid = F.affine_grid(matrices, (len(patches), 1, PATCH, PATCH), align_corners=False)
    return F.grid_sample(patches, grid, align_corners=False)


def train(model, device, steps, seed, output):
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(16):
        gray = texture(rng, 192, 192)
        centers = torch.tensor(rng.uniform(20, 172, (64, 2)), device=device, dtype=torch.float32)
        pool.append(extract(image_tensor(gray, device), centers, SOURCE))
    pool = torch.cat(pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        # Unique source patches avoid treating duplicate positives as negatives.
        batch = pool[torch.randperm(len(pool), device=device)[:128]]
        views = []
        for _ in range(2):
            angles = torch.empty(len(batch), device=device).uniform_(-12, 12)
            view = rotate_crop(batch, angles)
            views.append((view + torch.randn_like(view) * 0.02).clamp(0, 1))
        left, right = model(views[0]), model(views[1])
        logits = left @ right.T / 0.07
        labels = torch.arange(len(batch), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append((step, loss.item()))
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Training {step:4d}/{steps}: loss={loss.item():.4f}", flush=True)
    seconds = time.perf_counter() - start
    torch.save({"model": model.state_dict(), "seed": seed, "steps": steps}, output / "model.pt")
    with (output / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])
        writer.writerows(history)
    return seconds


def scene(seed, count):
    """Render an unseen rigid object and its transforms for evaluation only."""
    rng = np.random.default_rng(seed)
    object_image = texture(rng, 180, 140)
    frames, transforms, masks = [], [], []
    for index in range(count):
        phase = 2 * math.pi * index / max(count - 1, 1)
        angle = 28 * math.sin(phase)
        center = (WIDTH / 2 + 65 * math.sin(phase), HEIGHT / 2 + 35 * math.sin(2 * phase))
        transform = cv2.getRotationMatrix2D((90, 70), angle, 1.0)
        transform[:, 2] += np.array(center) - [90, 70]
        mask = cv2.warpAffine(np.full_like(object_image, 255), transform, (WIDTH, HEIGHT))
        warped = cv2.warpAffine(object_image, transform, (WIDTH, HEIGHT))
        frame = np.full((HEIGHT, WIDTH), 24, dtype=np.uint8)
        frame[mask > 0] = warped[mask > 0]
        frames.append(frame)
        transforms.append(transform)
        masks.append(mask)
    return frames, transforms, masks[0]


def select_points(frame, mask):
    interior = cv2.erode(mask, np.ones((SOURCE + 2, SOURCE + 2), np.uint8))
    points = cv2.goodFeaturesToTrack(frame, POINTS, 0.01, 27, mask=interior)
    if points is None or len(points) != POINTS:
        raise RuntimeError("Could not select ten well-spaced textured points.")
    return points[:, 0, :]


@torch.inference_mode()
def track(model, frames, initial_points, device):
    """Only frames and initial points enter the tracker, never ground truth."""
    model.eval()
    positions = torch.tensor(initial_points, device=device)
    sources = extract(image_tensor(frames[0], device), positions, SOURCE)
    # shortcut: ±30° template bank, extend training and search for larger rotations.
    angles = torch.arange(-30, 31, 10, device=device, dtype=torch.float32)
    bank = model(rotate_crop(sources.repeat_interleave(len(angles), 0), angles.repeat(POINTS)))
    bank = bank.reshape(POINTS, len(angles), -1)
    radius = 7
    yy, xx = torch.meshgrid(torch.arange(-radius, radius + 1, device=device),
                           torch.arange(-radius, radius + 1, device=device), indexing="ij")
    offsets = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    paths = [initial_points.copy()]
    scores = [np.ones(POINTS)]
    for frame in frames[1:]:
        candidates = positions[:, None, :] + offsets[None, :, :]
        candidates[..., 0].clamp_(PATCH // 2, WIDTH - 1 - PATCH // 2)
        candidates[..., 1].clamp_(PATCH // 2, HEIGHT - 1 - PATCH // 2)
        patches = extract(image_tensor(frame, device), candidates.reshape(-1, 2), PATCH)
        descriptors = model(patches).reshape(POINTS, len(offsets), -1)
        similarities = torch.bmm(descriptors, bank.transpose(1, 2)).amax(dim=2)
        best = similarities.argmax(dim=1)
        positions = candidates[torch.arange(POINTS, device=device), best]
        paths.append(positions.cpu().numpy())
        scores.append(similarities.max(dim=1).values.cpu().numpy())
    return np.stack(paths), np.stack(scores)


def ground_truth(initial_points, transforms):
    local = cv2.transform(initial_points[None], cv2.invertAffineTransform(transforms[0]))[0]
    return np.stack([cv2.transform(local[None], matrix)[0] for matrix in transforms])



def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=positive_int, default=400, help="training optimizer steps")
    parser.add_argument("--frames", type=positive_int, default=180, help="at least 120 for the local search")
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--checkpoint", type=Path, help="reuse weights, skipping training")
    args = parser.parse_args()
    if args.frames < 120:
        parser.error("--frames must be at least 120 so motion stays inside the local search window")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check your NVIDIA driver or explicitly use --device cpu.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty, choose a new --output to preserve existing results")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    model = TinySiamese().to(device)
    training_seconds = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    else:
        training_seconds = train(model, device, args.steps, args.seed, args.output)
    frames, transforms, mask = scene(args.seed + 10000, args.frames)
    points = select_points(frames[0], mask)
    print("Tracking ten points on an unseen object…", flush=True)
    start = time.perf_counter()
    paths, scores = track(model, frames, points, device)
    tracking_seconds = time.perf_counter() - start
    truth = ground_truth(points, transforms)
    export(args.output, frames, paths, truth, scores, args.fps)
    errors = np.linalg.norm(paths[1:] - truth[1:], axis=2)
    metrics = {
        "device": str(device), "seed": args.seed, "frames": args.frames,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "model.pt",
        "training_steps": None if args.checkpoint else args.steps,
        "training_seconds": training_seconds, "tracking_seconds": tracking_seconds,
        "mean_error_px": float(errors.mean()), "max_error_px": float(errors.max()),
        "within_3px_percent": float((errors <= 3).mean() * 100),
        "per_point_mean_error_px": errors.mean(axis=0).tolist(),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved video and paths to {args.output.resolve()}")


if __name__ == "__main__":
    main()
