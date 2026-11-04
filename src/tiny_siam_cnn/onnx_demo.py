"""Track the saved synthetic video using ONNX Runtime, without importing PyTorch."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .tracking_io import PATCH, SOURCE, POINTS, export


def bilinear(images, x, y):
    """Sample a batch with border padding and float32 bilinear interpolation."""
    x = np.clip(x, 0, images.shape[2] - 1)
    y = np.clip(y, 0, images.shape[1] - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, images.shape[2] - 1), np.minimum(y0 + 1, images.shape[1] - 1)
    dx, dy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    batch = np.arange(len(images))[:, None, None]
    return ((images[batch, y0, x0] * (1 - dx) + images[batch, y0, x1] * dx) * (1 - dy)
            + (images[batch, y1, x0] * (1 - dx) + images[batch, y1, x1] * dx) * dy)


def extract(frame, centers, size):
    offsets = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy = np.meshgrid(offsets, offsets)
    return bilinear(frame[None], centers[:, 0, None, None] + xx,
                    centers[:, 1, None, None] + yy)[:, None]


def rotate_crop(patches, degrees):
    angles = np.deg2rad(degrees)[:, None, None]
    offsets = np.arange(PATCH, dtype=np.float32) - (PATCH - 1) / 2
    xx, yy = np.meshgrid(offsets, offsets)
    x = np.cos(angles) * xx - np.sin(angles) * yy + (SOURCE - 1) / 2
    y = np.sin(angles) * xx + np.cos(angles) * yy + (SOURCE - 1) / 2
    return bilinear(patches[:, 0], x, y)[:, None]


def create_session(model, device):
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if device == "cuda":
        ort.preload_dlls(directory="")
    if provider not in ort.get_available_providers():
        raise ValueError(f"{provider} unavailable; check CUDA dependencies or use --device cpu")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model), sess_options=options, providers=[provider])
    if session.get_providers()[0] != provider:
        raise ValueError(f"Failed to activate {provider}; check CUDA libraries and NVIDIA driver")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "patches" or inputs[0].type != "tensor(float)"
            or len(inputs[0].shape) != 4 or inputs[0].shape[1:] != [1, PATCH, PATCH]
            or isinstance(inputs[0].shape[0], int) or len(outputs) != 1
            or outputs[0].name != "descriptors" or outputs[0].type != "tensor(float)"
            or len(outputs[0].shape) != 2 or outputs[0].shape[1] != 64):
        raise ValueError("Expected dynamic float32 patches [N,1,21,21] → descriptors [N,64]")
    return session


def read_inputs(video, reference):
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if len(frames) < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video must contain at least two readable frames and a positive FPS")
    height, width = frames[0].shape
    if min(height, width) < SOURCE or any(frame.shape != (height, width) for frame in frames):
        raise ValueError("Video dimensions must be constant and at least 31×31")
    truth = np.full((len(frames), POINTS, 2), np.nan, dtype=np.float32)
    initial = np.full((POINTS, 2), np.nan, dtype=np.float32)
    with reference.open(newline="") as file:
        reader = csv.DictReader(file)
        if not {"frame", "point", "x", "y", "true_x", "true_y"}.issubset(reader.fieldnames or []):
            raise ValueError("Reference CSV is missing required coordinate columns")
        for row in reader:
            frame, point = int(row["frame"]), int(row["point"]) - 1
            if not (0 <= frame < len(frames) and 0 <= point < POINTS):
                raise ValueError("Reference frame/point does not match the video")
            if not np.isnan(truth[frame, point]).all():
                raise ValueError("Duplicate reference frame/point")
            coords = np.array([float(row["true_x"]), float(row["true_y"])], dtype=np.float32)
            if not np.isfinite(coords).all():
                raise ValueError("Reference coordinates must be finite")
            truth[frame, point] = coords
            if frame == 0:
                initial[point] = [float(row["x"]), float(row["y"])]
    if not np.isfinite(truth).all() or not np.isfinite(initial).all():
        raise ValueError("Reference must contain ten points for every decoded frame")
    margin = SOURCE // 2
    if ((initial < margin).any() or (initial > [width - 1 - margin, height - 1 - margin]).any()):
        raise ValueError("Initial patches must fit inside the video frame")
    return frames, fps, initial, truth


def track(session, frames, initial):
    positions = initial.copy()
    sources = extract(frames[0].astype(np.float32) / 255, positions, SOURCE)
    angles = np.arange(-30, 31, 10, dtype=np.float32)
    patches = rotate_crop(np.repeat(sources, len(angles), axis=0), np.tile(angles, POINTS))
    bank = session.run(["descriptors"], {"patches": patches})[0].reshape(POINTS, len(angles), 64)
    xx, yy = np.meshgrid(np.arange(-7, 8, dtype=np.float32), np.arange(-7, 8, dtype=np.float32))
    offsets = np.stack((xx.ravel(), yy.ravel()), axis=1)
    paths, scores = [positions.copy()], [np.ones(POINTS, dtype=np.float32)]
    height, width = frames[0].shape
    for frame in frames[1:]:
        candidates = positions[:, None] + offsets
        candidates[..., 0] = np.clip(candidates[..., 0], PATCH // 2, width - 1 - PATCH // 2)
        candidates[..., 1] = np.clip(candidates[..., 1], PATCH // 2, height - 1 - PATCH // 2)
        patches = extract(frame.astype(np.float32) / 255, candidates.reshape(-1, 2), PATCH)
        descriptors = session.run(["descriptors"], {"patches": patches})[0].reshape(POINTS, len(offsets), 64)
        similarities = (descriptors @ bank.transpose(0, 2, 1)).max(axis=2)
        best = similarities.argmax(axis=1)
        positions = candidates[np.arange(POINTS), best]
        paths.append(positions.copy())
        scores.append(similarities[np.arange(POINTS), best])
    return np.stack(paths), np.stack(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/demo/model.onnx"))
    parser.add_argument("--video", type=Path, default=Path("outputs/demo/input.mp4"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/demo/tracks.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/onnx_demo"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Choose a new or empty output directory to preserve existing results")
        frames, fps, initial, truth = read_inputs(args.video, args.reference)
        session = create_session(args.model, args.device)
        print(f"Provider: {session.get_providers()[0]}", flush=True)
        start = time.perf_counter()
        paths, scores = track(session, frames, initial)
        seconds = time.perf_counter() - start
        args.output.mkdir(parents=True, exist_ok=True)
        export(args.output, frames, paths, truth, scores, fps)
        errors = np.linalg.norm(paths[1:] - truth[1:], axis=2)
        metrics = {"provider": session.get_providers()[0], "frames": len(frames), "fps": fps,
                   "model": str(args.model), "video": str(args.video), "reference": str(args.reference),
                   "tracking_seconds": seconds, "tracking_fps": (len(frames) - 1) / seconds,
                   "mean_error_px": float(errors.mean()), "max_error_px": float(errors.max()),
                   "within_3px_percent": float((errors <= 3).mean() * 100),
                   "per_point_mean_error_px": errors.mean(axis=0).tolist()}
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        print(f"Saved results to {args.output.resolve()}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
