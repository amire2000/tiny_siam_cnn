"""Shared constants and output rendering, with no PyTorch dependency."""
import csv

import cv2
import numpy as np

PATCH = 21
SOURCE = 31
POINTS = 10
WIDTH, HEIGHT = 480, 360


def export(output, frames, paths, truth, scores, fps):
    height, width = frames[0].shape
    colors = [tuple(int(v) for v in cv2.cvtColor(
        np.uint8([[[i * 18, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for i in range(POINTS)]
    writers = []
    try:
        for name in ("input.mp4", "tracked.mp4"):
            writer = cv2.VideoWriter(str(output / name), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            writers.append(writer)
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not open the MP4 writer for {name}.")
        for index, gray in enumerate(frames):
            original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            annotated = original.copy()
            for point, color in enumerate(colors):
                trail = np.rint(paths[:index + 1, point]).astype(np.int32)
                cv2.polylines(annotated, [trail], False, color, 1, cv2.LINE_AA)
                center = tuple(trail[-1])
                cv2.circle(annotated, center, 4, color, 1, cv2.LINE_AA)
                cv2.putText(annotated, str(point + 1), (center[0] + 5, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            error = np.linalg.norm(paths[index] - truth[index], axis=1).mean()
            cv2.putText(annotated, f"Tiny Siamese CNN | frame {index + 1}/{len(frames)}", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"10 points | mean error {error:.2f} px", (12, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            writers[0].write(original)
            writers[1].write(annotated)
        if not cv2.imwrite(str(output / "paths.png"), annotated):
            raise RuntimeError("Could not write paths.png.")
    finally:
        for writer in writers:
            writer.release()
    with (output / "tracks.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "point", "x", "y", "true_x", "true_y", "error_px", "similarity"])
        for frame in range(len(frames)):
            for point in range(POINTS):
                writer.writerow([frame, point + 1, *paths[frame, point], *truth[frame, point],
                                 np.linalg.norm(paths[frame, point] - truth[frame, point]), scores[frame, point]])

