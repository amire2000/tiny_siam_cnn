"""Interactively track features selected inside a video bounding box."""
import argparse
from pathlib import Path
import cv2
import torch
from tiny_siam_cnn import TinySiamese, Tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default='/home/user/projects/nano_tracker/src/nanotracker/demo_data/car4.mp4')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    model = TinySiamese()
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True)['model'])
    capture = cv2.VideoCapture(args.video)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError('Cannot read video')
        box = cv2.selectROI('Tracker', frame, False, False)
        tracker = Tracker(model, args.device)
        tracker.initialize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), box)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            points, _, bbox = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            for x, y in points:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow('Tracker', frame)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xff in (27, ord('q')):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
