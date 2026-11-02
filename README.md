# Tiny Siamese CNN: track 10 points

An educational Python/uv project that trains a small Siamese CNN from scratch,
then tracks **10 automatically selected points on one textured rigid object**.
The object translates and rotates. Every point has a numbered marker and its own
colored path in the output video. No dataset or pretrained weights are needed.

## Install the required packages

Use Linux x86-64 with an NVIDIA GPU and a driver compatible with CUDA 12.8.
This project uses Python 3.12. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
if it is not already installed, then run from this directory:

```bash
uv sync --locked
uv run python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU detected')"
```

`uv sync` creates `.venv` and installs everything in `uv.lock`. You do not need to
activate the environment when using `uv run`.

| Direct package | Purpose |
| --- | --- |
| `torch` (CUDA 12.8 build) | CNN, automatic differentiation, optimizer, GPU patch matching |
| `numpy` | Synthetic images, trajectories, error measurements |
| `opencv-python-headless` | Texture generation, corner selection, image transforms, MP4/PNG export |
| `onnx`, `onnxscript` | ONNX export and graph validation |
| `onnxruntime-gpu` | Standalone ONNX inference on CUDA or CPU |

Python's standard library provides the CLI, CSV/JSON writing, and tests. There is
no need for torchvision, matplotlib, a GUI toolkit, or a separate video encoder.
The CUDA runtime dependencies come through PyTorch; this demo compiles no CUDA
code. A working NVIDIA driver is required. The initial install downloads several
GB of GPU packages.

The PyTorch CUDA index is explicitly assigned only to `torch`, following the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
See also [PyTorch installation guidance](https://pytorch.org/get-started/locally/).

## Run the complete demo

```bash
uv run demo.py
```

This trains on CUDA, generates a separate 180-frame sequence (6 seconds at 30 fps),
tracks the points, and writes `outputs/demo/`. CUDA is required by default:
the program reports an error rather than silently switching to CPU.

| Output | Contents |
| --- | --- |
| `input.mp4` | Generated moving object without tracking overlays |
| `tracked.mp4` | Ten numbered points and their colored trails |
| `paths.png` | Last annotated frame showing all complete paths |
| `model.pt` | Trained weights and training seed/step count |
| `training.csv` | Loss at every training step |
| `tracks.csv` | Frame, point ID, predicted x/y, true x/y, error, cosine similarity |
| `metrics.json` | Training/tracking time, parameter count, aggregate and per-point error |

Open the videos in a desktop video player. The program runs headlessly and does
not open a window. MP4 uses OpenCV's `mp4v` codec, which some browsers do not play.
Existing nonempty output directories are rejected to protect previous runs.

Reuse saved weights without retraining:

```bash
uv run demo.py --checkpoint outputs/demo/model.pt --output outputs/reused
```

Try a different unseen object, or train for longer:

```bash
uv run demo.py --checkpoint outputs/demo/model.pt --seed 42 --output outputs/new-object
uv run demo.py --steps 800 --output outputs/longer-training
```

An explicit CPU option is available for debugging (using the same installed
PyTorch build): `uv run demo.py --device cpu --output outputs/cpu`.
Use `uv run demo.py --help` for all arguments.

## Training: what the network learns

A Siamese network uses **one encoder with shared weights** for both image patches.
It learns descriptors that are similar for two views of the same location and
different for other locations. It does not predict coordinates directly.

1. Generate 16 random textured images and sample 64 source patches from each.
   These 1,024 patches form a small training pool, independent of the demo object.
2. For each training step, choose 128 distinct source patches. Make two views of
   each: rotate independently by up to 12 degrees, crop to 21×21, and add noise.
3. Pass both batches through the same CNN. Each patch becomes a normalized
   64-dimensional descriptor.
4. Compute a 128×128 matrix of cosine similarities, divided by temperature 0.07.
   The diagonal contains matching pairs; off-diagonal entries are negatives.
5. Minimize cross-entropy in both directions, so each patch recognizes its paired
   view among the 128 candidates. Adam updates the weights at learning rate 0.001.

The default is **400 optimizer steps**, batch size 128, with seed 7. Steps are
updates, not epochs: views are newly augmented each time. `training.csv` records
training loss, not a held-out validation loss. The separate tracking sequence is
an evaluation of localization on an unseen object. Increasing steps does not
guarantee better tracking, so compare `metrics.json` as well as the loss.

The encoder is deliberately small:

```text
21×21 grayscale patch
  → per-patch mean/std normalization
  → Conv 1→8, 3×3, stride 1 + ReLU
  → Conv 8→16, 3×3, stride 2 + ReLU
  → Conv 16→16, 3×3, stride 2 + ReLU
  → adaptive average pooling to 5×5
  → flatten → Linear 400→64
  → L2 normalization
```

Spatial pooling retains some patch layout for localization. The model has about
29,000 trainable parameters. CUDA runs the encoder, loss, optimizer, and matching;
NumPy/OpenCV generate scenes and encode video on the CPU.

## How tracking works

OpenCV selects ten well-spaced corners inside the object's first-frame mask. The
mask is supplied by the generator **only for initialization**. The tracker then
receives only the images and those ten initial coordinates.

For each point, the CNN encodes a bank of seven initial patches rotated from
−30° to +30°. On every subsequent frame, it encodes 225 candidate patches in a
15×15 search window around the previous estimated position. The candidate with
the highest cosine similarity to any of that point's templates becomes its new
position. Templates stay fixed, avoiding repeated template updates that could
accumulate appearance drift. No rigid-motion solver or ground-truth coordinates
correct the predicted positions.

The generator separately transforms the original points to obtain ground truth.
Errors are Euclidean distances in pixels. Aggregate metrics exclude frame zero,
whose positions are given to the tracker. Similarity is **not** a calibrated
confidence probability.

## Limits

This is a synthetic teaching example, not a general-purpose real-video tracker.
It assumes a fully visible, textured rigid object, rotation within ±28°, no scale
change, no occlusion, and small frame-to-frame motion. Search is limited to ±7
pixels per coordinate; sequences require at least 120 frames to keep the generated
motion within that window. The ten points are matched independently and can fail
on ambiguous textures. There is no lost-point detection or reinitialization.
The first-frame mask would need another source in a real-video application.

Seeds make the generated data repeatable; exact results may differ across
PyTorch versions and devices. A single synthetic sequence is not evidence of
real-world tracking quality.

## Code and checks

`demo.py` keeps generation, training, and tracking in separate functions.
`tracking_io.py` shares constants and rendering with the standalone ONNX demo.

```bash
uv run python -m unittest discover -s tests -v
```

The CPU tests check patch coordinates, rotation/crop alignment, rigid ground
truth, finite training gradients, exact tracking of a known translation, and
readable video/CSV output. Run the default demo for the full trained CUDA check.

## Measured example

Verified on this machine with an **NVIDIA GeForce RTX 5070 Laptop GPU** and
PyTorch **2.11.0+cu128**:

| Run | Mean error | Maximum error | Points within 3 px |
| --- | --- | --- | --- |
| Default seed 7, 180 frames | 0.393 px | 0.926 px | 100% |
| Reused weights, seed 42, 120 frames | 0.393 px | 0.886 px | 100% |

The default run took 3.51 seconds for the training loop and 1.54 seconds for
tracking. These timings exclude package installation, process/CUDA startup,
training-pool generation, and video export. All three CPU tests passed, and both
default videos were decoded successfully through all 180 frames.

![Ten colored tracking paths](outputs/demo/paths.png)

## ONNX version

See the [step-by-step ONNX conversion guide](docs/onnx-conversion.md) for the
package list, export internals, validation, CUDA setup, and measured results.
Reuse the trained weights and the exact saved input video:

```bash
uv sync --locked
uv run export_onnx.py
uv run onnx_demo.py
xdg-open outputs/onnx_demo/tracked.mp4
```

The exporter uses PyTorch; `onnx_demo.py` uses ONNX Runtime with CUDA plus
NumPy/OpenCV, with no PyTorch imports. Export refuses to overwrite an existing
model; the demo requires an empty/new output directory. Use `--help` for paths
and explicit CPU mode. ONNX tracking keeps the video's 30 FPS playback rate,
but processing throughput is measured separately and is not necessarily faster.

For the INT8 workflow, see the [ONNX quantization guide](docs/onnx-quantization.md).
It explains dynamic quantization, CPU execution, accuracy checks, and how to
compare the quantized model with the FP32 CUDA baseline.
