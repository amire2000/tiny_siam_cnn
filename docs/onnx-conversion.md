# Convert the tiny Siamese CNN to ONNX

ONNX stores a trained network's computation graph and weights in a portable file.
Here we export the **shared patch encoder**, not the complete tracker. Both
Siamese branches use the same encoder, so only one copy is needed. Patch
extraction, template rotation, candidate search, and rendering remain Python
code. Training still uses PyTorch; `onnx_demo.py` never imports it.

## 1. Install packages

From the project directory:

```bash
uv sync --locked
```

| Package | Role |
| --- | --- |
| `torch` | Train, load the checkpoint, and export the network |
| `onnx` | Read and validate the exported graph |
| `onnxscript` | Operator translation used by the modern PyTorch exporter |
| `onnxruntime-gpu` | Execute the exported model on CUDA or explicitly on CPU |
| `numpy` | Patch preparation, interpolation, similarity matching, metrics |
| `opencv-python-headless` | Decode/encode video and draw tracking trails |

Do not install `onnxruntime` alongside `onnxruntime-gpu`: the GPU package already
includes the CPU execution provider. `uv.lock` pins the tested dependencies.
The project environment includes PyTorch for training/export, but the ONNX
runtime path has no PyTorch imports. It uses NVIDIA CUDA libraries installed
in the environment through the existing GPU dependencies.

Tested versions: Python 3.12, PyTorch 2.11.0+cu128, ONNX 1.23.2,
ONNX Script 0.7.2, ONNX Runtime GPU 1.31.1. GPU: RTX 5070 Laptop GPU.

## 2. Obtain the trained weights and source video

The existing `outputs/demo/` contains `model.pt`, `input.mp4`, and `tracks.csv`.
Reuse those files; no retraining is required. On a fresh checkout, create them:

```bash
uv run examples/demo.py
```

This trains the network and renders the 180-frame, 30 FPS source video. The
CSV supplies the ten frame-zero positions and ground truth for evaluation.
The tracker never receives the later ground-truth coordinates.

## 3. Export and validate

```bash
uv run examples/export_onnx.py
```

Defaults are `outputs/demo/model.pt` → `outputs/demo/model.onnx`. For other paths:

```bash
uv run examples/export_onnx.py --checkpoint outputs/demo/model.pt --output outputs/export2/model.onnx
```

The exporter refuses to overwrite a model. It writes the final file only after
validation succeeds. Weights are embedded in one `.onnx` file, without a separate
external-data file.

### What the exporter does

1. Construct `TinySiamese`, load the state dictionary using `weights_only=True`,
   and call `eval()` to select inference behavior. Export runs on CPU.
2. Replace adaptive average pooling in the export copy with `AvgPool2d(2, stride=1)`.
   For this model's **6×6 input and 5×5 output**, each adaptive bin is a 2×2 region
   with stride 1, so this is equivalent. This replacement is specific to the
   fixed 21×21 input size and changes neither parameters nor the original model.
3. Supply a float32 example tensor shaped `[2, 1, 21, 21]` to capture the graph.
   Export with `dynamo=True`, **ONNX opset 18**, named inputs/outputs, and a dynamic
   batch dimension using `torch.export.Dim`. Opset 18 identifies the ONNX operator
   definitions used by this model; it is not the installed ONNX package version.
4. Preserve per-patch mean subtraction, sample standard deviation with its 0.05
   floor, learned convolutions/linear projection, and final L2 normalization.
5. Run `onnx.checker.check_model`, then compare ONNX Runtime outputs against
   PyTorch with `atol=1e-5, rtol=1e-4` for batches of 1, 70, and 2,250. Each batch
   includes a constant patch to check the normalization floor.

| Tensor | Name | Type and shape | Meaning |
| --- | --- | --- | --- |
| Input | `patches` | float32 `[N,1,21,21]` | Grayscale patches scaled to `[0,1]` |
| Output | `descriptors` | float32 `[N,64]` | Unit-length patch descriptors |

Only `N` is dynamic. Seventy patches represent ten points × seven rotated
templates; 2,250 represent ten points × 225 candidate locations. Do not normalize
patches outside the graph beyond dividing image intensities by 255.

For the saved trained model, the largest descriptor difference during export
validation was **5.36e-7**. Tests additionally compare the original adaptive-pooling
model against ONNX, so the pooling replacement is checked independently.

## 4. Run ONNX tracking on the same video

```bash
uv run examples/onnx_demo.py
xdg-open outputs/onnx_demo/tracked.mp4
```

The default CUDA run reads `outputs/demo/input.mp4`, takes initial points from
`outputs/demo/tracks.csv`, and loads `outputs/demo/model.onnx`. It preserves the
source frame count, dimensions, and playback FPS. Full arguments:

```bash
uv run examples/onnx_demo.py --model outputs/demo/model.onnx --video outputs/demo/input.mp4 --reference outputs/demo/tracks.csv --output outputs/onnx_run2 --device cuda
```

CPU execution is explicit:

```bash
uv run examples/onnx_demo.py --device cpu --output outputs/onnx_cpu
```

Choose a new output directory each time. Outputs include `tracked.mp4`,
`paths.png`, `tracks.csv`, and `metrics.json`. The shared renderer also produces
an `input.mp4` containing the decoded input frames re-encoded for inspection;
the source video is never modified. Scores are cosine similarities, not
calibrated confidence probabilities.

The runtime uses NumPy float32 bilinear sampling with the same patch centers and
rotation convention as PyTorch. It retains the original seven rotated templates,
15×15 local search, and independent matching of all ten points. The entire demo
is checked to run without importing PyTorch.

## 5. CUDA verification and measured behavior

Before creating a session, the runtime calls `onnxruntime.preload_dlls(directory="")`
to load CUDA/cuDNN libraries from NVIDIA packages without importing PyTorch.
It requests `CUDAExecutionProvider` and verifies that it was activated. It fails
rather than silently switching the entire session to CPU. Some shape operations
may still run on CPU; profiling confirmed the CNN convolutions execute on CUDA.

The saved-video CUDA run produced **0.394 px mean error**, **0.860 px maximum
error**, and **100% of tracked points within 3 px**. It processed 179 tracking
updates in **11.44 seconds**, approximately **15.65 tracking FPS**. This timing
includes template encoding, CPU preprocessing, host/device transfers, and
matching, but excludes video decoding, session creation, and rendering.

The output video still plays at **30 FPS**. Playback FPS is not processing speed.
This standalone implementation was slower than the original PyTorch pipeline:
patch preparation and matching now run on CPU, and every CNN call transfers
arrays between host and GPU. ONNX is a portability format, not a promise of speed.

The MP4 is lossy. For a fair accuracy comparison, both trackers were run on the
**same decoded MP4 frames**, using the same initial points. Do not compare a
compressed-input ONNX run to the original uncompressed PyTorch frames and attribute
all differences to conversion. The measured comparison is saved in
`outputs/onnx_demo/comparison.json` for this workspace. PyTorch measured
0.394070 px mean error versus ONNX's 0.393945 px; the maximum difference
between their predicted positions was 1 pixel. Near-tied candidate scores can
select neighboring pixels despite close descriptor agreement.

## Checks and troubleshooting

```bash
uv run python -m unittest discover -s tests -v
```

Tests cover descriptor equivalence, the pooling replacement, bilinear sampling,
known translation, malformed references, output protection, and existing demo
behavior. End-to-end acceptance requires mean tracking error below 1 px and within
0.1 px of PyTorch on identical decoded frames.

- **Missing packages:** run `uv sync --locked` and use `uv run`, not a different
  system Python environment.
- **CUDA provider fails:** check `nvidia-smi`, the NVIDIA driver, and the CUDA/cuDNN
  library errors printed by ONNX Runtime. Use `--device cpu` to run explicitly on
  CPU while resolving GPU setup.
- **Shape/type mismatch:** use contiguous float32 `[N,1,21,21]` input. This model
  does not accept RGB images, arbitrary patch dimensions, or uint8 arrays.
- **Reference mismatch:** use the CSV associated with the source video. It must
  contain one row per point per decoded frame, with finite coordinates.
- **Output already exists:** choose another `--output`; existing files are protected.
- **Exporter mentions missing torchvision operators:** those operators are unused
  by this CNN. Successful graph validation and parity checks establish export success.
- **Video will not play in a browser:** open it in a desktop player supporting
  OpenCV's `mp4v` codec.

This remains a synthetic rigid-object example with limited rotation/motion and
no occlusion handling. Quantization, TensorRT, and live preview are not included.

References: [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html),
[ONNX Runtime CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
