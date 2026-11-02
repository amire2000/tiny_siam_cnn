# Quantize the ONNX Siamese encoder

Quantization reduces model storage and can speed up inference on CPUs by storing
learned weights as 8-bit integers. The original FP32 ONNX model remains the
recommended path for CUDA. Quantized ONNX models should be tested with
`CPUExecutionProvider`, because CUDA support for quantized operators depends on
the ONNX Runtime build and GPU.

This project uses ONNX Runtime's **dynamic INT8 quantization** as the smallest
useful next step. Weights are quantized when the model is created, while
activation scales are calculated during inference. It does not need a
calibration dataset. Static quantization can be added later when CPU throughput
or accuracy requires calibration.

## Install the tools

```bash
uv sync --locked
```

The existing ONNX dependencies provide the required API:

- `onnx`: validates the graph.
- `onnxruntime-gpu`: provides `onnxruntime.quantization` and CPU/CUDA execution
  providers.
- `numpy` and `opencv-python-headless`: prepare patches and run the tracker.

## Start with the FP32 model

Create the trained checkpoint and FP32 ONNX model if they do not exist:

```bash
uv run demo.py
uv run export_onnx.py
```

The input to quantization is `outputs/demo/model.onnx`. Keep the original file;
it is the GPU baseline for accuracy and speed comparisons.

## Create an INT8 model

Run the following command from the project root. It writes a new model and
refuses to overwrite an existing one:

```bash
uv run python - <<'PY'
from pathlib import Path
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

source = Path("outputs/demo/model.onnx")
target = Path("outputs/demo/model.int8.onnx")
if target.exists():
    raise FileExistsError(f"Refusing to overwrite {target}")
if not source.is_file():
    raise FileNotFoundError(source)

# ONNX Runtime's quantizer runs shape inference. The dynamo exporter stores
# intermediate value_info annotations that conflict with that inference, so
# remove only those annotations while preserving inputs, outputs, and weights.
model = onnx.load(source)
del model.graph.value_info[:]
clean_source = target.with_suffix(".quant-input.onnx")
onnx.save(model, clean_source)
quantize_dynamic(
    model_input=str(clean_source),
    model_output=str(target),
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)
onnx.checker.check_model(onnx.load(target))
clean_source.unlink()
print(f"Created and checked {target}")
PY
```

The temporary copy is a compatibility cleanup, not a model conversion: it
removes inferred intermediate annotations only. Learned initializers, graph
inputs, graph outputs, and operators are retained. The current ONNX Runtime
quantizer otherwise reports a shape-inference mismatch for the exported
`[batch,400] → [batch,64]` linear path.

`per_channel=True` gives each output channel its own weight scale, which is a
good fit for the convolution and linear layers in this small encoder. The graph
still accepts float32 input patches shaped `[N,1,21,21]` and returns float32
descriptors shaped `[N,64]`; quantization is internal to the graph.

## Run the quantized model

The current `onnx_demo.py` accepts an ONNX path, so use the INT8 model with the
CPU provider and a new output directory:

```bash
uv run onnx_demo.py \
  --model outputs/demo/model.int8.onnx \
  --video outputs/demo/input.mp4 \
  --reference outputs/demo/tracks.csv \
  --output outputs/onnx_int8 \
  --device cpu

xdg-open outputs/onnx_int8/tracked.mp4
```

The video must be the same decoded input used by the FP32 run when comparing
accuracy. The output directory contains the annotated video, paths image,
per-point CSV, and metrics JSON. The source video's playback FPS is preserved;
it is separate from inference throughput.

## Compare FP32 and INT8

Compare the two `metrics.json` files:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for name in ("onnx_demo", "onnx_int8"):
    metrics = json.loads((Path("outputs") / name / "metrics.json").read_text())
    print(name, {
        "mean_error_px": metrics["mean_error_px"],
        "max_error_px": metrics["max_error_px"],
        "tracking_fps": metrics.get("tracking_fps"),
    })
PY
```

For this educational tracker, accept the quantized model only when its mean
error stays below 1.5 pixels and within 1 pixel of the FP32 mean error. Also
decode the complete video and confirm that its frame count, dimensions, and FPS
match the FP32 output. INT8 can change a close descriptor tie by one pixel even
when the descriptor values remain visually similar.

## Dynamic versus static quantization

Dynamic quantization is convenient because it needs no representative data. It
may be slower than static INT8 on CPUs that benefit from precomputed activation
scales. Static quantization requires a calibration reader containing patches
that resemble the tracking workload, followed by calibration and accuracy
validation. Quantization-aware training is a separate training workflow and is
not needed for this tiny demonstration.

Do not assume that a smaller model is faster. Measure preprocessing, ONNX
Runtime execution, host/device transfers, matching, and video writing
separately. Keep the FP32 CUDA model for GPU use unless the selected ONNX Runtime
build demonstrates correct and faster quantized GPU execution.

## Troubleshooting

- `ModuleNotFoundError: onnxruntime.quantization`: run `uv sync --locked` and
  execute the command with `uv run`.
- Output already exists: choose another target path or remove the specific
  generated model after confirming it is disposable.
- CUDA provider errors: run the INT8 model with `--device cpu`; use the FP32
  model for the existing CUDA path.
- Accuracy changes substantially: verify that the same MP4 and reference CSV
  were used, then compare descriptor outputs before and after quantization.
- Unsupported quantized operator: try the default per-tensor setting
  (`per_channel=False`) or keep the FP32 model for that execution provider.

The synthetic scene remains the same limitation as the FP32 demo: it represents
a visible, textured, rigid object with limited rotation and no occlusion.
