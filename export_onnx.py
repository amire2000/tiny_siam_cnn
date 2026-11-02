"""Export the trained patch encoder and verify numerical equivalence."""
import argparse
import copy
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from demo import TinySiamese


def export_model(checkpoint, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    torch.set_num_threads(4)
    model = TinySiamese().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    original = copy.deepcopy(model)
    # For 6→5 bins adaptive pooling is exactly 2×2 average pooling, stride 1.
    model.encoder[6] = nn.AvgPool2d(2, stride=1)
    example = torch.zeros(2, 1, 21, 21)
    with tempfile.TemporaryDirectory() as folder:
        temporary = Path(folder) / "model.onnx"
        torch.onnx.export(model, (example,), str(temporary), dynamo=True,
                          opset_version=18, external_data=False,
                          input_names=["patches"], output_names=["descriptors"],
                          dynamic_shapes={"patches": {0: torch.export.Dim("batch", min=1)}})
        onnx.checker.check_model(onnx.load(temporary))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(7)
        for count in (1, 70, 2250):
            patches = rng.random((count, 1, 21, 21), dtype=np.float32)
            patches[0] = 0.5
            with torch.inference_mode():
                expected = original(torch.from_numpy(patches)).numpy()
            actual = session.run(["descriptors"], {"patches": patches})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            print(f"Batch {count}: max descriptor difference {np.abs(actual - expected).max():.3g}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as target:
            target.write(temporary.read_bytes())
    print(f"Validated ONNX model: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/demo/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/demo/model.onnx"))
    args = parser.parse_args()
    try:
        export_model(args.checkpoint, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
