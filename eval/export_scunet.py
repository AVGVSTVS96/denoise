"""Export SCUNet (color, real noise, PSNR weights) to ONNX for the browser: fp32, then fp16.

    uv run export_scunet.py

The graph is fixed at one 512 px tile: SCUNet's window attention builds its masks from the
input size. Prints how far each file is from PyTorch on a real noisy tile.
"""

import contextlib
import io
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16

import evaluate

OUT = evaluate.MODELS
TILE = evaluate.TILE


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        model = evaluate.scunet("psnr").eval()
    _, _, noisy, _ = next(evaluate.nind())
    tile = evaluate.to_tensor(noisy)
    with torch.no_grad():
        expected = model(tile).numpy()

    fp32 = OUT / "scunet-psnr-fp32.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (torch.rand(1, 3, TILE, TILE),),
            str(fp32),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        fp16 = OUT / "scunet-psnr-fp16.onnx"
        onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))

    for path in (fp32, fp16):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (actual,) = session.run(None, {"input": tile.numpy()})
        ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
        print(
            path.name,
            path.stat().st_size >> 20,
            "MB",
            "max diff vs PyTorch",
            float(np.abs(actual - expected).max()),
            ops if path == fp32 else "",
        )


if __name__ == "__main__":
    main()
