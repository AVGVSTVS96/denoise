"""Turn a fine-tuned checkpoint into the model file the app loads.

    uv run ship.py data/models/finetune/step6000.pth

Same steps as Hush's export (tools/models, Apache-2.0, by ms0242808), which made the model the
app shipped with: bound every block's channel attention to the range it spans on real photos, so
JPEG shadows can't run it away into stripes, export to ONNX, and convert to fp16 with the layer
norms and pools left in fp32. The ranges are measured again here, on SIDD and on the training
crops, because fine-tuning moves them. Writes data/models/nafnet-w32-hiso-fp16.onnx.
"""

import pathlib
import sys

import cv2
import export
import numpy as np
import sidd
import torch
from nafnet import blocks, bound_attention

import evaluate

OUT = evaluate.MODELS / "nafnet-w32-hiso"


def attention_ranges(model):
    low, high = {}, {}

    def record(name):
        def hook(_module, _inputs, output):
            weights = output.reshape(output.shape[1]).numpy()
            low[name] = np.minimum(low.get(name, weights), weights)
            high[name] = np.maximum(high.get(name, weights), weights)

        return hook

    hooks = [block.sca.register_forward_hook(record(name)) for name, block in blocks(model)]
    photos = [noisy for _, noisy, _ in sidd.pairs(every=4)]
    photos += [
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for p in sorted((evaluate.DATA / "train").glob("*_iso*.png"))[::6]
    ]
    with torch.no_grad():
        for photo in photos:
            model(evaluate.to_tensor(photo))
    for hook in hooks:
        hook.remove()
    return {name: (low[name].tolist(), high[name].tolist()) for name in low}


def main(checkpoint):
    model = evaluate.nafnet(32)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("params", state))
    model.eval()
    bound_attention(model, attention_ranges(model), export.ATTENTION_MARGIN)
    fp32, fp16 = pathlib.Path(f"{OUT}-fp32.onnx"), pathlib.Path(f"{OUT}-fp16.onnx")
    export.export_fp32(model, fp32)
    export.convert_fp16(fp32, fp16)
    print(export.audit_fp16_ranges(fp32, fp16), export.audit_shadow(fp16), export.check_against_torch(model, fp16))
    fp32.unlink()
    print(fp16, fp16.stat().st_size, export.sha256_file(fp16))


if __name__ == "__main__":
    main(sys.argv[1])
