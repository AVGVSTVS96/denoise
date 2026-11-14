"""RealRestorer (StepFun) on his crops on a 32 GB Apple Silicon Mac.

The 12B transformer and the 7B Qwen2.5-VL encoder don't fit together, so this runs in two passes:
encode every crop's prompt with the encoder alone, free it, then denoise with the transformer quantized to int8.

    python realrestorer_mac.py encode <weights> <in-dir> <emb-dir>
    python realrestorer_mac.py restore <weights> <in-dir> <emb-dir> <out-dir>
"""

import gc
import pathlib
import sys

import torch
from diffusers import RealRestorerPipeline
from PIL import Image

PROMPT = "Please remove noise from the image."
mode, weights, inp, emb = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
crops = sorted(inp.glob("*.png"))
emb.mkdir(exist_ok=True)

if mode == "encode":
    pipe = RealRestorerPipeline.from_pretrained(weights, transformer=None, torch_dtype=torch.bfloat16).to("mps")
    for f in crops:
        resized, _ = pipe._resize_image(Image.open(f).convert("RGB"), img_size=1024)
        ref = pipe.load_image(resized).to(device="mps", dtype=torch.float32)
        embeds, mask = pipe._encode_prompt(PROMPT, "", ref, edit_type=1)
        torch.save({"embeds": embeds.cpu(), "mask": mask.cpu()}, emb / f"{f.stem}.pt")
        print("encoded", f.name, flush=True)
else:
    from optimum.quanto import freeze, qint8, quantize

    out = pathlib.Path(sys.argv[5])
    out.mkdir(exist_ok=True)
    pipe = RealRestorerPipeline.from_pretrained(weights, text_encoder=None, torch_dtype=torch.bfloat16)
    quantize(pipe.transformer, weights=qint8)
    freeze(pipe.transformer)
    gc.collect()
    pipe.to("mps")
    for f in crops:
        saved = torch.load(emb / f"{f.stem}.pt")
        pipe._encode_prompt = lambda *a, **k: (saved["embeds"].to("mps"), saved["mask"].to("mps"))
        image = pipe(image=Image.open(f).convert("RGB"), prompt=PROMPT, negative_prompt="",
                     num_inference_steps=28, guidance_scale=3.0, seed=42, size_level=1024).images[0]
        image.save(out / f.name)
        print("restored", f.name, flush=True)
