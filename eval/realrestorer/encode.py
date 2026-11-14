"""Text encoder + VAE encode for every crop; saves inputs for the transformer stage."""
import sys
import time
from types import SimpleNamespace

from common import NAMES, PROMPT, ROOT, SEED, W
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from diffusers import RealRestorerAutoencoderKL, RealRestorerPipeline as P
from diffusers.utils.torch_utils import randn_tensor

dev = torch.device("mps")
t0 = time.time()
vae = RealRestorerAutoencoderKL.from_pretrained(W / "vae", torch_dtype=torch.float32).to(dev)
te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    W / "text_encoder", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to(dev).eval()
proc = AutoProcessor.from_pretrained(W / "processor")
ns = SimpleNamespace(
    vae=vae, text_encoder=te, processor=proc, max_token_length=640,
    _split_string=P._split_string, _prepare_qwen_image=P._prepare_qwen_image,
)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

with torch.inference_mode():
    for name in sys.argv[1:] or NAMES:
        t = time.time()
        torch.manual_seed(SEED)
        img = Image.open(ROOT / f"in/{name}.png").convert("RGB")
        resized, orig = P._resize_image(img, 1024)
        ref = P.load_image(resized).to(dev, torch.float32)
        h, w = ref.shape[-2:]
        ref_lat = P._pack_latents(P._encode_vae_image(ns, ref).to(torch.bfloat16))
        gen = torch.Generator("cpu").manual_seed(SEED)
        noise = randn_tensor((1, 16, h // 8, w // 8), generator=gen, device=dev, dtype=torch.bfloat16)
        pil = P._tensor_to_pil(ref[0])
        emb, mask = P._get_qwenvl_embeds(ns, [PROMPT, ""], [pil, pil], [1, 1], device=dev, dtype=torch.bfloat16)
        torch.save(
            {"ref_lat": ref_lat.cpu(), "latents": P._pack_latents(noise).cpu(), "emb": emb.cpu(),
             "mask": mask.cpu(), "h": h, "w": w, "orig": orig},
            ROOT / f"emb/{name}.pt",
        )
        print(f"{name}: {h}x{w} emb {tuple(emb.shape)} valid {mask.sum(1).tolist()} {time.time()-t:.0f}s", flush=True)
