"""Transformer denoise + VAE decode, bf16, streaming DiT blocks from disk so the 25 GB model never sits in RAM at once."""
import json, sys, time

from common import NAMES, ROOT, W
import torch
import torch.nn.functional as F
from safetensors import safe_open
from diffusers import RealRestorerAutoencoderKL, RealRestorerTransformer2DModel, RealRestorerPipeline as P
from diffusers.models.realrestorer import attention as rr_attn
from diffusers.schedulers import RealRestorerFlowMatchScheduler

dev = torch.device("mps")
STEPS, CFG, MODEL_GUIDANCE, TRUNC, POWER = 28, 3.0, 3.5, 0.93, 0.4
names = sys.argv[1:] or NAMES


def chunked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, chunk=512):
    """Attention a slice of queries at a time, so the full attention matrix (tens of GB here) never exists at once."""
    if is_causal or q.shape[-2] <= chunk:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    rows = lambda i: attn_mask if attn_mask is None or attn_mask.shape[-2] == 1 else attn_mask[..., i : i + chunk, :]
    return torch.cat(
        [F.scaled_dot_product_attention(q[..., i : i + chunk, :], k, v, attn_mask=rows(i)) for i in range(0, q.shape[-2], chunk)],
        dim=-2,
    )


class _F:
    scaled_dot_product_attention = staticmethod(chunked_sdpa)


rr_attn.F = _F

tdir = W / "transformer"
cfg = {k: v for k, v in json.loads((tdir / "config.json").read_text()).items() if not k.startswith("_")}
with torch.device("meta"):
    tr = RealRestorerTransformer2DModel(**cfg)
wmap = json.loads((tdir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]


def load(prefix, skip=()):
    """Reads one prefix's weights, opening each file only for the read: a file kept open pins every page read from it in memory."""
    keys = {}
    for k, f in wmap.items():
        if k.startswith(prefix) and not k[len(prefix):].startswith(skip):
            keys.setdefault(f, []).append(k)
    out = {}
    for f, ks in keys.items():
        with safe_open(str(tdir / f), "pt", device="cpu") as fh:
            out.update({k[len(prefix):]: fh.get_tensor(k).to(dev, torch.bfloat16) for k in ks})
            torch.mps.synchronize()
    return out


m = tr.inner_model
static = load("inner_model.", skip=("double_blocks.", "single_blocks."))
m.load_state_dict(static, strict=False, assign=True)


def fetch_block(block, args):
    block.load_state_dict(load(block.prefix), strict=True, assign=True)


def drop_block(block, args, output):
    block.to("meta")
    torch.mps.empty_cache()


for kind, blocks in (("double_blocks", m.double_blocks), ("single_blocks", m.single_blocks)):
    for i, b in enumerate(blocks):
        b.prefix = f"inner_model.{kind}.{i}."
        b.register_forward_pre_hook(fetch_block)
        b.register_forward_hook(drop_block)
assert all(p.device.type == "mps" for n, p in m.named_parameters() if not n.startswith(("double_blocks", "single_blocks")))

vae = RealRestorerAutoencoderKL.from_pretrained(W / "vae", torch_dtype=torch.float32).to(dev)
dec = lambda z: vae.decode(z).sample
sched = RealRestorerFlowMatchScheduler()

for name in names:
    d = torch.load(ROOT / f"emb/{name}.pt")
    h, w = d["h"], d["w"]
    lat, ref = d["latents"].to(dev), d["ref_lat"].to(dev)
    emb, mask = d["emb"].to(dev), d["mask"].to(dev)
    ph, pw = h // 16, w // 16
    ids = P._prepare_img_ids(2, ph, pw, torch.bfloat16, dev, 0.0)
    ref_ids = P._prepare_img_ids(2, ph, pw, torch.bfloat16, dev, 0.0 if cfg["version"] == "v1.0" else 1.0)
    img_ids = torch.cat([ids, ref_ids], 1)
    txt_ids = torch.zeros(2, emb.shape[1], 3, dtype=torch.bfloat16, device=dev)
    sched.set_timesteps(num_inference_steps=STEPS, device=dev, image_seq_len=lat.shape[1])
    ts = sched.timesteps.tolist()
    t0 = time.time()
    with torch.inference_mode():
        for s, t in enumerate(ts[:-1]):
            x = torch.cat([lat.repeat(2, 1, 1), ref.repeat(2, 1, 1)], 1)
            tv = torch.full((2,), float(t), dtype=x.dtype, device=dev)
            gv = torch.full((2,), MODEL_GUIDANCE, dtype=x.dtype, device=dev)
            pred = tr(hidden_states=x, encoder_hidden_states=emb, prompt_embeds_mask=mask, timestep=tv,
                      txt_ids=txt_ids, img_ids=img_ids, guidance=gv, return_dict=False)[0][:, : lat.shape[1]]
            cond, uncond = pred.chunk(2)
            if float(t) > TRUNC:
                pred = uncond + CFG * (cond - uncond) / P.process_diff_norm(torch.norm(cond - uncond, dim=2, keepdim=True), k=POWER)
            else:
                pred = uncond + CFG * (cond - uncond)
            lat = sched.step(pred, t, lat, return_dict=False)[0]
            torch.mps.synchronize()
            print(f"{name} step {s+1}/{len(ts)-1} t={t:.3f} {time.time()-t0:.0f}s", flush=True)
        out = P._unpack_latents(lat.float(), h, w)
        img = dec(out).clamp(-1, 1).mul(0.5).add(0.5)
    P._tensor_to_pil(img[0].float()).resize(d["orig"]).save(ROOT / f"out/{name}.png")
    print(f"{name} done in {time.time()-t0:.0f}s", flush=True)
