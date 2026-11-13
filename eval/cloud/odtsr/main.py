"""ODTSR (RedMediaTech, Apache) at 1x on his crops, on a Cerebrium L40.

Upload the crops with `cerebrium cp <dir> denoise/in`, run `cerebrium run main.py::run --fidelity 1.0`,
then `cerebrium download denoise/out-f1.0 .`.
"""
import os, pathlib, subprocess, sys

S = pathlib.Path("/persistent-storage/denoise")
REPO, QWEN, CKPT = S / "ODTSR", S / "Qwen-Image", S / "ODTSR-w" / "weight.pth"


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def run(fidelity: float = 1.0):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download, snapshot_download

    if not REPO.exists():
        sh(f"git clone --depth 1 https://github.com/RedMediaTech/ODTSR {REPO}")
    sh(f"sed -i 's/fidelity = [0-9.]*,/fidelity = {fidelity},/' {REPO}/examples/qwen_image/test_sr.py")
    sh(f"grep -vE '^(torch|torchvision|cupy)' {REPO}/requirements.txt > /tmp/req.txt && {sys.executable} -m pip install -q -r /tmp/req.txt && {sys.executable} -m pip install -q --no-build-isolation basicsr")
    import importlib.util
    deg = pathlib.Path(importlib.util.find_spec("basicsr").origin).parent / "data" / "degradations.py"
    deg.write_text(deg.read_text().replace("functional_tensor import", "_functional_tensor import"))
    snapshot_download("Qwen/Qwen-Image", local_dir=QWEN, allow_patterns=["transformer/*", "text_encoder/*", "vae/*", "tokenizer/*"])
    hf_hub_download("double8fun/ODTSR", "weight.pth", local_dir=CKPT.parent)
    inp, out = S / "in", S / f"out-f{fidelity}"
    env = {**os.environ, "qwen_path": str(QWEN), "PYTHONPATH": f"{REPO}:{REPO}/examples/qwen_image"}
    subprocess.run([sys.executable, "examples/qwen_image/test_sr.py", "--input_path", str(inp), "--output_path", str(out),
                    "--trained_ckpt", str(CKPT), "--scale", "1.0", "--cfg", "1.0", "--align_method", "wavelet"],
                   cwd=REPO, env=env, check=True)
    sh(f"ls -la {out}")
    return {"done": str(out)}
