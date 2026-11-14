#!/usr/bin/env bash
# Waits for the Mac on Tailscale, then runs RealRestorer and Qwen-Image-Edit-2511 there on his crops and the
# GR IIIx pair crops, copying each result back as soon as it exists. Safe to rerun: finished outputs are kept.
set -euo pipefail
until tailscale status | rg 'bassims-macbook-pro-3' | rg -vq offline && ssh -o ConnectTimeout=15 mac true; do sleep 30; done
echo "mac back $(date)"
E=~/Developer/denoise/eval
R='~/scratch/denoise-rr'
ssh mac "mkdir -p $R/in $R/emb $R/out $R/qout"
scp -q $E/data/his/R0006362-{signs,truck,facade}.png mac:scratch/denoise-rr/in/
for f in $E/data/his/pairs/*.png; do case $f in *-clean.png) ;; *) scp -q "$f" mac:scratch/denoise-rr/in/ ;; esac; done
scp -q $E/realrestorer_mac.py mac:scratch/denoise-rr/run.py

fetch() { # fetch <mac-dir> <suffix>
  for n in $(ssh mac "ls $R/$1"); do
    stem=${n%.png}
    case $stem in R0006362-*) dst=$E/data/out/frontier/${stem}_$2.png ;; *) dst=$E/data/out/pairs/${stem}_$2.png ;; esac
    mkdir -p "$(dirname "$dst")"; scp -q "mac:scratch/denoise-rr/$1/$n" "$dst"
  done
}

ssh mac "cd $R && export PATH=\$HOME/.local/bin:\$PATH UV_CACHE_DIR=\$PWD/uvcache HF_HOME=\$PWD/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
  ([ -d repo/.git ] || git clone -q --depth 1 https://github.com/yfyang007/RealRestorer repo) && \
  ([ -x rrvenv/bin/python ] || uv venv -q -p 3.11 rrvenv) && \
  VIRTUAL_ENV=\$PWD/rrvenv uv pip install -q torch torchvision -e repo/diffusers -r repo/requirements.txt optimum-quanto huggingface_hub hf_transfer && \
  rrvenv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"RealRestorer/RealRestorer\", local_dir=\"w\")' && \
  export PYTHONPATH=\$PWD/repo:\$PWD/repo/diffusers/src && \
  rrvenv/bin/python run.py encode w in emb && \
  rrvenv/bin/python run.py restore w in emb out"
fetch out realrestorer
echo "realrestorer done $(date)"

ssh mac "cd $R && export PATH=\$HOME/.local/bin:\$PATH HF_HUB_ENABLE_HF_TRANSFER=1 && for f in in/gr3x-*.png; do n=\$(basename \$f); [ -f qout/\$n ] && continue; \
  mflux-generate-qwen-edit --model AbstractFramework/qwen-image-edit-2511-4bit --base-model qwen-image-edit-2511 \
    --image-paths \$f --width 1024 --height 1024 --steps 30 --guidance 4.0 --seed 42 --no-exif \
    --prompt 'Remove the image noise. Keep every detail, letter and color exactly as it is; change nothing else.' \
    --output qout/\$n; done"
fetch qout qwen-image-edit-2511
echo "qwen pairs done $(date)"
