#!/usr/bin/env bash
# RealRestorer on the Mac: transformer on his three crops (already encoded), then encode and restore the pair crops,
# then Qwen-Image-Edit-2511 on the pair crops. Each result is copied back as it lands.
set -euo pipefail
until tailscale status | rg 'bassims-macbook-pro-3' | rg -vq offline && ssh -o ConnectTimeout=15 mac true; do sleep 30; done
echo "mac back $(date); $(ssh mac uptime)"
E=~/Developer/denoise/eval
R='~/scratch/denoise-rr'
on() { ssh -o ServerAliveInterval=30 mac "cd $R && export PATH=\$HOME/.local/bin:\$PATH HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_ENABLE_MPS_FALLBACK=1 && $1"; }
fetch() { # fetch <mac-dir> <suffix>
  for n in $(ssh mac "ls $R/$1"); do
    stem=${n%.png}
    case $stem in R0006362-*) dst=$E/data/out/frontier/${stem}_$2.png ;; *) dst=$E/data/out/pairs/${stem}_$2.png ;; esac
    scp -q "mac:scratch/denoise-rr/$1/$n" "$dst"
  done
}
scp -q $E/realrestorer/*.py mac:scratch/denoise-rr/
on 'for n in signs truck facade; do [ -f emb/$n.pt ] && mv emb/$n.pt emb/R0006362-$n.pt; done; true'
his="R0006362-signs R0006362-truck R0006362-facade"
pairs="gr3x-025600-labels gr3x-102400-labels gr3x-025600-dial gr3x-102400-dial"
for n in $his; do on "[ -f out/$n.png ] || venv/bin/python restore.py $n"; fetch out realrestorer; done
echo "realrestorer his done $(date)"
on "[ -d w/text_encoder ] || venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"RealRestorer/RealRestorer\", local_dir=\"w\", allow_patterns=[\"text_encoder/*\"])'"
on "venv/bin/python encode.py $pairs && rm -rf w/text_encoder"
for n in $pairs; do on "[ -f out/$n.png ] || venv/bin/python restore.py $n"; fetch out realrestorer; done
echo "realrestorer pairs done $(date)"
on 'rm -rf w; for f in in/gr3x-*.png; do n=$(basename $f); [ -f qout/$n ] && continue; \
  mflux-generate-qwen-edit --model AbstractFramework/qwen-image-edit-2511-4bit --base-model qwen-image-edit-2511 \
    --image-paths $f --width 1024 --height 1024 --steps 30 --guidance 4.0 --seed 42 --no-exif \
    --prompt "Remove the image noise. Keep every detail, letter and color exactly as it is; change nothing else." \
    --output qout/$n; done'
fetch qout qwen-image-edit-2511
echo "qwen pairs done $(date)"
