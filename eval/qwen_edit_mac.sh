#!/bin/zsh
# Qwen-Image-Edit-2511 (4-bit MLX) on his crops, on an Apple Silicon Mac with mflux; about 45 minutes a crop on 32 GB.
export PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
cd ~/scratch/denoise-qwen
for c in signs truck facade; do
  mflux-generate-qwen-edit --model AbstractFramework/qwen-image-edit-2511-4bit --base-model qwen-image-edit-2511 \
    --image-paths in/R0006362-$c.png --width 1024 --height 1024 --steps 30 --guidance 4.0 --seed 42 --no-exif \
    --prompt "Remove the image noise. Keep every detail, letter and color exactly as it is; change nothing else." \
    --output out-$c.png
done
echo ALLDONE
