# denoise

Clean up high-ISO photos in the browser. The model runs on your GPU through WebGPU and the photo never leaves your device, unless you sign in to clean it in the cloud.

```sh
bun install
bun dev
```

## How it works

1. The photo is decoded in a worker (resized to 16 MP if larger, to stay inside iOS Safari's canvas and memory limits).
2. It runs through NAFNet in 512 px tiles, one at a time on the GPU, with 64 px feathered overlaps so tile edges don't show.
3. The result is shown against the original and saved as a JPEG.

Raw files (DNG) are read through the camera's full-size JPEG preview, not the tiny thumbnail browsers decode.

## Cloud

Sign in with your own OpenRouter account and the photo is cleaned by Gemini 3.1 Flash Image or Gemini 3 Pro Image instead, billed to that account. They see only a small picture, so a whole photo comes back with made-up text; the app sends it in 1024 px tiles and stitches them back, joining tiles in a narrow band so their details don't double up and putting back the photo's own brightness.

## Model

NAFNet width 32 ([megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), MIT), as fp16 ONNX in `public/model/` (split into parts under 20 MB). It starts from the published SIDD weights and is fine-tuned in `eval/train.py` on real high-ISO camera shots from NIND (Brummer & De Vleeschouwer) mixed with SIDD phone crops.

| Held-out test shots (PSNR, dB)        | SIDD weights | Fine-tuned |
| ------------------------------------- | ------------ | ---------- |
| Camera, ISO 12,800–102,400 (NIND, 78) | 28.55        | 29.23      |
| Camera, ISO ≤6,400 (PolyU, 100)       | 37.39        | 37.69      |
| Phone (SIDD, 320 crops)               | 40.43        | 40.32      |

`eval/ship.py` exports it with the model tools of [Hush](https://github.com/ms0242808/Hush) (Apache-2.0) by ms0242808, which bound each block's channel attention so dark high-ISO JPEG shadows don't turn into a striped grid, and keep LayerNorm and pooling in fp32 so WebGPU doesn't return NaN.

## License

MIT
