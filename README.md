# denoise

Clean up high-ISO photos in the browser. The model runs on your GPU through WebGPU; the photo never leaves your device.

```sh
bun install
bun dev
```

## How it works

1. The photo is decoded in a worker (resized to 16 MP if larger, to stay inside iOS Safari's canvas and memory limits).
2. It runs through NAFNet in 512 px tiles, one at a time on the GPU, with 64 px feathered overlaps so tile edges don't show.
3. The result is shown against the original and saved as a JPEG.

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
