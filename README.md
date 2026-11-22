# denoise

Clean up high-ISO photos in the browser. The model runs on your GPU through WebGPU. Each finished clean, the photo as given and the result, is kept privately as training data for the model; turn that off under the Choose photo card and the photo never leaves your device, unless you sign in to clean it in the cloud.

```sh
bun install
bun dev
```

## How it works

1. The photo is decoded in a worker (resized to 16 MP if larger, to stay inside iOS Safari's canvas and memory limits).
2. It runs through NAFNet in 512 px tiles, one at a time on the GPU, with 64 px feathered overlaps so tile edges don't show.
3. The result is shown against the original and saved as a JPEG.

## Raw files

A DNG with a Bayer sensor is cleaned from its raw sensor data, not from its JPEG preview:

1. The worker reads the mosaic itself (uncompressed or lossless JPEG, strips or tiles, 8 to 16 bits), the black and white levels, and the camera's colour from the DNG's tags.
2. The mosaic is scaled and tiled the way `eval/train_raw.py`'s `denoise()` does it and runs through the joint model in 1024 px tiles (64 px of mirrored border, tiles fade into each other over the middle of what they share). A model that was not trained on lit parts of dark photos needs a tile with a light in it held to a lower gain of its own, by `train_raw.tile_gain`'s rule: no 128 px block brighter than a blown block in a photo that needs no gain. The model's manifest (`public/model/raw.json`) says whether it does; the one shipped, trained with `--bright`, does not, so every tile gets the photo's gain. The model denoises and demosaics in one pass and hands back the camera's RGB at full size.
3. That is developed on the spot: white balance as shot, highlights clipped where the sensor clipped, the camera's matrix to sRGB, the DNG's baseline exposure, a filmic tone curve. The result is shown against the camera's own JPEG preview, cropped and turned upright as the DNG says.
4. The noisy mosaic is developed the same way beside it, filled in by a plain bilinear demosaic. The Strength slider blends the result with that picture, from 0% (the photo as shot) to 100% (the model's result), and Save keeps the blend.

Black offsets that `eval/blacks.py` knows a camera body by come off first (`src/blacks.json`), for a raw that names its body by a serial number. A raw without one (Ricoh, Sony, Samsung) can only take the offsets of another body of its camera, which don't carry over, so it gets none unless that is turned on under the photo picker. Any other raw file, a DNG without a Bayer mosaic (ProRAW, monochrome, X-Trans), one packed in a way the reader doesn't know (lossy or JPEG XL), and every other photo goes the first way, through its preview and the sRGB model. Cloud cleaning always uses the preview.

## Cloud

Sign in with your own OpenRouter account and the photo is cleaned by Gemini 3.1 Flash Image or Gemini 3 Pro Image instead, billed to that account, and every cleaned photo is kept on that account: it opens again from Your photos on any device signed in to it. They see only a small picture, so a whole photo comes back with made-up text; the app sends it in 1024 px tiles and stitches them back, joining tiles in a narrow band so their details don't double up and putting back the photo's own brightness.

## Model

NAFNet width 32 ([megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), MIT), as fp16 ONNX in `public/model/` (split into parts under 20 MB). It starts from the published SIDD weights and is fine-tuned in `eval/train.py` on real high-ISO camera shots from NIND (Brummer & De Vleeschouwer) mixed with SIDD phone crops.

| Held-out test shots (PSNR, dB)        | SIDD weights | Fine-tuned |
| ------------------------------------- | ------------ | ---------- |
| Camera, ISO 12,800–102,400 (NIND, 78) | 28.55        | 29.23      |
| Camera, ISO ≤6,400 (PolyU, 100)       | 37.39        | 37.69      |
| Phone (SIDD, 320 crops)               | 40.43        | 40.32      |

The raw model is the same network with twelve planes in and out (packed RGGB in, each colour at each of a 2x2 cell's four places out), trained by `eval/train_raw.py --joint`. `eval/ship_raw.py` exports any joint checkpoint to `public/model/` and `public/model/raw.json`, after checking that no fp16 step comes near fp16's range and that the fp16 file matches PyTorch on test windows:

```sh
cd eval && uv run ship_raw.py data/models/raw/joint-v1/joint.pth
cd eval && uv run ship_raw.py data/models/raw/r8-p00/joint.pth bright
```

The second form is for a model trained with `--bright`: it ships without the hold-down for tiles with a light in them.

`eval/raw_check.ts` runs the app's raw path outside a browser and `eval/raw_develop.py` compares its picture with the Python pipeline's.

`eval/ship.py` exports the sRGB model with the model tools of [Hush](https://github.com/ms0242808/Hush) (Apache-2.0) by ms0242808, which bound each block's channel attention so dark high-ISO JPEG shadows don't turn into a striped grid, and keep LayerNorm and pooling in fp32 so WebGPU doesn't return NaN.

## License

MIT
