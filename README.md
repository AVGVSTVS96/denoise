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

NAFNet, SIDD width 32 ([megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), MIT), as fp16 ONNX in `public/model/` (split into parts under 20 MB).

The ONNX file was exported with the model tools of [Hush](https://github.com/ms0242808/Hush) (Apache-2.0) by ms0242808, which bound each block's channel attention so dark high-ISO JPEG shadows don't turn into a striped grid, and keep LayerNorm and pooling in fp32 so WebGPU doesn't return NaN. Run `uv run export.py nafnet-sidd-w32` in Hush's `tools/models`; the output's sha256 matches `public/model/manifest.json`.

## License

MIT
