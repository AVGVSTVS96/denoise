import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import manifest from "../public/model/raw.json";
import { blackOffsets } from "../src/blacks";
import { rawPhoto } from "../src/dng";
import { clean, levelsOf, type Brightest } from "../src/raw";

const [photo, out] = process.argv.slice(2);
const model = join(import.meta.dirname, "../public/model");
const parts = await Promise.all(manifest.parts.map((part) => readFile(join(model, part))));

ort.env.wasm.numThreads = 4;

const session = await ort.InferenceSession.create(Buffer.concat(parts), { executionProviders: ["wasm"] });
const raw = await rawPhoto(new Blob([await readFile(photo)]));

if (!raw) throw new Error(`${photo} is not a Bayer DNG`);

const offsets = await blackOffsets(raw.body, raw.iso);
const { width, height } = raw.crop;
const picture = new Uint8Array(width * height * 3);
const started = performance.now();

const brightest: Brightest | undefined = manifest.brightest;
const { gain } = levelsOf(raw, offsets, brightest?.block ?? 1024);

console.log(JSON.stringify({ body: raw.body, iso: raw.iso, offsets, gain, width, height }));

await clean(
  raw,
  offsets,
  brightest,
  async (planes, h, w) => {
    const result = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", planes, [1, 4, h, w]) });

    // SAFETY: the model's output is declared float32 in its ONNX graph.
    return result[session.outputNames[0]].data as Float32Array;
  },
  (piece, done, total) => {
    for (let y = 0; y < piece.height; y++) {
      for (let x = 0; x < piece.width; x++) {
        const from = (y * piece.width + x) * 4;
        const to = ((piece.y + y) * width + piece.x + x) * 3;
        const shown = piece.pixels[from + 3] / 255;

        for (let k = 0; k < 3; k++)
          picture[to + k] = Math.round(piece.pixels[from + k] * shown + picture[to + k] * (1 - shown));
      }
    }

    console.log(`${done}/${total} tiles, ${Math.round((performance.now() - started) / 1000)}s`);
  },
);

await writeFile(out, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), picture]));
