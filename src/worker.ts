import * as ort from "onnxruntime-web/webgpu";
import * as cloud from "./cloud";
import { rawPreview } from "./dng";
import { loadModel } from "./model";

export type FromWorker =
  | { type: "model"; loaded: number }
  | { type: "start"; width: number; height: number }
  | { type: "tile"; x: number; y: number; w: number; h: number; pixels: Uint8ClampedArray<ArrayBuffer> }
  | { type: "progress"; done: number; total: number }
  | { type: "done"; blob: Blob }
  | { type: "error"; message: string };

const TILE = 512;
const OVERLAP = 64;
const MAX_PIXELS = 16_000_000;

let session: Promise<ort.InferenceSession> | undefined;

const post = (msg: FromWorker, transfer: Transferable[] = []) => postMessage(msg, { transfer });

const FEATURES: GPUFeatureName[] = ["shader-f16", "subgroups", "timestamp-query"];

const LIMITS = [
  "maxBindGroups",
  "maxComputeWorkgroupStorageSize",
  "maxComputeWorkgroupsPerDimension",
  "maxStorageBufferBindingSize",
  "maxBufferSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
] as const;

async function requestDevice() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });

  if (!adapter) throw new Error("This browser doesn't have WebGPU.");

  if (!adapter.features.has("shader-f16")) throw new Error("This GPU can't run the model (no shader-f16).");

  ort.env.webgpu.adapter = adapter;

  return adapter.requestDevice({
    requiredFeatures: FEATURES.filter((f) => adapter.features.has(f)),
    requiredLimits: Object.fromEntries(LIMITS.map((name) => [name, adapter.limits[name]])),
  });
}

function getSession() {
  session ??= (async () => {
    const device = await requestDevice();
    const model = await loadModel((loaded) => post({ type: "model", loaded }));

    return ort.InferenceSession.create(model, {
      executionProviders: [{ name: "webgpu", preferredLayout: "NCHW", device }],
      graphOptimizationLevel: "all",
    });
  })();
  session.catch(() => (session = undefined));

  return session;
}

async function decode(file: Blob) {
  const bitmap = await createImageBitmap((await rawPreview(file)) ?? file);
  const scale = Math.min(1, Math.sqrt(MAX_PIXELS / (bitmap.width * bitmap.height)));
  const canvas = new OffscreenCanvas(Math.floor(bitmap.width * scale), Math.floor(bitmap.height * scale));
  const ctx = canvas.getContext("2d")!;

  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  return { canvas, ctx };
}

const positions = (size: number, tile = TILE, overlap = OVERLAP) => {
  if (size <= tile) return [0];
  const n = Math.ceil((size - overlap) / (tile - overlap));

  return Array.from({ length: n }, (_, i) => Math.round((i * (size - tile)) / (n - 1)));
};

const ramp = (n: number, first: boolean, last: boolean) =>
  Float32Array.from({ length: n }, (_, i) =>
    Math.min(1, first ? 1 : (i + 0.5) / OVERLAP, last ? 1 : (n - i - 0.5) / OVERLAP),
  );

// Generated tiles disagree on fine detail, so averaging their overlap would show both versions:
// each pixel takes the tile it is deepest in, with a short crossfade at the middle of the overlap.
const clamp = (v: number) => Math.min(1, Math.max(0, v));

const seam = (starts: number[], i: number, tile: number) => {
  const blend = 32;
  const from = i > 0 ? (starts[i - 1] + tile + starts[i]) / 2 : -Infinity;
  const to = i < starts.length - 1 ? (starts[i] + tile + starts[i + 1]) / 2 : Infinity;

  return Float32Array.from({ length: tile }, (_, t) => {
    const g = starts[i] + t + 0.5;

    return clamp((g - from) / blend + 0.5) * clamp((to - g) / blend + 0.5);
  });
};

// Turns one finished band of weighted sums into pixels, shows it, and shifts the next band up.
function flush(
  ctx: OffscreenCanvasRenderingContext2D,
  acc: Float32Array,
  W: number,
  y: number,
  rows: number,
  scale: number,
) {
  const pixels = new Uint8ClampedArray(W * rows * 4);

  for (let p = 0; p < pixels.length; p += 4) {
    const w = acc[p + 3];
    pixels[p] = (acc[p] / w) * scale + 0.5;
    pixels[p + 1] = (acc[p + 1] / w) * scale + 0.5;
    pixels[p + 2] = (acc[p + 2] / w) * scale + 0.5;
    pixels[p + 3] = 255;
  }

  ctx.putImageData(new ImageData(pixels, W, rows), 0, y);
  post({ type: "tile", x: 0, y, w: W, h: rows, pixels }, [pixels.buffer]);
  const band = acc.length / (W * 4);
  acc.copyWithin(0, W * rows * 4);
  acc.fill(0, W * (band - rows) * 4);
}

async function runCloud(file: Blob, key: string, model: cloud.Model) {
  const { canvas, ctx } = await decode(file);
  const { width: W, height: H } = canvas;
  post({ type: "start", width: W, height: H });
  const tile = Math.min(cloud.TILE, W, H);
  const overlap = Math.min(cloud.OVERLAP, tile >> 2);
  const xs = positions(W, tile, overlap);
  const ys = positions(H, tile, overlap);
  const total = xs.length * ys.length;
  let done = 0;
  post({ type: "progress", done, total });
  const acc = new Float32Array(W * tile * 4);

  for (let r = 0; r < ys.length; r++) {
    const wy = ys[r];
    const ry = seam(ys, r, tile);

    const outs = await Promise.all(
      xs.map(async (wx) => {
        const out = await cloud.cleanTile(key, model, ctx.getImageData(wx, wy, tile, tile));
        post({ type: "progress", done: ++done, total });

        return out;
      }),
    );

    outs.forEach(({ data }, c) => {
      const wx = xs[c];
      const rx = seam(xs, c, tile);

      for (let y = 0; y < tile; y++) {
        for (let x = 0; x < tile; x++) {
          const w = ry[y] * rx[x];
          const o = (y * tile + x) * 4;
          const a = (y * W + wx + x) * 4;
          acc[a] += data[o] * w;
          acc[a + 1] += data[o + 1] * w;
          acc[a + 2] += data[o + 2] * w;
          acc[a + 3] += w;
        }
      }
    });

    flush(ctx, acc, W, wy, r === ys.length - 1 ? tile : ys[r + 1] - wy, 1);
  }

  post({ type: "done", blob: await canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 }) });
}

const padTo16 = (n: number) => Math.ceil(n / 16) * 16;

async function run(file: Blob) {
  const { canvas, ctx } = await decode(file);
  const { width: W, height: H } = canvas;
  post({ type: "start", width: W, height: H });
  const src = ctx.getImageData(0, 0, W, H).data;
  const sess = await getSession();
  const xs = positions(W);
  const ys = positions(H);
  const total = xs.length * ys.length;
  let done = 0;
  post({ type: "progress", done, total });

  const winW = Math.min(TILE, W);
  const winH = Math.min(TILE, H);
  const tw = padTo16(winW);
  const th = padTo16(winH);
  const plane = tw * th;
  const input = new Float32Array(3 * plane);
  const acc = new Float32Array(W * winH * 4);

  for (let r = 0; r < ys.length; r++) {
    const wy = ys[r];
    const ry = ramp(winH, r === 0, r === ys.length - 1);

    for (let c = 0; c < xs.length; c++) {
      const wx = xs[c];
      const rx = ramp(winW, c === 0, c === xs.length - 1);

      for (let y = 0; y < th; y++) {
        const row = (wy + Math.min(y, winH - 1)) * W + wx;

        for (let x = 0; x < tw; x++) {
          const i = (row + Math.min(x, winW - 1)) * 4;
          const o = y * tw + x;
          input[o] = src[i] / 255;
          input[plane + o] = src[i + 1] / 255;
          input[2 * plane + o] = src[i + 2] / 255;
        }
      }

      const result = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, th, tw]) });
      const tensor = result[sess.outputNames[0]];
      // SAFETY: the model's output is declared float32 in its ONNX graph.
      const out = (await tensor.getData()) as Float32Array;
      tensor.dispose();

      for (let y = 0; y < winH; y++) {
        for (let x = 0; x < winW; x++) {
          const w = ry[y] * rx[x];
          const o = y * tw + x;
          const a = (y * W + wx + x) * 4;
          acc[a] += out[o] * w;
          acc[a + 1] += out[plane + o] * w;
          acc[a + 2] += out[2 * plane + o] * w;
          acc[a + 3] += w;
        }
      }

      post({ type: "progress", done: ++done, total });
    }

    flush(ctx, acc, W, wy, r === ys.length - 1 ? winH : ys[r + 1] - wy, 255);
  }

  post({ type: "done", blob: await canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 }) });
}

onmessage = (e: MessageEvent<{ file: Blob; key?: string; model: cloud.Model }>) => {
  (e.data.key ? runCloud(e.data.file, e.data.key, e.data.model) : run(e.data.file)).catch((err: Error) =>
    post({ type: "error", message: err.message }),
  );
};
