import * as ort from "onnxruntime-web/webgpu";
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
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, Math.sqrt(MAX_PIXELS / (bitmap.width * bitmap.height)));
  const canvas = new OffscreenCanvas(Math.floor(bitmap.width * scale), Math.floor(bitmap.height * scale));
  const ctx = canvas.getContext("2d")!;

  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  return { canvas, ctx };
}

const positions = (size: number) => {
  if (size <= TILE) return [0];
  const n = Math.ceil((size - OVERLAP) / (TILE - OVERLAP));

  return Array.from({ length: n }, (_, i) => Math.round((i * (size - TILE)) / (n - 1)));
};

const ramp = (n: number, first: boolean, last: boolean) =>
  Float32Array.from({ length: n }, (_, i) =>
    Math.min(1, first ? 1 : (i + 0.5) / OVERLAP, last ? 1 : (n - i - 0.5) / OVERLAP),
  );

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

    const rows = r === ys.length - 1 ? winH : ys[r + 1] - wy;
    const pixels = new Uint8ClampedArray(W * rows * 4);

    for (let p = 0; p < pixels.length; p += 4) {
      const w = acc[p + 3];
      pixels[p] = (acc[p] / w) * 255 + 0.5;
      pixels[p + 1] = (acc[p + 1] / w) * 255 + 0.5;
      pixels[p + 2] = (acc[p + 2] / w) * 255 + 0.5;
      pixels[p + 3] = 255;
    }

    ctx.putImageData(new ImageData(pixels, W, rows), 0, wy);
    post({ type: "tile", x: 0, y: wy, w: W, h: rows, pixels }, [pixels.buffer]);
    acc.copyWithin(0, W * rows * 4);
    acc.fill(0, W * (winH - rows) * 4);
  }

  post({ type: "done", blob: await canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 }) });
}

onmessage = (e: MessageEvent<{ file: Blob }>) => {
  run(e.data.file).catch((err: Error) => post({ type: "error", message: err.message }));
};
