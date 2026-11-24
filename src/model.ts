import crisp from "../public/model/crisp.json";
import photo from "../public/model/manifest.json";
import raw from "../public/model/raw.json";
import type { Brightest } from "./raw";

type Manifest = { name: string; bytes: number; sha256: string; parts: string[] };

type RawManifest = Manifest & { brightest: Brightest | null };

type Models = { photo: Manifest; raw: RawManifest; crisp: RawManifest };

export const models: Models = { photo, raw, crisp };

export type ModelKind = keyof typeof models;

export type RawKind = Exclude<ModelKind, "photo">;

export type Look = RawKind | "burst";

const cacheName = (manifest: Manifest) => `model-${manifest.sha256.slice(0, 12)}`;
const ATTEMPTS = 3;

async function forgetOld() {
  const current = Object.values(models).map(cacheName);
  const names = await caches.keys();
  await Promise.all(names.filter((n) => n.startsWith("model-") && !current.includes(n)).map((n) => caches.delete(n)));
}

export async function loadModel(kind: ModelKind, onProgress: (loaded: number) => void): Promise<Uint8Array> {
  const manifest = models[kind];
  const cache = await caches.open(cacheName(manifest)).catch(() => undefined);
  const out = new Uint8Array(manifest.bytes);
  let offset = 0;
  forgetOld().catch(() => {});

  for (const part of manifest.parts) {
    const url = `/model/${part}`;
    const cached = await cache?.match(url);

    const body = cached
      ? new Uint8Array(await cached.arrayBuffer())
      : await download(url, (n) => onProgress(offset + n));

    if (!cached) await cache?.put(url, new Response(body)).catch(() => {});
    out.set(body, offset);
    offset += body.byteLength;
    onProgress(offset);
  }

  if (offset !== manifest.bytes) throw new Error("Model download was incomplete");

  return out;
}

async function download(url: string, onProgress: (n: number) => void) {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(url);

      if (!res.ok) throw new Error(`Model download failed (${res.status})`);

      return new Uint8Array(await readAll(res, onProgress));
    } catch (err) {
      if (attempt === ATTEMPTS) throw err;
      onProgress(0);
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  }
}

async function readAll(res: Response, onProgress: (n: number) => void) {
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let n = 0;
  const reader = res.body!.getReader();

  for (;;) {
    const { done, value } = await reader.read();

    if (done) break;
    chunks.push(value);
    n += value.byteLength;
    onProgress(n);
  }

  return new Blob(chunks).arrayBuffer();
}
