import manifest from "../public/model/manifest.json";

export const modelBytes = manifest.bytes;

const cacheName = `model-${manifest.sha256.slice(0, 12)}`;
const ATTEMPTS = 3;

export async function loadModel(onProgress: (loaded: number) => void): Promise<Uint8Array> {
  const cache = await caches.open(cacheName).catch(() => undefined);
  const out = new Uint8Array(manifest.bytes);
  let offset = 0;

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
  const chunks: Uint8Array[] = [];
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
