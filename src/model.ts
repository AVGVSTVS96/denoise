import manifest from "../public/model/manifest.json";

export const modelBytes = manifest.bytes;

const cacheName = `model-${manifest.sha256.slice(0, 12)}`;

export async function loadModel(onProgress: (loaded: number) => void): Promise<Uint8Array> {
  const cache = await caches.open(cacheName).catch(() => undefined);
  const out = new Uint8Array(manifest.bytes);
  let offset = 0;

  for (const part of manifest.parts) {
    const url = `/model/${part}`;
    let res = await cache?.match(url);

    if (!res) {
      const net = await fetch(url);

      if (!net.ok) throw new Error(`Model download failed (${net.status})`);
      const body = await readAll(net, (n) => onProgress(offset + n));
      await cache?.put(url, new Response(body)).catch(() => {});
      out.set(body, offset);
      offset += body.byteLength;
      continue;
    }

    const body = new Uint8Array(await res.arrayBuffer());
    out.set(body, offset);
    offset += body.byteLength;
    onProgress(offset);
  }

  if (offset !== manifest.bytes) throw new Error("Model download was incomplete");

  return out;
}

async function readAll(res: Response, onProgress: (n: number) => void): Promise<Uint8Array<ArrayBuffer>> {
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

  const body = new Uint8Array(n);
  let at = 0;

  for (const c of chunks) {
    body.set(c, at);
    at += c.byteLength;
  }

  return body;
}
