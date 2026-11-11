// One tile through a Gemini image model on OpenRouter. It sees only a small picture, so a whole photo
// comes back with invented text; 1024 px tiles keep it to what each part of the photo shows.
export const MODELS = {
  flash: { id: "google/gemini-3.1-flash-image", name: "Gemini 3.1 Flash", perTile: 0.068 },
  pro: { id: "google/gemini-3-pro-image", name: "Gemini 3 Pro", perTile: 0.139 },
};

export type Model = keyof typeof MODELS;

export const TILE = 1024;
export const OVERLAP = 128;
const ATTEMPTS = 3;
const GRID = 8;

const PROMPT =
  "This is a real photo shot at very high ISO. Remove the sensor noise (grain and color speckles) only. " +
  "Keep it the same photo: same framing, geometry, colors, exposure and every real edge and texture. " +
  "Do not add, remove, sharpen into, or redraw any detail that is not visibly there: where text or texture is " +
  "too noisy to read, leave it soft and unreadable rather than guessing it. Return the cleaned photo.";

export async function cleanTile(key: string, model: Model, tile: ImageData): Promise<ImageData> {
  const canvas = new OffscreenCanvas(tile.width, tile.height);
  canvas.getContext("2d")!.putImageData(tile, 0, 0);
  const upload = await dataUrl(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.95 }));

  for (let attempt = 1; ; attempt++) {
    try {
      return matchTone(await resize(await request(key, model, upload), tile.width, tile.height), tile);
    } catch (err) {
      if (attempt === ATTEMPTS || (err instanceof CloudError && err.status === 401)) throw err;
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  }
}

async function dataUrl(blob: Blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let text = "";

  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));

  return `data:${blob.type};base64,${btoa(text)}`;
}

class CloudError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request(key: string, model: Model, image: string) {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "denoise" },
    body: JSON.stringify({
      model: MODELS[model].id,
      modalities: ["image", "text"],
      image_config: { aspect_ratio: "1:1", image_size: "1K" },
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: PROMPT },
            { type: "image_url", image_url: { url: image } },
          ],
        },
      ],
    }),
  });

  const body: {
    error?: { message?: string };
    choices?: { message?: { images?: { image_url: { url: string } }[] } }[];
  } = await res.json().catch(() => ({}));

  const url = body.choices?.[0]?.message?.images?.[0]?.image_url.url;

  if (!res.ok || !url) {
    const message =
      res.status === 401
        ? "OpenRouter didn't accept the sign-in. Sign out and in again."
        : (body.error?.message ?? `OpenRouter returned no picture (${res.status})`);

    throw new CloudError(message, res.status);
  }

  return (await fetch(url)).blob();
}

async function resize(blob: Blob, width: number, height: number) {
  const bitmap = await createImageBitmap(blob, { resizeWidth: width, resizeHeight: height, resizeQuality: "high" });
  const ctx = new OffscreenCanvas(width, height).getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  return ctx.getImageData(0, 0, width, height);
}

// Mean brightness of each cell of a GRID × GRID grid, averaged exactly (canvas downscaling skips pixels).
function brightness({ width: w, height: h, data }: ImageData) {
  const sum = new Float32Array(GRID * GRID);
  const count = new Float32Array(GRID * GRID);

  for (let y = 0; y < h; y++) {
    const row = Math.floor((y * GRID) / h) * GRID;

    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      const cell = row + Math.floor((x * GRID) / w);
      sum[cell] += data[p] + data[p + 1] + data[p + 2];
      count[cell] += 3;
    }
  }

  return sum.map((v, i) => v / count[i]);
}

// The model sometimes lights a tile a little differently: keep its detail and colour, but put back the
// photo's own brightness, broadly, so tiles don't show. Colour is left alone; in deep shadow the photo's
// colour average is tinted by the noise itself.
function matchTone(out: ImageData, input: ImageData) {
  const a = brightness(input);
  const b = brightness(out);
  const delta = a.map((v, i) => v - b[i]);
  const { width: w, height: h, data } = out;

  for (let y = 0; y < h; y++) {
    const gy = Math.min(GRID - 1, Math.max(0, ((y + 0.5) * GRID) / h - 0.5));
    const y0 = Math.floor(gy);
    const y1 = Math.min(GRID - 1, y0 + 1);
    const fy = gy - y0;

    for (let x = 0; x < w; x++) {
      const gx = Math.min(GRID - 1, Math.max(0, ((x + 0.5) * GRID) / w - 0.5));
      const x0 = Math.floor(gx);
      const x1 = Math.min(GRID - 1, x0 + 1);
      const fx = gx - x0;
      const top = delta[y0 * GRID + x0] * (1 - fx) + delta[y0 * GRID + x1] * fx;
      const bottom = delta[y1 * GRID + x0] * (1 - fx) + delta[y1 * GRID + x1] * fx;
      const shift = top * (1 - fy) + bottom * fy;
      const p = (y * w + x) * 4;
      data[p] += shift;
      data[p + 1] += shift;
      data[p + 2] += shift;
    }
  }

  return out;
}
