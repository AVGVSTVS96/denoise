import type { Photo } from "../api/photos";

export type { Photo };

const THUMB = 480;

const api = (key: string, init?: RequestInit) =>
  fetch("/api/photos", {
    ...init,
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
  }).then(async (res) => {
    if (!res.ok) throw new Error(`Your photos couldn't be reached (${res.status})`);

    return res.json();
  });

export const listPhotos = (key: string): Promise<Photo[]> => api(key);

export async function savePhoto(key: string, name: string, original: Blob, result: Blob): Promise<Photo> {
  const { put, photo }: { put: Record<"thumb" | "result" | "original", string>; photo: Photo } = await api(key, {
    method: "POST",
    body: JSON.stringify({ name }),
  });

  // The library lists a photo once its result and thumbnail are in, so the original goes first.
  await upload(put.original, original);
  await Promise.all([upload(put.result, result), upload(put.thumb, await thumbnail(result))]);

  return photo;
}

async function upload(url: string, blob: Blob) {
  const res = await fetch(url, { method: "PUT", body: blob, headers: { "Content-Type": blob.type || "image/jpeg" } });

  if (!res.ok) throw new Error(`Saving to your account failed (${res.status})`);
}

async function thumbnail(blob: Blob) {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, THUMB / Math.min(bitmap.width, bitmap.height));
  const canvas = new OffscreenCanvas(Math.round(bitmap.width * scale), Math.round(bitmap.height * scale));
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  return canvas.convertToBlob({ type: "image/jpeg", quality: 0.85 });
}
