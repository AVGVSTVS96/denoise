import { upload } from "./library";

const OFF = "training-off";

export const sharing = () => localStorage.getItem(OFF) === null;

export const setSharing = (on: boolean) => (on ? localStorage.removeItem(OFF) : localStorage.setItem(OFF, "1"));

export async function share(input: File, result: Blob, info: { model: string; width: number; height: number }) {
  const res = await fetch("/api/training", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: input.name, ...info }),
  });

  if (!res.ok) return;
  const put: { input: string; result: string } = await res.json();
  await Promise.all([upload(put.input, input, input.type || "image/x-adobe-dng"), upload(put.result, result)]);
}
