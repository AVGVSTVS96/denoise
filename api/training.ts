// Every finished clean, unless its user turned sharing off: the photo exactly as given and the result,
// kept privately as training data for the model, with a note of how it was cleaned.
// Cleaning on the device needs no sign-in, so a request can't prove a clean happened. What keeps the store
// from being a free upload bucket: BotID lets only a real browser on the site through, each visitor gets
// a few dozen cleans a day, and the whole site a few hundred.
import { issueSignedToken, list, presignUrl, put } from "@vercel/blob";
import { checkBotId } from "botid/server";
import { MODELS } from "../src/cloud.js";

const PER_VISITOR = 40;
const PER_DAY = 200;
const MODEL_IDS = new Set<string>(["on-device", ...Object.values(MODELS).map((m) => m.id)]);

const LIMITS = {
  input: { allowedContentTypes: ["image/*"], maximumSizeInBytes: 100_000_000 },
  result: { allowedContentTypes: ["image/jpeg"], maximumSizeInBytes: 40_000_000 },
};

// A keyed hash of the address, so the store never holds anyone's IP.
async function visitor(req: Request) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0].trim() ?? "unknown";
  const enc = new TextEncoder();

  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(process.env.BLOB_READ_WRITE_TOKEN),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );

  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, enc.encode(ip)));

  return Array.from(mac.slice(0, 8), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function POST(req: Request) {
  const { isBot } = await checkBotId({ advancedOptions: { headers: Object.fromEntries(req.headers) } });

  if (isBot) return new Response("Only the denoise app can share training photos", { status: 403 });

  const body: { name?: string; model?: string; width?: number; height?: number } = await req.json();

  const meta = {
    at: new Date().toISOString(),
    name: String(body.name ?? "photo").slice(0, 200),
    model: String(body.model),
    width: Number(body.width),
    height: Number(body.height),
  };

  if (!MODEL_IDS.has(meta.model) || !Number.isInteger(meta.width) || !Number.isInteger(meta.height))
    return new Response("Not a finished clean", { status: 400 });

  const day = `training/${new Date().toISOString().slice(0, 10)}/`;
  const who = await visitor(req);
  const { folders } = await list({ prefix: day, mode: "folded", limit: 1000 });

  if (folders.length >= PER_DAY || folders.filter((f) => f.startsWith(day + who)).length >= PER_VISITOR)
    return new Response("Enough training photos for today", { status: 429 });

  const folder = `${day}${who}-${Date.now()}`;
  await put(`${folder}/meta.json`, JSON.stringify(meta), {
    access: "private",
    contentType: "application/json",
  });

  const token = await issueSignedToken({ operations: ["put"], validUntil: Date.now() + 900_000 });

  const urls = await Promise.all(
    Object.entries(LIMITS).map(([file, limit]) =>
      presignUrl(token, { operation: "put", pathname: `${folder}/${file}`, access: "private", ...limit }).then(
        (url) => [file, url.presignedUrl],
      ),
    ),
  );

  return Response.json(Object.fromEntries(urls));
}
