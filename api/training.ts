// Every finished clean, unless its user turned sharing off: the photo exactly as given and the result,
// kept privately as training data for the model, with a note of how it was cleaned.
import { issueSignedToken, presignUrl, put } from "@vercel/blob";

const FILES = ["input", "result"] as const;

export async function POST(req: Request) {
  const { name, model, width, height }: { name?: string; model?: string; width?: number; height?: number } =
    await req.json();

  const folder = `training/${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
  const at = new Date().toISOString();
  await put(`${folder}/meta.json`, JSON.stringify({ at, name, model, width, height }), {
    access: "private",
    contentType: "application/json",
  });

  const token = await issueSignedToken({ operations: ["put"], validUntil: Date.now() + 3_600_000 });

  const urls = await Promise.all(
    FILES.map((file) =>
      presignUrl(token, {
        operation: "put",
        pathname: `${folder}/${file}`,
        access: "private",
        allowedContentTypes: ["image/*"],
        maximumSizeInBytes: 150_000_000,
      }),
    ),
  );

  return Response.json(Object.fromEntries(FILES.map((file, i) => [file, urls[i].presignedUrl])));
}
