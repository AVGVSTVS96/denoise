// A signed-in user's finished photos, kept in a private Blob store under their OpenRouter user id.
// The browser sends its OpenRouter key; OpenRouter says whose it is, so no second login is needed.
// Files move straight between the browser and the store on short-lived presigned URLs.
import { issueSignedToken, list, presignUrl } from "@vercel/blob";

const HOUR = 3_600_000;
const FILES = ["thumb", "result", "original"] as const;

export type Photo = { id: string; name: string; at: number } & Partial<Record<(typeof FILES)[number], string>>;

async function owner(req: Request) {
  const auth = req.headers.get("authorization");

  if (!auth) return;
  const res = await fetch("https://openrouter.ai/api/v1/key", { headers: { Authorization: auth } });

  if (!res.ok) return;
  const { data }: { data?: { creator_user_id?: string | null } } = await res.json();

  return data?.creator_user_id ?? undefined;
}

const signer = () => issueSignedToken({ operations: ["get", "put"], validUntil: Date.now() + HOUR });

async function links(token: Awaited<ReturnType<typeof signer>>, folder: string, files: readonly string[]) {
  const urls = await Promise.all(
    files.map((file) => presignUrl(token, { operation: "get", pathname: `${folder}/${file}`, access: "private" })),
  );

  return Object.fromEntries(files.map((file, i) => [file, urls[i].presignedUrl]));
}

// Folder names are `<ms>-<name>`, so they sort and date themselves.
const describe = (id: string) => {
  const [, at, name] = id.match(/^(\d+)-(.*)$/) ?? [];

  return { id, at: Number(at), name: name || "photo" };
};

export async function GET(req: Request) {
  const user = await owner(req);

  if (!user) return new Response("Sign in with OpenRouter", { status: 401 });
  const prefix = `photos/${user}/`;
  const folders = new Map<string, Set<string>>();
  let cursor: string | undefined;

  do {
    const page = await list({ prefix, cursor, limit: 1000 });

    for (const blob of page.blobs) {
      const [id, file] = blob.pathname.slice(prefix.length).split("/");
      folders.set(id, (folders.get(id) ?? new Set()).add(file));
    }

    cursor = page.cursor;
  } while (cursor);

  const token = await signer();

  const photos: Photo[] = await Promise.all(
    [...folders].flatMap(([id, files]) =>
      files.has("result") && files.has("thumb")
        ? [links(token, prefix + id, [...files]).then((urls) => ({ ...describe(id), ...urls }))]
        : [],
    ),
  );

  return Response.json(photos.toSorted((a, b) => b.at - a.at));
}

// Hands out upload URLs for one new photo, and where to read it back.
export async function POST(req: Request) {
  const user = await owner(req);

  if (!user) return new Response("Sign in with OpenRouter", { status: 401 });
  const { name }: { name?: string } = await req.json();

  const slug = (name ?? "")
    .replace(/[^\w.-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);

  const id = `${Date.now()}-${slug || "photo"}`;
  const folder = `photos/${user}/${id}`;
  const token = await signer();

  const put = await Promise.all(
    FILES.map((file) =>
      presignUrl(token, {
        operation: "put",
        pathname: `${folder}/${file}`,
        access: "private",
        allowedContentTypes: ["image/*"],
        maximumSizeInBytes: 100_000_000,
      }),
    ),
  );

  return Response.json({
    put: Object.fromEntries(FILES.map((file, i) => [file, put[i].presignedUrl])),
    photo: { ...describe(id), ...(await links(token, folder, FILES)) },
  });
}
