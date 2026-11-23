import { useCallback, useEffect, useRef, useState } from "react";
import { borrowing, setBorrowing } from "./blacks";
import { MODELS, type Model } from "./cloud";
import { rawPreview } from "./dng";
import { listPhotos, savePhoto, type Photo } from "./library";
import type { RawKind } from "./model";
import { detailed, MARGIN } from "./raw";
import { setSharing, share, sharing } from "./training";
import { cloudKey, finishSignIn, signIn, signOut } from "./openrouter";
import type { FromWorker, ToWorker } from "./worker";
import Worker from "./worker?worker";

type Run = {
  stage: "decode" | "model" | "clean" | "done";
  band?: number;
  loaded: number;
  modelBytes: number;
  done: number;
  total: number;
  startedAt: number;
  firstTileAt?: number;
  finishedAt?: number;
  result?: Blob;
};

type Job = {
  name: string;
  file?: File;
  original: string;
  cloud?: string;
  width: number;
  height: number;
  raw?: boolean;
  shown: RawKind;
  runs: Partial<Record<RawKind, Run>>;
  cleaned?: ImageBitmap;
  savedAt?: number;
  saving?: "saving" | "saved" | "failed";
};

type Canvases = React.RefObject<Partial<Record<RawKind, HTMLCanvasElement | null>>>;

type Finished = (result: Blob, info: { model: string; width: number; height: number }) => void;

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

const CHOICES: Model[] = ["flash", "pro"];
const LOOKS: RawKind[] = ["raw", "new", "crisp"];
const LOOK = { raw: "Standard", new: "Standard 2", crisp: "Crisper" };

const fresh = (): Run => ({
  stage: "decode",
  loaded: 0,
  modelBytes: 0,
  done: 0,
  total: 0,
  startedAt: performance.now(),
});

const isPhoto = (f: File) => f.type.startsWith("image/") || /\.dng$/i.test(f.name);

export function App() {
  const [job, setJob] = useState<Job>();
  const [error, setError] = useState<string>();
  const [key, setKey] = useState(cloudKey);
  const [model, setModel] = useState<Model>(() => (localStorage.getItem("cloud-model") === "pro" ? "pro" : "flash"));
  const [photos, setPhotos] = useState<Photo[]>();
  const canvases: Canvases = useRef({});
  const input = useRef<HTMLCanvasElement>(null);
  const worker = useRef<Worker>(undefined);

  useEffect(() => {
    finishSignIn()
      .then(() => setKey(cloudKey()))
      .catch((err: Error) => setError(err.message));
  }, []);

  // Photos cleaned on another device show up when this page comes back into view.
  useEffect(() => {
    if (!key) return;

    const refresh = () => {
      if (document.visibilityState === "visible") listPhotos(key).then(setPhotos, () => {});
    };

    refresh();
    document.addEventListener("visibilitychange", refresh);

    return () => document.removeEventListener("visibilitychange", refresh);
  }, [key]);

  const clean = useCallback(
    (file: File, kind: RawKind, finished?: Finished) => {
      worker.current?.terminate();
      const w = new Worker();
      worker.current = w;
      let lock: WakeLockSentinel | undefined;
      navigator.wakeLock
        ?.request("screen")
        .then((l) => (lock = l))
        .catch(() => {});
      let size = { width: 0, height: 0 };
      let onDevice = "on-device";

      const change = (to: (run: Run) => Partial<Run>) =>
        setJob((j) => {
          const run = j?.runs[kind];

          return j && run ? { ...j, runs: { ...j.runs, [kind]: { ...run, ...to(run) } } } : j;
        });

      w.addEventListener("message", ({ data }: MessageEvent<FromWorker>) => {
        switch (data.type) {
          case "start":
            size = { width: data.width, height: data.height };
            onDevice = data.raw ? "on-device-raw" : "on-device";
            setJob((j) => j && { ...j, width: data.width, height: data.height, raw: data.raw });
            change(() => ({ stage: "model" }));

            return;
          case "original":
            setJob((j) => {
              if (!j) return j;
              URL.revokeObjectURL(j.original);

              return { ...j, original: URL.createObjectURL(data.blob) };
            });

            return;
          case "model":
            change(() => ({ loaded: data.loaded, modelBytes: data.total }));

            return;
          case "progress":
            change((run) => ({
              stage: "clean",
              done: data.done,
              total: data.total,
              firstTileAt: data.done === 1 ? performance.now() : run.firstTileAt,
            }));

            return;
          case "tile":
          case "input":
            (data.type === "tile" ? canvases.current[kind] : input.current)
              ?.getContext("2d")
              ?.putImageData(new ImageData(data.pixels, data.w, data.h), data.x, data.y);

            return;
          case "done":
            lock?.release();
            change(() => ({ stage: "done", result: data.blob, band: data.band, finishedAt: performance.now() }));
            finished?.(data.blob, { model: onDevice, ...size });

            return;
          case "error":
            lock?.release();
            setError(data.message);

            if (kind !== "raw")
              setJob(
                (j) => j && { ...j, shown: j.shown === kind ? "raw" : j.shown, runs: { ...j.runs, [kind]: undefined } },
              );

            return;
        }
      });

      const order: ToWorker = { file, key, model, borrowed: borrowing(), kind };
      w.postMessage(order);
    },
    [key, model],
  );

  const open = useCallback(
    async (file: File) => {
      worker.current?.terminate();
      setError(undefined);
      const shown = (await rawPreview(file)) ?? file;
      const name = file.name.replace(/\.[^.]+$/, "") || "photo";
      setJob((old) => {
        if (old) URL.revokeObjectURL(old.original);

        return {
          name,
          file,
          original: URL.createObjectURL(shown),
          cloud: key && MODELS[model].name,
          width: 0,
          height: 0,
          shown: "raw",
          runs: { raw: fresh() },
        };
      });
      clean(file, "raw", (result, info) => {
        if (sharing()) share(file, result, { ...info, model: key ? MODELS[model].id : info.model }).catch(() => {});

        if (key) {
          setJob((j) => j && { ...j, saving: "saving" });
          savePhoto(key, name, shown, result).then(
            (photo) => {
              setPhotos((p) => [photo, ...(p ?? [])]);
              setJob((j) => j && { ...j, saving: "saved" });
            },
            () => setJob((j) => j && { ...j, saving: "failed" }),
          );
        }
      });
    },
    [key, model, clean],
  );

  const look = (kind: RawKind) => {
    setError(undefined);
    setJob((j) => j && { ...j, shown: kind, runs: { ...j.runs, [kind]: j.runs[kind] ?? fresh() } });

    if (job?.file && !job.runs[kind]) clean(job.file, kind);
  };

  const view = async (photo: Photo) => {
    worker.current?.terminate();
    setError(undefined);

    try {
      const result = await fetch(photo.result!).then((res) => res.blob());
      const cleaned = await createImageBitmap(result);
      setJob((old) => {
        if (old) URL.revokeObjectURL(old.original);

        return {
          name: photo.name,
          original: photo.original ?? photo.result!,
          width: cleaned.width,
          height: cleaned.height,
          shown: "raw",
          runs: { raw: { ...fresh(), stage: "done", done: 1, total: 1, result } },
          cleaned,
          savedAt: photo.at,
        };
      });
    } catch {
      setError("That photo couldn't be opened. Try again.");
    }
  };

  const reset = () => {
    worker.current?.terminate();

    if (job) URL.revokeObjectURL(job.original);
    setJob(undefined);
    setError(undefined);
  };

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = [...(e.clipboardData?.files ?? [])].find(isPhoto);

      if (file) open(file);
    };

    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const file = [...(e.dataTransfer?.files ?? [])].find(isPhoto);

      if (file) open(file);
    };

    const onDragOver = (e: DragEvent) => e.preventDefault();
    addEventListener("paste", onPaste);
    addEventListener("drop", onDrop);
    addEventListener("dragover", onDragOver);

    return () => {
      removeEventListener("paste", onPaste);
      removeEventListener("drop", onDrop);
      removeEventListener("dragover", onDragOver);
    };
  }, [open]);

  return (
    <div className="flex h-full flex-col pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]">
      <header className="flex h-14 shrink-0 items-center justify-between px-5">
        <button onClick={reset} className="flex items-center gap-2 text-[15px] font-medium tracking-tight">
          <img src="/favicon.svg" alt="" className="size-5" />
          denoise
        </button>
        <a
          href="https://github.com/AVGVSTVS96/denoise"
          className="text-[13px] text-muted transition-colors hover:text-fg"
        >
          GitHub
        </a>
      </header>
      <main className="flex min-h-0 flex-1 flex-col">
        {job ? (
          <Editor job={job} error={error} canvases={canvases} input={input} onLook={look} onNew={reset} />
        ) : (
          <Pick
            onFile={open}
            cloud={Boolean(key)}
            model={model}
            onModel={(m) => {
              localStorage.setItem("cloud-model", m);
              setModel(m);
            }}
            error={error}
            photos={photos}
            onPhoto={view}
            onCloud={() => {
              if (!key) return signIn();
              signOut();
              setPhotos(undefined);
              setKey(undefined);
            }}
          />
        )}
      </main>
    </div>
  );
}

function Pick({
  onFile,
  cloud,
  model,
  onModel,
  error,
  photos,
  onPhoto,
  onCloud,
}: {
  onFile: (f: File) => void;
  cloud: boolean;
  model: Model;
  onModel: (m: Model) => void;
  error?: string;
  photos?: Photo[];
  onPhoto: (p: Photo) => void;
  onCloud: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [train, setTrain] = useState(sharing);
  const [borrowed, setBorrowed] = useState(borrowing);

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="flex min-h-[26rem] flex-1 items-center justify-center p-5">
        <button
          onClick={() => input.current?.click()}
          className="group flex w-full max-w-md flex-col items-center gap-5 rounded-2xl border border-line bg-card px-8 py-14 text-center shadow-[0_1px_2px_rgb(0_0_0/0.04)] transition hover:border-muted/40"
        >
          <svg viewBox="0 0 56 56" className="size-14 rounded-xl" aria-hidden>
            <defs>
              <linearGradient id="mark" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="#6366f1" />
                <stop offset="1" stopColor="#22d3ee" />
              </linearGradient>
              <filter id="grain">
                <feTurbulence type="fractalNoise" baseFrequency="1.1" numOctaves="2" seed="4" />
                <feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1.6 -0.55" />
              </filter>
              <clipPath id="noisy">
                <path d="M0 0H56L0 56Z" />
              </clipPath>
            </defs>
            <rect width="56" height="56" fill="url(#mark)" />
            <rect width="56" height="56" filter="url(#grain)" clipPath="url(#noisy)" opacity="0.7" />
            <path d="M56 0L0 56" stroke="white" strokeOpacity="0.9" strokeWidth="1.5" />
          </svg>
          <div className="space-y-1.5">
            <h1 className="text-xl font-semibold tracking-tight">Clean up a noisy photo</h1>
            <p className="text-sm text-balance text-muted">
              {cloud
                ? `Cleaned at full size by ${MODELS[model].name} on your OpenRouter account, about $${(MODELS[model].perTile * 24).toFixed(2)} for a 16 MP photo.`
                : `High-ISO shots, cleaned on your device.${train ? "" : " Nothing is uploaded."}`}
            </p>
          </div>
          <span className="rounded-full bg-fg px-4 py-2 text-sm font-medium text-bg transition group-hover:opacity-90">
            Choose photo
          </span>
          <input
            ref={input}
            type="file"
            accept="image/*,.dng"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];

              if (f) onFile(f);
              e.target.value = "";
            }}
          />
        </button>
      </div>
      {cloud && photos && photos.length > 0 && (
        <section className="mx-auto w-full max-w-3xl px-5 pb-2">
          <h2 className="mb-2.5 text-[13px] font-medium text-muted">Your photos</h2>
          <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4 md:grid-cols-5">
            {photos.map((photo) => (
              <button
                key={photo.id}
                onClick={() => onPhoto(photo)}
                title={photo.name}
                className="aspect-square overflow-hidden rounded-lg bg-line transition hover:opacity-90"
              >
                <img src={photo.thumb} alt={photo.name} loading="lazy" className="size-full object-cover" />
              </button>
            ))}
          </div>
        </section>
      )}
      <div className="shrink-0 space-y-2 px-6 py-6 text-center text-xs text-muted">
        {error && <p className="text-red-500">{error}</p>}
        {cloud && (
          <div className="flex justify-center gap-1">
            {CHOICES.map((m) => (
              <button
                key={m}
                onClick={() => onModel(m)}
                className={`rounded-full px-3 py-1.5 pointer-coarse:py-2 ${m === model ? "bg-fg text-bg" : "border border-line bg-card hover:text-fg"}`}
              >
                {MODELS[m].name}
              </button>
            ))}
          </div>
        )}
        <button onClick={onCloud} className="underline-offset-2 hover:text-fg hover:underline">
          {cloud ? "Cloud on · Sign out of OpenRouter" : "Clean in the cloud: sign in with OpenRouter"}
        </button>
        <button
          onClick={() => {
            setSharing(!train);
            setTrain(!train);
          }}
          className="block w-full underline-offset-2 hover:text-fg hover:underline"
        >
          {train
            ? "Photos you clean help train denoise · Turn off"
            : "Photos you clean stay private · Share for training"}
        </button>
        {!cloud && (
          <button
            onClick={() => {
              setBorrowing(!borrowed);
              setBorrowed(!borrowed);
            }}
            className="block w-full underline-offset-2 hover:text-fg hover:underline"
          >
            {borrowed
              ? "Black offsets from another body of your camera: on · Turn off"
              : "Black offsets from another body of your camera: off · Turn on"}
          </button>
        )}
        {!cloud && !hasWebGPU && <p>This browser has no WebGPU. Use Safari 26+ or Chrome.</p>}
      </div>
    </div>
  );
}

function Editor({
  job,
  error,
  canvases,
  input,
  onLook,
  onNew,
}: {
  job: Job;
  error?: string;
  canvases: Canvases;
  input: React.RefObject<HTMLCanvasElement | null>;
  onLook: (kind: RawKind) => void;
  onNew: () => void;
}) {
  const [split, setSplit] = useState(50);
  const [strength, setStrength] = useState(() => Number(localStorage.getItem("strength") ?? 100));
  const [fine, setFine] = useState(() => Number(localStorage.getItem("detail") ?? 50));
  const [actual, setActual] = useState(false);
  const [view, setView] = useState({ scroll: 0, width: 1, offset: 0, image: 1 });
  const frame = useRef<HTMLDivElement>(null);
  const image = useRef<HTMLDivElement>(null);
  const details: Canvases = useRef({});
  const { shown } = job;
  const run = job.runs[shown]!;
  const done = run.stage === "done";
  const band = run.band ?? 0;
  const finer = band > 0 ? fine : 0;
  const busy = LOOKS.some((kind) => job.runs[kind] && job.runs[kind].stage !== "done");

  useEffect(() => {
    if (job.cleaned) canvases.current.raw?.getContext("2d")?.drawImage(job.cleaned, 0, 0);
  }, [job.cleaned, canvases]);

  useEffect(() => {
    if (done && finer > 0)
      withDetail(input.current!, canvases.current[shown]!, details.current[shown]!, band).catch(() => {});
  }, [done, finer, band, shown, input, canvases]);

  const measure = useCallback(() => {
    if (!frame.current || !image.current) return;
    setView({
      scroll: frame.current.scrollLeft,
      width: frame.current.clientWidth,
      offset: image.current.offsetLeft,
      image: image.current.offsetWidth,
    });
  }, []);

  useEffect(() => {
    const el = frame.current;

    if (!actual || !el) return;
    el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
    el.scrollTop = (el.scrollHeight - el.clientHeight) / 2;
    measure();
    addEventListener("resize", measure);

    return () => removeEventListener("resize", measure);
  }, [actual, measure]);

  useEffect(() => {
    const prefix = {
      decode: "",
      model: run.loaded ? `${Math.round((run.loaded / run.modelBytes) * 100)}% model · ` : "",
      clean: `${Math.round((run.done / run.total) * 100)}% · `,
      done: "Done · ",
    }[run.stage];

    document.title = `${prefix}denoise`;

    return () => {
      document.title = "denoise";
    };
  }, [run.stage, run.done, run.total, run.loaded, run.modelBytes]);

  const dragOver = (e: React.PointerEvent, box: Element) => {
    if (!done || (e.type === "pointermove" && e.buttons === 0)) return;
    const r = box.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)));
  };

  const ratio = job.width && job.height ? job.width / job.height : 3 / 2;
  const clip = actual ? ((view.scroll + (split / 100) * view.width - view.offset) / view.image) * 100 : split;

  return (
    <>
      <div className="relative flex min-h-0 flex-1">
        <div
          ref={frame}
          onScroll={actual ? measure : undefined}
          className={`relative flex min-w-0 flex-1 ${actual ? "overflow-auto [scrollbar-width:none]" : "items-center justify-center px-5"}`}
        >
          <div
            ref={image}
            onPointerDown={(e) => {
              if (actual) return;
              e.currentTarget.setPointerCapture(e.pointerId);
              dragOver(e, e.currentTarget);
            }}
            onPointerMove={(e) => !actual && dragOver(e, e.currentTarget)}
            style={{
              width: actual ? job.width / devicePixelRatio : `min(100%, calc((100dvh - 11rem) * ${ratio}))`,
              aspectRatio: ratio,
            }}
            className={`relative m-auto shrink-0 overflow-hidden bg-line select-none ${actual ? "" : "rounded-lg"} ${done && !actual ? "cursor-ew-resize touch-none" : ""}`}
          >
            <img src={job.original} alt="" draggable={false} className="absolute inset-0 size-full" />
            <div
              className="absolute inset-0"
              style={done ? { clipPath: `inset(0 0 0 ${Math.min(100, Math.max(0, clip))}%)` } : undefined}
            >
              {job.raw && (
                <canvas ref={input} width={job.width} height={job.height} className="absolute inset-0 size-full" />
              )}
              {LOOKS.map(
                (kind) =>
                  job.runs[kind] && (
                    <div
                      key={kind}
                      hidden={kind !== shown}
                      className="absolute inset-0"
                      style={job.raw ? { opacity: strength / 100 } : undefined}
                    >
                      <canvas
                        ref={(made) => {
                          canvases.current[kind] = made;
                        }}
                        width={job.width}
                        height={job.height}
                        className="absolute inset-0 size-full"
                      />
                      {(job.runs[kind].band ?? 0) > 0 && (
                        <canvas
                          ref={(made) => {
                            details.current[kind] = made;
                          }}
                          width={job.width}
                          height={job.height}
                          className="absolute inset-0 size-full"
                          style={{ opacity: fine / 100 }}
                        />
                      )}
                    </div>
                  ),
              )}
            </div>
            {done && !actual && <Divider split={split} />}
          </div>
        </div>
        {done && actual && (
          <Divider
            split={split}
            onHandle={(e) => {
              if (e.type === "pointerdown") e.currentTarget.setPointerCapture(e.pointerId);
              dragOver(e, frame.current!);
            }}
          />
        )}
      </div>
      <footer className="relative shrink-0 px-5 py-3">
        <Progress run={run} error={error} />
        <div className="flex flex-wrap items-center gap-x-2 gap-y-3">
          <p className="min-w-0 flex-1 truncate text-[13px] text-muted tabular-nums">
            {error ? <span className="text-red-500">{error}</span> : status(job, run)}
          </p>
          {job.raw && job.runs.raw?.stage === "done" && (
            <div className="order-first flex basis-full sm:order-none sm:mr-2 sm:basis-auto">
              <div className="flex rounded-full border border-line bg-card p-0.5 text-[13px] font-medium pointer-coarse:text-sm">
                {LOOKS.map((kind) => (
                  <button
                    key={kind}
                    onClick={() => onLook(kind)}
                    disabled={busy && !job.runs[kind]}
                    aria-pressed={kind === shown}
                    className={`h-6.5 rounded-full px-3 transition disabled:opacity-40 pointer-coarse:h-8.5 pointer-coarse:px-4 ${kind === shown ? "bg-line" : "text-muted enabled:hover:text-fg"}`}
                  >
                    {LOOK[kind]}
                  </button>
                ))}
              </div>
            </div>
          )}
          {done && job.raw && (
            <>
              <Dial name="Strength" value={strength} onChange={setStrength} />
              {band > 0 && <Dial name="Detail" value={fine} onChange={setFine} />}
            </>
          )}
          <Button onClick={() => setActual((a) => !a)} disabled={!job.width}>
            {actual ? "Fit" : "1:1"}
          </Button>
          <Button onClick={onNew}>New</Button>
          {done && run.result && (
            <Button
              primary
              onClick={async () =>
                save(
                  job.raw && (strength < 100 || finer > 0)
                    ? await blended(
                        input.current!,
                        canvases.current[shown]!,
                        details.current[shown],
                        strength,
                        finer,
                        band,
                      )
                    : run.result!,
                  `${job.name}-denoised.jpg`,
                )
              }
            >
              Save
            </Button>
          )}
        </div>
      </footer>
    </>
  );
}

function Dial({ name, value, onChange }: { name: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="order-first flex basis-full items-center gap-3 text-[13px] text-muted tabular-nums sm:order-none sm:mr-2 sm:basis-auto">
      <span className="w-[8ch] sm:w-auto">{name}</span>
      <input
        type="range"
        value={value}
        onChange={(e) => {
          localStorage.setItem(name.toLowerCase(), e.target.value);
          onChange(e.target.valueAsNumber);
        }}
        className="strength min-w-0 flex-1 sm:w-28 sm:flex-none"
      />
      <span className="w-[4ch] text-right">{value}%</span>
    </label>
  );
}

function Divider({ split, onHandle }: { split: number; onHandle?: (e: React.PointerEvent<HTMLDivElement>) => void }) {
  return (
    <div className="pointer-events-none absolute inset-0">
      <div
        className="absolute inset-y-0 w-px bg-white/90 shadow-[0_0_8px_rgb(0_0_0/0.5)]"
        style={{ left: `${split}%` }}
      />
      <div
        onPointerDown={onHandle}
        onPointerMove={onHandle}
        className={`absolute top-1/2 grid size-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-black shadow-lg pointer-coarse:size-10 ${onHandle ? "pointer-events-auto cursor-ew-resize touch-none" : ""}`}
        style={{ left: `${split}%` }}
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M5 3.5 1.5 7 5 10.5M9 3.5 12.5 7 9 10.5" />
        </svg>
      </div>
      <Tag className={`left-2.5 ${split < 15 ? "opacity-0" : ""}`}>Before</Tag>
      <Tag className={`right-2.5 ${split > 85 ? "opacity-0" : ""}`}>After</Tag>
    </div>
  );
}

function Tag({ children, className }: { children: React.ReactNode; className: string }) {
  return (
    <span
      className={`absolute top-2.5 rounded-md bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white backdrop-blur transition-opacity ${className}`}
    >
      {children}
    </span>
  );
}

function Button({ primary, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { primary?: boolean }) {
  return (
    <button
      {...props}
      className={`h-8 shrink-0 rounded-full px-3.5 text-[13px] pointer-coarse:h-10 pointer-coarse:px-4 pointer-coarse:text-sm font-medium transition disabled:opacity-40 ${
        primary ? "bg-fg text-bg hover:opacity-90" : "border border-line bg-card hover:border-muted/40"
      }`}
    />
  );
}

function Progress({ run, error }: { run: Run; error?: string }) {
  const value =
    run.stage === "done"
      ? 1
      : run.stage === "clean"
        ? run.done / run.total
        : run.stage === "model"
          ? (run.loaded / run.modelBytes) * 0.999
          : 0;

  return (
    <div
      className={`absolute inset-x-5 top-0 h-1 overflow-hidden rounded-full bg-line transition-opacity delay-700 duration-700 ${run.stage === "done" ? "opacity-0" : ""}`}
    >
      <div
        className={`h-full rounded-full transition-[width] duration-500 ${error ? "bg-red-500" : "bg-gradient-to-r from-indigo-500 to-cyan-400"}`}
        style={{ width: `${(run.stage === "model" && run.loaded === 0 ? 0 : value) * 100}%` }}
      />
    </div>
  );
}

function status(job: Job, run: Run) {
  const megapixels = (job.width * job.height) / 1e6;
  const mp = `${megapixels < 10 ? megapixels.toFixed(1) : Math.round(megapixels)} MP`;

  switch (run.stage) {
    case "decode":
      return "Opening photo…";
    case "model":
      if (job.cloud) return `Sending to ${job.cloud}…`;

      return run.loaded > 0 && run.loaded < run.modelBytes
        ? `Model · ${mb(run.loaded)} / ${mb(run.modelBytes)} MB`
        : "Starting the GPU…";
    case "clean": {
      const left =
        run.firstTileAt && run.done > 1
          ? ((performance.now() - run.firstTileAt) / (run.done - 1)) * (run.total - run.done)
          : undefined;

      return `Cleaning · ${Math.round((run.done / run.total) * 100)}%${left ? ` · ${eta(left)} left` : ""}`;
    }

    case "done": {
      if (job.savedAt) return `Saved ${new Date(job.savedAt).toLocaleDateString()} · ${mp}`;
      const saving = { saving: " · Saving to your photos…", saved: " · In your photos", failed: " · Not saved" };

      return `Done · ${eta(run.finishedAt! - run.startedAt)} · ${mp}${job.saving ? saving[job.saving] : ""}`;
    }
  }
}

const mb = (b: number) => Math.round(b / 1e6);

function eta(ms: number) {
  const s = Math.max(1, Math.round(ms / 1000));

  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

const STRIP = 256;
const filling = new WeakMap<HTMLCanvasElement, Promise<void>>();

async function fill(under: HTMLCanvasElement, over: HTMLCanvasElement, onto: HTMLCanvasElement, band: number) {
  const { width, height } = over;
  const [shot, made, drawn] = [under, over, onto].map((canvas) => canvas.getContext("2d")!);

  for (let y = 0; y < height && onto.isConnected; y += STRIP) {
    const top = Math.max(0, y - MARGIN);
    const rows = Math.min(height, y + STRIP + MARGIN) - top;
    const result = made.getImageData(0, top, width, rows).data;
    const pixels = detailed(result, shot.getImageData(0, top, width, rows).data, width, band);
    drawn.putImageData(new ImageData(pixels, width, rows), 0, top, 0, y - top, width, STRIP);
    await new Promise((next) => setTimeout(next));
  }
}

function withDetail(under: HTMLCanvasElement, over: HTMLCanvasElement, onto: HTMLCanvasElement, band: number) {
  const filled = filling.get(onto) ?? fill(under, over, onto, band);
  filling.set(onto, filled);

  return filled;
}

async function blended(
  under: HTMLCanvasElement,
  over: HTMLCanvasElement,
  onto: HTMLCanvasElement | null | undefined,
  strength: number,
  fine: number,
  band: number,
) {
  const mixed = new OffscreenCanvas(over.width, over.height);
  const ctx = mixed.getContext("2d")!;
  ctx.drawImage(over, 0, 0);

  if (onto && fine > 0) {
    await withDetail(under, over, onto, band);
    ctx.globalAlpha = fine / 100;
    ctx.drawImage(onto, 0, 0);
  }

  ctx.globalAlpha = 1 - strength / 100;
  ctx.drawImage(under, 0, 0);

  return mixed.convertToBlob({ type: "image/jpeg", quality: 0.95 });
}

async function save(blob: Blob, filename: string) {
  const file = new File([blob], filename, { type: "image/jpeg" });

  if (matchMedia("(pointer: coarse)").matches && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });

      return;
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
    }
  }

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
