import { useCallback, useEffect, useRef, useState } from "react";
import { modelBytes } from "./model";
import type { FromWorker } from "./worker";
import Worker from "./worker?worker";

type Job = {
  name: string;
  original: string;
  width: number;
  height: number;
  scaled: boolean;
  stage: "decode" | "model" | "clean" | "done";
  loaded: number;
  done: number;
  total: number;
  startedAt: number;
  firstTileAt?: number;
  finishedAt?: number;
  result?: Blob;
};

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

export function App() {
  const [job, setJob] = useState<Job>();
  const [error, setError] = useState<string>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const worker = useRef<Worker>(undefined);

  const open = useCallback((file: File) => {
    worker.current?.terminate();
    setError(undefined);
    setJob((old) => {
      if (old) URL.revokeObjectURL(old.original);

      return {
        name: file.name.replace(/\.[^.]+$/, "") || "photo",
        original: URL.createObjectURL(file),
        width: 0,
        height: 0,
        scaled: false,
        stage: "decode",
        loaded: 0,
        done: 0,
        total: 0,
        startedAt: performance.now(),
      };
    });
    const w = new Worker();
    worker.current = w;
    let lock: WakeLockSentinel | undefined;
    navigator.wakeLock
      ?.request("screen")
      .then((l) => (lock = l))
      .catch(() => {});
    w.addEventListener("message", ({ data }: MessageEvent<FromWorker>) => {
      switch (data.type) {
        case "start":
          setJob((j) => j && { ...j, width: data.width, height: data.height, scaled: data.scaled, stage: "model" });

          return;
        case "model":
          setJob((j) => j && { ...j, loaded: data.loaded });

          return;
        case "progress":
          setJob((j) => {
            if (!j) return j;
            const now = performance.now();

            return {
              ...j,
              stage: "clean",
              done: data.done,
              total: data.total,
              firstTileAt: data.done === 1 ? now : j.firstTileAt,
            };
          });

          return;
        case "tile":
          canvas.current?.getContext("2d")?.putImageData(new ImageData(data.pixels, data.w, data.h), data.x, data.y);

          return;
        case "done":
          lock?.release();
          setJob((j) => j && { ...j, stage: "done", result: data.blob, finishedAt: performance.now() });

          return;
        case "error":
          lock?.release();
          setError(data.message);

          return;
      }
    });

    w.postMessage({ file });
  }, []);

  const reset = () => {
    worker.current?.terminate();

    if (job) URL.revokeObjectURL(job.original);
    setJob(undefined);
    setError(undefined);
  };

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/"));

      if (file) open(file);
    };

    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const file = [...(e.dataTransfer?.files ?? [])].find((f) => f.type.startsWith("image/"));

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
        {job ? <Editor job={job} error={error} canvas={canvas} onNew={reset} /> : <Pick onFile={open} />}
      </main>
    </div>
  );
}

function Pick({ onFile }: { onFile: (f: File) => void }) {
  const input = useRef<HTMLInputElement>(null);

  return (
    <div className="flex flex-1 items-center justify-center p-5">
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
            High-ISO shots, cleaned on your device. Nothing is uploaded.
          </p>
        </div>
        <span className="rounded-full bg-fg px-4 py-2 text-sm font-medium text-bg transition group-hover:opacity-90">
          Choose photo
        </span>
        <input
          ref={input}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];

            if (f) onFile(f);
            e.target.value = "";
          }}
        />
      </button>
      {!hasWebGPU && (
        <p className="fixed inset-x-0 bottom-[max(1.5rem,env(safe-area-inset-bottom))] px-6 text-center text-xs text-muted">
          This browser has no WebGPU. Use Safari 26+ or Chrome.
        </p>
      )}
    </div>
  );
}

function Editor({
  job,
  error,
  canvas,
  onNew,
}: {
  job: Job;
  error?: string;
  canvas: React.RefObject<HTMLCanvasElement | null>;
  onNew: () => void;
}) {
  const [split, setSplit] = useState(50);
  const [actual, setActual] = useState(false);
  const frame = useRef<HTMLDivElement>(null);
  const done = job.stage === "done";

  const drag = (e: React.PointerEvent) => {
    if (!done || (e.type === "pointermove" && e.buttons === 0)) return;
    const r = e.currentTarget.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)));
  };

  useEffect(() => {
    if (!actual || !frame.current) return;
    const el = frame.current;
    el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
    el.scrollTop = (el.scrollHeight - el.clientHeight) / 2;
  }, [actual]);

  const ratio = job.width && job.height ? job.width / job.height : 3 / 2;

  const size = actual
    ? { width: job.width / devicePixelRatio }
    : { width: `min(100%, calc((100dvh - 11rem) * ${ratio}))` };

  return (
    <>
      <div
        ref={frame}
        className={`flex min-h-0 flex-1 px-5 ${actual ? "overflow-auto" : "items-center justify-center"}`}
      >
        <div
          onPointerDown={(e) => {
            if (!actual) e.currentTarget.setPointerCapture(e.pointerId);
            drag(e);
          }}
          onPointerMove={drag}
          style={{ ...size, aspectRatio: ratio }}
          className={`relative m-auto shrink-0 overflow-hidden rounded-lg bg-line select-none ${done && !actual ? "cursor-ew-resize touch-none" : ""}`}
        >
          <img src={job.original} alt="" draggable={false} className="absolute inset-0 size-full" />
          <canvas
            ref={canvas}
            width={job.width}
            height={job.height}
            className="absolute inset-0 size-full"
            style={done ? { clipPath: `inset(0 0 0 ${split}%)` } : undefined}
          />
          {done && (
            <>
              <div
                className="pointer-events-none absolute inset-y-0 w-px bg-white/90 shadow-[0_0_8px_rgb(0_0_0/0.5)]"
                style={{ left: `${split}%` }}
              />
              <div
                className="pointer-events-none absolute top-1/2 grid size-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-black shadow-lg"
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
              <Tag className="left-2.5">Before</Tag>
              <Tag className="right-2.5">After</Tag>
            </>
          )}
        </div>
      </div>
      <footer className="shrink-0 px-5 pt-3 pb-4">
        <Progress job={job} error={error} />
        <div className="mt-3 flex items-center gap-2">
          <p className="min-w-0 flex-1 truncate text-[13px] text-muted tabular-nums">
            {error ? <span className="text-red-500">{error}</span> : status(job)}
          </p>
          <Button onClick={() => setActual((a) => !a)} disabled={!job.width}>
            {actual ? "Fit" : "1:1"}
          </Button>
          <Button onClick={onNew}>New</Button>
          {done && job.result && (
            <Button primary onClick={() => save(job.result!, `${job.name}-denoised.jpg`)}>
              Save
            </Button>
          )}
        </div>
      </footer>
    </>
  );
}

function Tag({ children, className }: { children: React.ReactNode; className: string }) {
  return (
    <span
      className={`pointer-events-none absolute top-2.5 rounded-md bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white backdrop-blur ${className}`}
    >
      {children}
    </span>
  );
}

function Button({ primary, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { primary?: boolean }) {
  return (
    <button
      {...props}
      className={`h-8 shrink-0 rounded-full px-3.5 text-[13px] font-medium transition disabled:opacity-40 ${
        primary ? "bg-fg text-bg hover:opacity-90" : "border border-line bg-card hover:border-muted/40"
      }`}
    />
  );
}

function Progress({ job, error }: { job: Job; error?: string }) {
  const value =
    job.stage === "done"
      ? 1
      : job.stage === "clean"
        ? job.done / job.total
        : job.stage === "model"
          ? (job.loaded / modelBytes) * 0.999
          : 0;

  return (
    <div className="h-1 overflow-hidden rounded-full bg-line">
      <div
        className={`h-full rounded-full transition-[width] duration-500 ${error ? "bg-red-500" : "bg-gradient-to-r from-indigo-500 to-cyan-400"}`}
        style={{ width: `${(job.stage === "model" && job.loaded === 0 ? 0 : value) * 100}%` }}
      />
    </div>
  );
}

function status(job: Job) {
  const mp = `${((job.width * job.height) / 1e6).toFixed(1)} MP`;

  switch (job.stage) {
    case "decode":
      return "Opening photo…";
    case "model":
      return job.loaded > 0 && job.loaded < modelBytes
        ? `Downloading model · ${mb(job.loaded)} of ${mb(modelBytes)} MB (once)`
        : "Starting the GPU…";
    case "clean": {
      const left =
        job.firstTileAt && job.done > 1
          ? ((performance.now() - job.firstTileAt) / (job.done - 1)) * (job.total - job.done)
          : undefined;

      return `Cleaning ${mp} · ${Math.round((job.done / job.total) * 100)}%${left ? ` · ${eta(left)} left` : ""}`;
    }

    case "done":
      return `Done in ${eta(job.finishedAt! - job.startedAt)} · ${mp}${job.scaled ? " (resized)" : ""}`;
  }
}

const mb = (b: number) => Math.round(b / 1e6);

function eta(ms: number) {
  const s = Math.max(1, Math.round(ms / 1000));

  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
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
