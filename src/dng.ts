import { losslessJpeg } from "./ljpeg";

type Field = { type: number; count: number; at: number };

type Directory = Map<number, Field>;

type Tiff = { data: DataView; little: boolean; directories: Directory[] };

export type Raw = {
  samples: Uint16Array;
  stride: number;
  top: number;
  left: number;
  width: number;
  height: number;
  black: number[];
  white: number;
  crop: { x: number; y: number; width: number; height: number };
  orientation: number;
  balance: number[];
  toSrgb: number[];
  exposure: number;
  body: string;
  iso: number;
};

const SIZES = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4];
const SUB_DIRECTORIES = [330, 34665];
const CFA = 32803;
const D65 = 21;
const UNKNOWN_ISO = [0, 65535];

const XYZ_FROM_SRGB = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.072175],
  [0.0193339, 0.119192, 0.9503041],
];

const unreadable = () => new Error("This raw file packs its sensor data in a way this app can't read.");

async function open(file: Blob): Promise<Tiff | undefined> {
  const head = new DataView(await file.slice(0, 8).arrayBuffer());
  const order = head.byteLength === 8 ? head.getUint16(0) : 0;

  if (order !== 0x4949 && order !== 0x4d4d) return;
  const little = order === 0x4949;

  if (head.getUint16(2, little) !== 42) return;
  const data = new DataView(await file.arrayBuffer());
  const tiff: Tiff = { data, little, directories: [] };
  const seen = new Set<number>();
  const queue = [data.getUint32(4, little)];

  while (queue.length) {
    const at = queue.shift()!;

    if (!at || seen.has(at) || at + 2 > data.byteLength) continue;
    seen.add(at);
    const count = data.getUint16(at, little);
    const directory: Directory = new Map();

    for (let i = 0; i < count; i++) {
      const entry = at + 2 + i * 12;
      const type = data.getUint16(entry + 2, little);
      const n = data.getUint32(entry + 4, little);
      const inline = (SIZES[type] ?? 1) * n <= 4;
      directory.set(data.getUint16(entry, little), {
        type,
        count: n,
        at: inline ? entry + 8 : data.getUint32(entry + 8, little),
      });
    }

    tiff.directories.push(directory);

    for (const tag of SUB_DIRECTORIES) queue.push(...numbers(tiff, directory, tag));
    queue.push(data.getUint32(at + 2 + count * 12, little));
  }

  return tiff;
}

function numbers({ data, little }: Tiff, directory: Directory, tag: number): number[] {
  const field = directory.get(tag);

  if (!field || field.at + (SIZES[field.type] ?? 1) * field.count > data.byteLength) return [];

  return Array.from({ length: field.count }, (_, i) => {
    const at = field.at + i * (SIZES[field.type] ?? 1);

    switch (field.type) {
      case 3:
        return data.getUint16(at, little);
      case 4:
      case 13:
        return data.getUint32(at, little);
      case 5:
        return data.getUint32(at, little) / data.getUint32(at + 4, little);
      case 6:
        return data.getInt8(at);
      case 8:
        return data.getInt16(at, little);
      case 9:
        return data.getInt32(at, little);
      case 10:
        return data.getInt32(at, little) / data.getInt32(at + 4, little);
      case 11:
        return data.getFloat32(at, little);
      case 12:
        return data.getFloat64(at, little);
      default:
        return data.getUint8(at);
    }
  });
}

const text = (tiff: Tiff, directory: Directory | undefined, tag: number) =>
  directory
    ? String.fromCharCode(...numbers(tiff, directory, tag))
        .replace(/\0.*$/, "")
        .trim()
    : "";

export async function rawPreview(file: Blob): Promise<Blob | undefined> {
  const tiff = await open(file);
  let best: { offset: number; length: number } | undefined;

  for (const directory of tiff?.directories ?? []) {
    const first = (...tags: number[]) => tags.map((tag) => numbers(tiff!, directory, tag)[0]).find(Boolean);
    const offset = first(513, 273);
    const length = first(514, 279);
    const compression = first(259);

    if (
      offset &&
      length &&
      (compression === 6 || compression === 7) &&
      first(254) === 1 &&
      length > (best?.length ?? 0)
    )
      best = { offset, length };
  }

  return best && file.slice(best.offset, best.offset + best.length, "image/jpeg");
}

function unpacked(bytes: Uint8Array, bits: number, little: boolean, out: Uint16Array, row: number) {
  const rows = Math.floor(out.length / row);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bits === 16) {
    for (let i = 0; i < rows * row; i++) out[i] = view.getUint16(2 * i, little);

    return;
  }

  const stride = Math.ceil((row * bits) / 8);

  for (let y = 0; y < rows; y++) {
    let at = y * stride;
    let buffer = 0;
    let held = 0;

    for (let x = 0; x < row; x++) {
      while (held < bits) {
        buffer = ((buffer << 8) | bytes[at++]) & 0xffffff;
        held += 8;
      }

      held -= bits;
      out[y * row + x] = (buffer >>> held) & ((1 << bits) - 1);
    }
  }
}

function sensor(tiff: Tiff, directory: Directory) {
  const one = (tag: number, missing = 0) => numbers(tiff, directory, tag)[0] ?? missing;
  const width = one(256);
  const height = one(257);
  const bits = one(258);
  const compression = one(259);
  const tiled = directory.has(324);
  const offsets = numbers(tiff, directory, tiled ? 324 : 273);
  const lengths = numbers(tiff, directory, tiled ? 325 : 279);
  const across = tiled ? one(322) : width;
  const down = tiled ? one(323) : Math.min(one(278, height), height);

  if ((compression !== 1 && compression !== 7) || one(277, 1) !== 1 || bits < 8 || bits > 16) throw unreadable();
  const samples = new Uint16Array(width * height);
  const columns = Math.ceil(width / across);

  offsets.forEach((offset, i) => {
    const x = (i % columns) * across;
    const y = Math.floor(i / columns) * down;
    const rows = Math.min(down, height - y);
    const bytes = new Uint8Array(tiff.data.buffer, offset, lengths[i]);
    const inPlace = across === width && y + down <= height;
    const piece = inPlace ? samples.subarray(y * width, (y + down) * width) : new Uint16Array(across * down);

    if (compression === 1) unpacked(bytes, bits, tiff.little, piece.subarray(0, across * rows), across);
    else if (losslessJpeg(bytes, piece).row !== across) throw unreadable();

    if (!inPlace)
      for (let r = 0; r < rows; r++)
        samples.set(piece.subarray(r * across, r * across + Math.min(across, width - x)), (y + r) * width + x);
  });

  const curve = numbers(tiff, directory, 50712);

  if (curve.length) for (let i = 0; i < samples.length; i++) samples[i] = curve[Math.min(samples[i], curve.length - 1)];

  return { samples, width, height, bits };
}

const inverse = (m: number[][]) => {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
};

function colour(tiff: Tiff, main: Directory) {
  const flat = numbers(tiff, main, main.has(50722) && numbers(tiff, main, 50778)[0] !== D65 ? 50722 : 50721);

  if (flat.length !== 9) throw unreadable();
  const fromXyz = [flat.slice(0, 3), flat.slice(3, 6), flat.slice(6, 9)];

  const fromSrgb = fromXyz.map((row) => {
    const mixed = [0, 1, 2].map((k) => row.reduce((sum, part, j) => sum + part * XYZ_FROM_SRGB[j][k], 0));
    const sum = mixed[0] + mixed[1] + mixed[2];

    return mixed.map((v) => v / sum);
  });

  const neutral = numbers(tiff, main, 50728);
  const balance = neutral.length === 3 ? neutral.map((v) => neutral[1] / v) : [1, 1, 1];

  return { balance, toSrgb: inverse(fromSrgb).flat() };
}

export async function rawPhoto(file: Blob): Promise<Raw | undefined> {
  const tiff = await open(file);
  const main = tiff?.directories[0];
  const directory = tiff?.directories.find((d) => numbers(tiff, d, 262)[0] === CFA && !numbers(tiff, d, 254)[0]);

  if (!tiff || !main || !directory || !main.has(50706)) return;
  const pattern = numbers(tiff, directory, 33422);
  const red = pattern.indexOf(0);

  if (numbers(tiff, directory, 33421).join() !== "2,2" || red < 0 || pattern[3 - red] !== 2) return;
  const { samples, width, height, bits } = sensor(tiff, directory);
  const declared = numbers(tiff, directory, 50829);
  const area = declared[2] > declared[0] && declared[3] > declared[1] ? declared : [0, 0, height, width];
  const top = area[0] + (red >> 1);
  const left = area[1] + (red & 1);
  const repeat = directory.has(50713) ? numbers(tiff, directory, 50713) : [1, 1];
  const levels = directory.has(50714) ? numbers(tiff, directory, 50714) : [0];

  if (repeat[0] > 2 || repeat[1] > 2) throw unreadable();

  const black = [0, 1, 2, 3].map((site) => {
    const y = ((red >> 1) + (site >> 1)) % repeat[0];
    const x = ((red & 1) + (site & 1)) % repeat[1];

    return levels[y * repeat[1] + x];
  });

  const origin = directory.has(50719) ? numbers(tiff, directory, 50719) : [0, 0];
  const shown = { width: (area[3] - left) & ~1, height: (area[2] - top) & ~1 };
  const size = directory.has(50720) ? numbers(tiff, directory, 50720) : [shown.width, shown.height];
  const x = Math.max(0, Math.round(origin[0]) - (red & 1));
  const y = Math.max(0, Math.round(origin[1]) - (red >> 1));
  const exif = tiff.directories.find((d) => d.has(34855) || d.has(34865) || d.has(34866));
  const speeds = exif ? [34866, 34865, 34855].map((tag) => numbers(tiff, exif, tag)[0] ?? 0) : [];
  const serial = tiff.directories.find((d) => d.has(42033));

  return {
    samples,
    stride: width,
    top,
    left,
    ...shown,
    black,
    white: numbers(tiff, directory, 50717)[0] ?? (1 << bits) - 1,
    crop: {
      x,
      y,
      width: Math.min(Math.round(size[0]), shown.width - x),
      height: Math.min(Math.round(size[1]), shown.height - y),
    },
    orientation: numbers(tiff, main, 274)[0] ?? 1,
    ...colour(tiff, main),
    exposure: numbers(tiff, main, 50730)[0] ?? 0,
    body: `${text(tiff, main, 272)} ${text(tiff, serial, 42033)}`.trim(),
    iso: speeds.find((speed) => !UNKNOWN_ISO.includes(speed)) ?? 0,
  };
}
