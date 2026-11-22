import type { Raw } from "./dng";

export type Network = (planes: Float32Array, height: number, width: number) => Promise<Float32Array>;

export type Piece = {
  x: number;
  y: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray<ArrayBuffer>;
  input: Uint8ClampedArray<ArrayBuffer>;
};

export type Brightest = { block: number; level: number };

const TILE = 1024;
const OVERLAP = 64;
const BLEND = OVERLAP / 2;
export const CORE = TILE - 2 * OVERLAP;
export const PIECE = CORE + 2 * BLEND;
const LEVEL = 0.1;
const MAX_GAIN = 64;
const EXPOSURE = 1.4;
const STEPS = 65535;
const UNHELD: Brightest = { block: TILE, level: Infinity };
const REACH = 8;
export const MARGIN = 2 * REACH;
const QUIET = 4;
const BUSY = 8;
const SHARPEST = 2;

const mirrored = (size: number, padded: number) =>
  Int32Array.from({ length: padded }, (_, p) => {
    const period = 2 * (size - 1);
    const at = (((p - OVERLAP) % period) + period) % period;

    return at < size ? at : period - at;
  });

const starts = (padded: number, tile: number) => {
  const found: number[] = [];

  for (let start = 0; start < padded - tile; start += CORE) found.push(start);

  return [...found, padded - tile];
};

const span = (size: number) => {
  const tile = Math.min(TILE, Math.ceil((size + 2 * OVERLAP) / 32) * 32);
  const padded = Math.max(size + 2 * OVERLAP, tile);

  return { tile, from: mirrored(size, padded), starts: starts(padded, tile) };
};

const filmic = (x: number) => (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);

const encoded = (x: number) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);

export function toneCurve(exposure: number) {
  const lift = 2 ** exposure;
  const white = filmic(EXPOSURE * Math.min(1, lift));

  return Uint8Array.from({ length: STEPS + 1 }, (_, i) =>
    Math.round(255 * encoded(Math.min(1, filmic((EXPOSURE * lift * i) / STEPS) / white))),
  );
}

export function photoGain(raw: Raw, offsets: number[]) {
  const sums = [0, 0, 0, 0];

  for (let y = 0; y < raw.height; y++) {
    const row = (raw.top + y) * raw.stride + raw.left;

    for (let x = 0; x < raw.width; x++) sums[((y & 1) << 1) | (x & 1)] += raw.samples[row + x];
  }

  const each = (raw.width * raw.height) / 4;

  const mean =
    sums.reduce(
      (sum, v, site) => sum + (v / each - raw.black[site]) / (raw.white - raw.black[site]) - offsets[site],
      0,
    ) / 4;

  return Math.min(MAX_GAIN, Math.max(1, LEVEL / Math.max(mean, 1e-6)));
}

const faded = (size: number, tile: number, begins: number[]) => {
  const before = new Float64Array(size);

  const tiles = begins.map((start) => {
    const own = Float64Array.from({ length: size }, (_, at) => {
      const inward = Math.min(at + OVERLAP - start, start + tile - 1 - at - OVERLAP) + 0.5;

      return Math.min(1, Math.max(0, (inward - BLEND) / OVERLAP));
    });

    const earlier = before.slice();
    own.forEach((weight, at) => (before[at] += weight));

    return { own, earlier };
  });

  return { tiles, all: before };
};

function bilinear(planes: Float32Array, across: number, y: number, x: number, into: Float32Array) {
  const plane = planes.length / 4;

  const at = (row: number, column: number) =>
    planes[(((row & 1) << 1) | (column & 1)) * plane + (row >> 1) * across + (column >> 1)];

  const here = at(y, x);
  const beside = (at(y, x - 1) + at(y, x + 1)) / 2;
  const above = (at(y - 1, x) + at(y + 1, x)) / 2;
  const corners = (at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1)) / 4;
  const green = (y ^ x) & 1;
  const onRed = (y & 1) === 0;
  into[0] = green ? (onRed ? beside : above) : onRed ? here : corners;
  into[1] = green ? here : (beside + above) / 2;
  into[2] = green ? (onRed ? above : beside) : onRed ? corners : here;
}

export async function clean(
  raw: Raw,
  offsets: number[],
  brightest: Brightest | null,
  network: Network,
  onPiece: (piece: Piece, done: number, total: number) => void,
) {
  const { block, level } = brightest ?? UNHELD;
  const gain = photoGain(raw, offsets);
  const lowest = Math.min(...raw.balance);
  const m = raw.toSrgb;
  const curve = toneCurve(raw.exposure);
  const step = (v: number) => curve[v <= 0 ? 0 : v >= 1 ? STEPS : (v * STEPS + 0.5) | 0];
  const down = span(raw.height);
  const across = span(raw.width);
  const fadeDown = faded(raw.height, down.tile, down.starts);
  const fadeAcross = faded(raw.width, across.tile, across.starts);
  const plane = (down.tile * across.tile) / 4;
  const planes = new Float32Array(4 * plane);
  const columns = Math.floor(across.tile / block) + 1;
  const sums = new Float64Array((Math.floor(down.tile / block) + 1) * columns);
  const whole = sums.map((_, i) => (i % columns < columns - 1 && i < sums.length - columns ? 1 : 0));
  const scale = raw.black.map((black) => gain / (raw.white - black));
  const shift = raw.black.map((black, site) => (black / (raw.white - black) + offsets[site]) * gain);
  const total = down.starts.length * across.starts.length;
  const { crop } = raw;
  let done = 0;

  for (let r = 0; r < down.starts.length; r++) {
    const top = down.starts[r];
    const y0 = Math.max(top - BLEND, crop.y);
    const y1 = Math.min(top + down.tile - OVERLAP - BLEND, crop.y + crop.height);
    const rows = fadeDown.tiles[r];

    for (let c = 0; c < across.starts.length; c++) {
      const left = across.starts[c];
      const x0 = Math.max(left - BLEND, crop.x);
      const x1 = Math.min(left + across.tile - OVERLAP - BLEND, crop.x + crop.width);
      const cols = fadeAcross.tiles[c];
      sums.fill(0);

      for (let y = 0; y < down.tile; y++) {
        const row = (raw.top + down.from[top + y]) * raw.stride + raw.left;
        const o = (y >> 1) * (across.tile >> 1);
        const b = Math.floor(y / block) * columns;

        for (let x = 0; x < across.tile; x++) {
          const site = ((y & 1) << 1) | (x & 1);
          const value = raw.samples[row + across.from[left + x]] * scale[site] - shift[site];
          planes[site * plane + o + (x >> 1)] = value;
          sums[b + Math.floor(x / block)] += value;
        }
      }

      const peak = sums.reduce((most, sum, i) => Math.max(most, sum * whole[i]), 0) / (block * block);
      const own = Math.min(1, Math.max(1 / gain, level / peak));

      if (own < 1) for (let i = 0; i < planes.length; i++) planes[i] *= own;

      const [red, green, blue] = raw.balance.map((b) => b / lowest / gain / own);
      const out = await network(planes, down.tile >> 1, across.tile >> 1);
      const picture = down.tile * across.tile;
      const pixels = new Uint8ClampedArray(Math.max(0, x1 - x0) * Math.max(0, y1 - y0) * 4);
      const input = new Uint8ClampedArray(pixels.length);
      const shot = new Float32Array(3);
      let p = 0;

      const paint = (onto: Uint8ClampedArray, ofRed: number, ofGreen: number, ofBlue: number) => {
        const rr = Math.min(1, ofRed * red);
        const gg = Math.min(1, ofGreen * green);
        const bb = Math.min(1, ofBlue * blue);
        onto[p] = step(m[0] * rr + m[1] * gg + m[2] * bb);
        onto[p + 1] = step(m[3] * rr + m[4] * gg + m[5] * bb);
        onto[p + 2] = step(m[6] * rr + m[7] * gg + m[8] * bb);
      };

      for (let y = y0; y < y1; y++) {
        const above = rows.earlier[y];
        const here = rows.own[y];

        for (let x = x0; x < x1; x++) {
          const i = (y - top + OVERLAP) * across.tile + x - left + OVERLAP;
          const weight = here * cols.own[x];
          paint(pixels, out[i], out[picture + i], out[2 * picture + i]);
          pixels[p + 3] = Math.round((255 * weight) / (above * fadeAcross.all[x] + here * cols.earlier[x] + weight));
          bilinear(planes, across.tile >> 1, y - top + OVERLAP, x - left + OVERLAP, shot);
          paint(input, shot[0], shot[1], shot[2]);
          input[p + 3] = 255;
          p += 4;
        }
      }

      done++;

      if (pixels.length)
        onPiece({ x: x0 - crop.x, y: y0 - crop.y, width: x1 - x0, height: y1 - y0, pixels, input }, done, total);
    }
  }
}

function brightness(rgba: Uint8ClampedArray) {
  const bright = new Float32Array(rgba.length / 4);

  for (let i = 0; i < bright.length; i++)
    bright[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];

  return bright;
}

function boxed(plane: Float32Array, width: number) {
  const height = plane.length / width;
  const across = width + 1;
  const sums = new Float64Array(across * (height + 1));

  for (let y = 0; y < height; y++) {
    let row = 0;

    for (let x = 0; x < width; x++) {
      row += plane[y * width + x];
      sums[(y + 1) * across + x + 1] = sums[y * across + x + 1] + row;
    }
  }

  for (let y = 0; y < height; y++) {
    const top = Math.max(0, y - REACH);
    const bottom = Math.min(height, y + REACH + 1);

    for (let x = 0; x < width; x++) {
      const left = Math.max(0, x - REACH);
      const right = Math.min(width, x + REACH + 1);
      const sum = sums[bottom * across + right] - sums[bottom * across + left] - sums[top * across + right];
      plane[y * width + x] = (sum + sums[top * across + left]) / ((bottom - top) * (right - left));
    }
  }

  return plane;
}

export function detailed(result: Uint8ClampedArray, width: number) {
  const bright = brightness(result);
  const busy = boxed(bright.slice(), width);

  for (let i = 0; i < busy.length; i++) busy[i] = (bright[i] - busy[i]) ** 2;

  boxed(busy, width);
  const detail = new Uint8ClampedArray(result);
  const height = bright.length / width;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const q = y * width + x;
      const share = (Math.sqrt(Math.max(0, busy[q])) - QUIET) / (BUSY - QUIET);

      if (share <= 0) continue;
      const beside = bright[q - 1] + bright[q + 1] + bright[q - width] + bright[q + width];
      const corners = bright[q - width - 1] + bright[q - width + 1] + bright[q + width - 1] + bright[q + width + 1];
      const fine = bright[q] - (4 * bright[q] + 2 * beside + corners) / 16;
      const back = SHARPEST * Math.min(1, share) * fine;

      for (let k = 0; k < 3; k++) detail[4 * q + k] = result[4 * q + k] + back;
    }
  }

  return detail;
}
