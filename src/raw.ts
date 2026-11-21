import type { Raw } from "./dng";

export type Network = (planes: Float32Array, height: number, width: number) => Promise<Float32Array>;

export type Piece = { x: number; y: number; width: number; height: number; pixels: Uint8ClampedArray<ArrayBuffer> };

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

export function levelsOf(raw: Raw, offsets: number[], block: number) {
  const columns = Math.ceil(raw.width / block);
  const sums = new Float64Array(Math.ceil(raw.height / block) * columns * 4);

  for (let y = 0; y < raw.height; y++) {
    const row = (raw.top + y) * raw.stride + raw.left;
    const at = Math.floor(y / block) * columns * 4 + ((y & 1) << 1);

    for (let x = 0; x < raw.width; x++) sums[at + Math.floor(x / block) * 4 + (x & 1)] += raw.samples[row + x];
  }

  const level = (site: number, sum: number, count: number) =>
    (sum / count - raw.black[site]) / (raw.white - raw.black[site]) - offsets[site];

  const whole = [0, 1, 2, 3].map((site) => sums.reduce((sum, v, i) => (i % 4 === site ? sum + v : sum), 0));
  const mean = whole.reduce((sum, v, site) => sum + level(site, v, (raw.width * raw.height) / 4), 0) / 4;
  const gain = Math.min(MAX_GAIN, Math.max(1, LEVEL / Math.max(mean, 1e-6)));

  const blocks = Float64Array.from({ length: sums.length / 4 }, (_, i) => {
    const wide = Math.min(block, raw.width - (i % columns) * block);
    const tall = Math.min(block, raw.height - Math.floor(i / columns) * block);
    const each = (wide * tall) / 4;

    return (
      ((level(0, sums[4 * i], each) + level(1, sums[4 * i + 1], each)) / 4) * gain +
      ((level(2, sums[4 * i + 2], each) + level(3, sums[4 * i + 3], each)) / 4) * gain
    );
  });

  return { gain, blocks, columns };
}

const ramp = (at: number, cut: number, blended: boolean) =>
  blended ? Math.min(1, Math.max(0, (at + 0.5 - cut + BLEND) / (2 * BLEND))) : at < cut ? 0 : 1;

export async function clean(
  raw: Raw,
  offsets: number[],
  brightest: Brightest | undefined,
  network: Network,
  onPiece: (piece: Piece, done: number, total: number) => void,
) {
  const block = brightest?.block ?? TILE;
  const { gain, blocks, columns } = levelsOf(raw, offsets, block);
  const lowest = Math.min(...raw.balance);
  const m = raw.toSrgb;
  const curve = toneCurve(raw.exposure);
  const step = (v: number) => curve[v <= 0 ? 0 : v >= 1 ? STEPS : (v * STEPS + 0.5) | 0];
  const down = span(raw.height);
  const across = span(raw.width);
  const plane = (down.tile * across.tile) / 4;
  const planes = new Float32Array(4 * plane);
  const total = down.starts.length * across.starts.length;
  const { crop } = raw;

  const within = (start: number, tile: number, size: number) => {
    const first = Math.floor(Math.max(0, start - OVERLAP) / block);

    return { first, last: Math.floor((Math.min(size, start - OVERLAP + tile) - 1) / block) };
  };

  const ownGain = (r: number, c: number) => {
    const ys = within(down.starts[r], down.tile, raw.height);
    const xs = within(across.starts[c], across.tile, raw.width);
    let peak = 0;

    for (let y = ys.first; y <= ys.last; y++)
      for (let x = xs.first; x <= xs.last; x++) peak = Math.max(peak, blocks[y * columns + x]);

    return brightest ? Math.min(1, brightest.level / peak) : 1;
  };

  let done = 0;

  for (let r = 0; r < down.starts.length; r++) {
    const top = down.starts[r];
    const y0 = Math.max(r ? top - BLEND : 0, crop.y);
    const y1 = Math.min(r + 1 < down.starts.length ? down.starts[r + 1] + BLEND : raw.height, crop.y + crop.height);

    for (let c = 0; c < across.starts.length; c++) {
      const left = across.starts[c];
      const x0 = Math.max(c ? left - BLEND : 0, crop.x);
      const x1 = Math.min(c + 1 < across.starts.length ? across.starts[c + 1] + BLEND : raw.width, crop.x + crop.width);
      const own = ownGain(r, c);
      const fromAbove = r > 0 && ownGain(r - 1, c) !== own;
      const fromLeft = c > 0 && ownGain(r, c - 1) !== own;
      const scale = raw.black.map((black) => (gain * own) / (raw.white - black));
      const shift = raw.black.map((black, site) => (black / (raw.white - black) + offsets[site]) * gain * own);
      const [red, green, blue] = raw.balance.map((b) => b / lowest / gain / own);

      for (let y = 0; y < down.tile; y++) {
        const row = (raw.top + down.from[top + y]) * raw.stride + raw.left;
        const o = (y >> 1) * (across.tile >> 1);

        for (let x = 0; x < across.tile; x++) {
          const site = ((y & 1) << 1) | (x & 1);
          planes[site * plane + o + (x >> 1)] = raw.samples[row + across.from[left + x]] * scale[site] - shift[site];
        }
      }

      const out = await network(planes, down.tile >> 1, across.tile >> 1);
      const picture = down.tile * across.tile;
      const pixels = new Uint8ClampedArray(Math.max(0, x1 - x0) * Math.max(0, y1 - y0) * 4);
      let p = 0;

      for (let y = y0; y < y1; y++) {
        const shown = r ? ramp(y, top, fromAbove) : 1;

        for (let x = x0; x < x1; x++) {
          const i = (y - top + OVERLAP) * across.tile + x - left + OVERLAP;
          const rr = Math.min(1, out[i] * red);
          const gg = Math.min(1, out[picture + i] * green);
          const bb = Math.min(1, out[2 * picture + i] * blue);
          pixels[p] = step(m[0] * rr + m[1] * gg + m[2] * bb);
          pixels[p + 1] = step(m[3] * rr + m[4] * gg + m[5] * bb);
          pixels[p + 2] = step(m[6] * rr + m[7] * gg + m[8] * bb);
          pixels[p + 3] = Math.round(255 * shown * (c ? ramp(x, left, fromLeft) : 1));
          p += 4;
        }
      }

      done++;

      if (pixels.length)
        onPiece({ x: x0 - crop.x, y: y0 - crop.y, width: x1 - x0, height: y1 - y0, pixels }, done, total);
    }
  }
}
