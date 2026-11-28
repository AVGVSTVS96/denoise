import { laid, OVERLAP, type Frame, type Merge } from "./raw";

type Plane = { values: Float32Array; width: number; height: number };

type Sat = { dx: number; dy: number; turn: number };

type Noise = { shot: number; read: number };

type Found = { sat: Sat; gain: number; noise: Noise[] };

const NEAR = 5;
const ALLOWED = 6;
const FADE = 4;
const FINE = 1;
const COARSE = 4;
const ROUNDS = 6;
const STILL = 0.001;
const MARGIN = 3;
const TRIM = 4;
const LOST = 3;
const WHOLE = [16, 4];
const SMALLEST = 64;
const SPARSE = 8;
const SKIPPED = 7;
const GROUPS = 24;
const BLOWN = 0.7;
const DARK = 0.01;
const MEDIAN_OF_SQUARE = 0.4549;
const POLE = Math.sqrt(3) - 2;
const SETTLED = 24;

const folded = (at: number, size: number) => (at < 0 ? -at : at >= size ? 2 * size - 2 - at : at);

function smoothed({ values, width, height }: Plane, sigma: number) {
  const reach = Math.round(4 * sigma);
  const bell = Float32Array.from({ length: 2 * reach + 1 }, (_, i) => Math.exp(-((i - reach) ** 2) / (2 * sigma ** 2)));
  const all = bell.reduce((sum, tap) => sum + tap, 0);
  const taps = bell.map((tap) => tap / all);
  const along = new Float32Array(values.length);
  const out = new Float32Array(values.length);

  for (let y = 0; y < height; y++) {
    const row = y * width;

    for (let x = 0; x < width; x++) {
      const inside = x >= reach && x < width - reach;
      let sum = 0;

      if (inside) for (let k = 0; k < taps.length; k++) sum += taps[k] * values[row + x + k - reach];
      else for (let k = 0; k < taps.length; k++) sum += taps[k] * values[row + folded(x + k - reach, width)];

      along[row + x] = sum;
    }
  }

  for (let y = 0; y < height; y++) {
    for (let k = 0; k < taps.length; k++) {
      const from = folded(y + k - reach, height) * width;
      const tap = taps[k];

      for (let x = 0; x < width; x++) out[y * width + x] += tap * along[from + x];
    }
  }

  return out;
}

function band(plane: Plane): Plane {
  const fine = smoothed(plane, FINE);
  const coarse = smoothed(plane, COARSE);

  return { ...plane, values: fine.map((value, i) => value - coarse[i]) };
}

function fitted(a: Plane, b: Plane, start: Sat, edge: number): Sat {
  const { width, height } = a;
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  const from = splined(a);
  const moved = new Float32Array(width * height);
  const rests = new Float32Array(Math.ceil((width - 2 * edge) / SKIPPED) * (height - 2 * edge));
  let { dx, dy, turn } = start;

  for (let round = 0; round < ROUNDS; round++) {
    const cos = Math.cos(turn);
    const sin = Math.sin(turn);

    for (let y = edge - 1; y <= height - edge; y++) {
      for (let x = edge - 1; x <= width - edge; x++) {
        const down = y - cy - dy;
        const along = x - cx - dx;
        const fromX = cx + sin * down + cos * along;
        const fromY = cy + cos * down - sin * along;
        const ix = Math.min(width - 3, Math.max(1, Math.floor(fromX)));
        const iy = Math.min(height - 3, Math.max(1, Math.floor(fromY)));
        const [x0, x1, x2, x3] = cubic(Math.min(1, Math.max(0, fromX - ix)));
        const [y0, y1, y2, y3] = cubic(Math.min(1, Math.max(0, fromY - iy)));
        const q = (iy - 1) * width + ix - 1;
        const line = (at: number) => x0 * from[at] + x1 * from[at + 1] + x2 * from[at + 2] + x3 * from[at + 3];
        moved[y * width + x] =
          y0 * line(q) + y1 * line(q + width) + y2 * line(q + 2 * width) + y3 * line(q + 3 * width);
      }
    }

    let kept = 0;

    for (let y = edge; y < height - edge; y++)
      for (let x = edge + ((y * 3) % SKIPPED); x < width - edge; x += SKIPPED)
        rests[kept++] = Math.abs(b.values[y * width + x] - moved[y * width + x]);

    const limit = TRIM * rests.subarray(0, kept).toSorted()[kept >> 1];
    const m = [0, 0, 0, 0, 0, 0];
    const v = [0, 0, 0];

    for (let y = edge; y < height - edge; y++) {
      for (let x = edge; x < width - edge; x++) {
        const q = y * width + x;
        const rest = b.values[q] - moved[q];

        if (Math.abs(rest) >= limit) continue;
        const gx = (moved[q - 1] - moved[q + 1]) / 2;
        const gy = (moved[q - width] - moved[q + width]) / 2;
        const gt = gy * (x - cx) - gx * (y - cy);
        m[0] += gx * gx;
        m[1] += gx * gy;
        m[2] += gx * gt;
        m[3] += gy * gy;
        m[4] += gy * gt;
        m[5] += gt * gt;
        v[0] += gx * rest;
        v[1] += gy * rest;
        v[2] += gt * rest;
      }
    }

    const step = solved(m, v);

    if (!step) break;
    dx += step[0];
    dy += step[1];
    turn += step[2];

    if (Math.hypot(step[0], step[1]) < STILL && Math.abs(step[2]) * Math.max(cx, cy) < STILL) break;
  }

  return { dx, dy, turn };
}

const cubic = (t: number) => [
  (1 - t) ** 3 / 6,
  ((3 * t - 6) * t * t + 4) / 6,
  (((3 - 3 * t) * t + 3) * t + 1) / 6,
  t ** 3 / 6,
];

function splined({ values, width, height }: Plane) {
  const found = Float32Array.from(values);

  const along = (start: number, step: number, count: number) => {
    const last = start + (count - 1) * step;
    let first = 0;

    for (let k = Math.min(count, SETTLED) - 1; k >= 0; k--) first = POLE * first + found[start + k * step];

    found[start] = 6 * first;

    for (let at = start + step; at <= last; at += step) found[at] = 6 * found[at] + POLE * found[at - step];

    found[last] = (POLE / (POLE * POLE - 1)) * (found[last] + POLE * found[last - step]);

    for (let at = last - step; at >= start; at -= step) found[at] = POLE * (found[at + step] - found[at]);
  };

  for (let y = 0; y < height; y++) along(y * width, 1, width);

  for (let x = 0; x < width; x++) along(x, width, height);

  return found;
}

function solved([a, b, c, d, e, f]: number[], [x, y, z]: number[]) {
  const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);

  const found = [
    (x * (d * f - e * e) - b * (y * f - e * z) + c * (y * e - d * z)) / det,
    (a * (y * f - e * z) - x * (b * f - e * c) + c * (b * z - y * c)) / det,
    (a * (d * z - y * e) - b * (b * z - y * c) + x * (b * e - d * c)) / det,
  ];

  return found.every(Number.isFinite) ? found : undefined;
}

function binned({ raw, offsets }: Frame, by: number): Plane {
  const width = Math.floor(raw.width / (2 * by));
  const height = Math.floor(raw.height / (2 * by));
  const values = new Float32Array(width * height);
  const level = (site: number) => raw.black[site] / (raw.white - raw.black[site]) + offsets[site];

  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      let first = 0;
      let second = 0;

      for (let y = 2 * by * v; y < 2 * by * (v + 1); y += 2) {
        const row = (raw.top + y) * raw.stride + raw.left + 2 * by * u;

        for (let x = 0; x < 2 * by; x += 2) {
          first += raw.samples[row + x + 1];
          second += raw.samples[row + raw.stride + x];
        }
      }

      values[v * width + u] =
        (first / (raw.white - raw.black[1]) + second / (raw.white - raw.black[2])) / (2 * by * by) -
        (level(1) + level(2)) / 2;
    }
  }

  return { values, width, height };
}

function shrunk({ values, width: wide, height: tall }: Plane, by: number): Plane {
  const width = Math.floor(wide / by);
  const height = Math.floor(tall / by);
  const small = new Float32Array(width * height);

  for (let v = 0; v < height; v++)
    for (let u = 0; u < width; u++) {
      let sum = 0;

      for (let y = by * v; y < by * (v + 1); y++)
        for (let x = by * u; x < by * (u + 1); x++) sum += values[y * wide + x];

      small[v * width + u] = sum / (by * by);
    }

  return { values: small, width, height };
}

const middle = ({ raw }: Frame) => {
  const finest = 2 * WHOLE[WHOLE.length - 1];

  return [(Math.floor(raw.width / finest) * finest - 1) / 2, (Math.floor(raw.height / finest) * finest - 1) / 2];
};

function linedUp(fine: Plane[]): Sat {
  const finest = WHOLE[WHOLE.length - 1];
  let sat = { dx: 0, dy: 0, turn: 0 };

  for (const cells of WHOLE) {
    const [a, b] = fine.map((plane) => (cells === finest ? plane : shrunk(plane, cells / finest)));

    if (Math.min(a.width, a.height) < SMALLEST) continue;
    const start = { dx: sat.dx / (2 * cells), dy: sat.dy / (2 * cells), turn: sat.turn };
    const found = fitted(band(a), band(b), start, MARGIN * COARSE);
    sat = { dx: found.dx * 2 * cells, dy: found.dy * 2 * cells, turn: found.turn };
  }

  return sat;
}

function brighter([a, b]: Plane[]) {
  const ratios = Float32Array.from(a.values, (value, i) => (value > DARK && value < BLOWN ? value / b.values[i] : NaN));
  const lit = ratios.filter((ratio) => ratio > 0 && Number.isFinite(ratio)).toSorted();

  return lit.length ? lit[lit.length >> 1] : 1;
}

function offAt(sat: Sat, [cx, cy]: number[], x: number, y: number) {
  const cos = Math.cos(sat.turn);
  const sin = Math.sin(sat.turn);

  return [sat.dx + (cos - 1) * (x - cx) - sin * (y - cy), sat.dy + sin * (x - cx) + (cos - 1) * (y - cy)];
}

const sampled = (frame: Frame, site: number) => {
  const { samples, stride, top, left, black, white } = frame.raw;
  const scale = 1 / (white - black[site]);
  const level = black[site] * scale + frame.offsets[site];
  const start = (top + (site >> 1)) * stride + left + (site & 1);

  return (y: number, x: number) => samples[start + y * stride + x] * scale - level;
};

function differing(picked: Frame, other: Frame, sat: Sat, gain: number): Noise[] {
  const { raw } = picked;
  const centre = middle(picked);
  const reach = NEAR >> 1;
  const across = Math.floor((raw.width / 2 - 2 * reach - 1) / SPARSE) + 1;
  const down = Math.floor((raw.height / 2 - 2 * reach - 1) / SPARSE) + 1;
  const bright = new Float32Array(across * down);
  const squares = new Float32Array(bright.length);

  return [0, 1, 2, 3].map((site) => {
    const first = sampled(picked, site);
    const second = sampled(other, site);
    let found = 0;

    for (let v = 0; v < down; v++) {
      for (let u = 0; u < across; u++) {
        const y = 2 * (reach + v * SPARSE);
        const x = 2 * (reach + u * SPARSE);
        const [dx, dy] = offAt(sat, centre, x + 0.5, y + 0.5);
        const fromY = y + 2 * Math.round(dy / 2);
        const fromX = x + 2 * Math.round(dx / 2);

        if (fromY < 0 || fromX < 0 || fromY >= raw.height - 1 || fromX >= raw.width - 1) continue;
        let sum = 0;

        for (let j = -reach; j <= reach; j++) for (let i = -reach; i <= reach; i++) sum += first(y + 2 * j, x + 2 * i);

        if (sum / NEAR ** 2 >= BLOWN) continue;
        bright[found] = sum / NEAR ** 2;
        squares[found++] = (second(fromY, fromX) * gain - first(y, x)) ** 2;
      }
    }

    const order = Uint32Array.from({ length: found }, (_, i) => i).toSorted((i, j) => bright[i] - bright[j]);

    const points = Array.from({ length: Math.min(GROUPS, found) }, (_, group) => {
      const from = Math.floor((group * found) / GROUPS);
      const to = Math.max(from + 1, Math.floor(((group + 1) * found) / GROUPS));
      const part = order.subarray(from, to);
      const square = Float32Array.from(part, (i) => squares[i]).toSorted();

      return [bright[part[part.length >> 1]], square[square.length >> 1] / MEDIAN_OF_SQUARE];
    });

    const mean = (k: number) => points.reduce((sum, point) => sum + point[k], 0) / Math.max(points.length, 1);
    const spread = points.reduce((sum, [x]) => sum + (x - mean(0)) ** 2, 0);
    const shot = points.reduce((sum, [x, y]) => sum + (x - mean(0)) * (y - mean(1)), 0) / Math.max(spread, 1e-30);

    return { shot: Math.max(shot, 1e-9), read: Math.max(mean(1) - shot * mean(0), 0) };
  });
}

export function measured(picked: Frame, other: Frame): Found {
  const finest = WHOLE[WHOLE.length - 1];
  const fine = [binned(picked, finest), binned(other, finest)];
  const sat = linedUp(fine);
  const gain = brighter(fine.map((plane) => shrunk(plane, WHOLE[0] / finest)));

  return { sat, gain, noise: differing(picked, other, sat, gain) };
}

function boxed(values: Float32Array, width: number, reach: number) {
  const height = values.length / width;
  const along = new Float32Array(values.length);
  const out = new Float32Array(values.length);

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;

    for (let x = 0; x < Math.min(reach, width); x++) sum += values[row + x];

    for (let x = 0; x < width; x++) {
      if (x + reach < width) sum += values[row + x + reach];

      if (x - reach - 1 >= 0) sum -= values[row + x - reach - 1];
      along[row + x] = sum;
    }
  }

  for (let x = 0; x < width; x++) {
    let sum = 0;

    for (let y = 0; y < Math.min(reach, height); y++) sum += along[y * width + x];

    for (let y = 0; y < height; y++) {
      if (y + reach < height) sum += along[(y + reach) * width + x];

      if (y - reach - 1 >= 0) sum -= along[(y - reach - 1) * width + x];
      out[y * width + x] = sum;
    }
  }

  return out;
}

function fired(picked: Float32Array, other: Float32Array, across: number, noise: Noise[], gain: number) {
  const cells = picked.length / 4;
  const reach = NEAR >> 1;
  const off = new Float32Array(cells);
  const apart = new Float32Array(cells);

  for (let site = 0; site < 4; site++) {
    const { shot, read } = noise[site];
    const level = boxed(picked.subarray(site * cells, (site + 1) * cells), across, reach);

    for (let i = 0; i < cells; i++) apart[i] = other[site * cells + i] - picked[site * cells + i];

    const sums = boxed(apart, across, reach);

    for (let i = 0; i < cells; i++) {
      const allowed = ALLOWED ** 2 * (shot * gain * Math.max(level[i], 0) + read * gain ** 2 * NEAR ** 2);

      if (sums[i] ** 2 > allowed) off[i] = 1;
    }
  }

  return boxed(off, across, reach).map((near) => (near > 0 ? 1 : 0));
}

function shares(where: Float32Array, across: number) {
  const width = 2 * across;
  const height = (2 * where.length) / across;
  const reach = FADE >> 1;
  const full = new Float32Array(width * height);

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) full[y * width + x] = where[(y >> 1) * across + (x >> 1)];

  const wide = boxed(full, width, reach).map((near) => (near > 0 ? 1 : 0));
  const sums = boxed(wide, width, reach);
  const inside = (at: number, size: number) => Math.min(at + reach, size - 1) - Math.max(at - reach, 0) + 1;

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) sums[y * width + x] /= inside(x, width) * inside(y, height);

  return sums;
}

export function burst(picked: Frame, later: Frame[]): Merge {
  const centre = middle(picked);
  const found = later.map((frame) => measured(picked, frame));

  return async (planes, tile, network) => {
    const across = tile.width >> 1;
    const cells = planes.length / 4;
    const other = new Float32Array(planes.length);
    const sum = Float32Array.from(planes);
    const trusted = new Float32Array(planes.length).fill(1);

    const green = (frame: Float32Array): Plane => {
      const first = frame.subarray(cells, 2 * cells);
      const second = frame.subarray(2 * cells, 3 * cells);

      return { values: first.map((value, i) => (value + second[i]) / 2), width: across, height: cells / across };
    };

    const detail = band(green(planes));

    later.forEach((frame, k) => {
      const { sat, gain, noise } = found[k];
      const matched = { ...tile, gain: tile.gain * gain };
      const [px, py] = offAt(sat, centre, tile.left + (tile.width - 1) / 2, tile.top + (tile.height - 1) / 2);
      const whole = [2 * Math.round(px / 2), 2 * Math.round(py / 2)];
      const start = { dx: (px - whole[0]) / 2, dy: (py - whole[1]) / 2, turn: sat.turn };
      laid(frame, matched, whole[0], whole[1], other);
      const left = fitted(detail, band(green(other)), start, OVERLAP / 2);
      const lost = Math.hypot(left.dx - start.dx, left.dy - start.dy) > LOST;
      const more = lost ? [0, 0] : [2 * Math.round(left.dx), 2 * Math.round(left.dy)];

      if (more[0] || more[1]) laid(frame, matched, whole[0] + more[0], whole[1] + more[1], other);

      const share = shares(fired(planes, other, across, noise, tile.gain), across);

      for (let y = 0; y < tile.height; y++) {
        for (let x = 0; x < tile.width; x++) {
          const at = (((y & 1) << 1) | (x & 1)) * cells + (y >> 1) * across + (x >> 1);
          const trust = 1 - share[y * tile.width + x];
          sum[at] += trust * other[at];
          trusted[at] += trust;
        }
      }
    });

    return network(
      sum.map((value, i) => value / trusted[i]),
      cells / across,
      across,
    );
  };
}
