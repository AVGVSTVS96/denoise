"""Real noise fields from the Ricoh GR III, for train_raw.py to lay over clean frames: uv run noise_bank.py

The GR III and his GR IIIx share a sensor, and its raw noise is not what made-up noise looks like: the
camera's own processing ties each red and blue pixel's noise to the greens beside it (about 0.2
correlation, at every ISO), and from ISO 25,600 to its same-colour neighbours too. So the indoor
ladder's noisy frames, minus their ISO 100 twin and divided by the noise level expected at each pixel,
are kept as fields of unit strength. train_raw.py scales a window of one back up for the clean crop
under it. The still-life ladders stay out: they are bench_raw.py's test.

Writes data/noise/gr3.npy, float16 (ISOs, H, W), and gr3.json with each ISO's shot and read noise per
colour plane, in the frame's own normalised raw units.
"""

import json

import numpy as np

import raw_pairs

OUT = raw_pairs.DATA / "noise"
PATTERN = "ricoh-gr-iii--GR3INBI{:06d}.DNG"
ISOS = (3200, 6400, 12800, 25600, 51200, 102400)
BLOCK = 16
SITES = ((0, 0), (0, 1), (1, 0), (1, 1))


def blocks(plane):
    h, w = plane.shape[0] // BLOCK * BLOCK, plane.shape[1] // BLOCK * BLOCK
    return plane[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2).reshape(-1, BLOCK * BLOCK)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = lambda iso: raw_pairs.NIGHT / "imaging-resource" / PATTERN.format(iso)
    clean = raw_pairs.mosaic(path(100))
    rng = np.random.default_rng(0)
    fields, levels = [], {}
    for iso in ISOS:
        noisy = raw_pairs.mosaic(path(iso))
        assert raw_pairs.shift_between(clean, noisy) == (0, 0), iso
        matched, gains, _ = raw_pairs.match(clean, noisy)
        field = np.empty_like(clean)
        shot, read = [], []
        for (i, j), gain in zip(SITES, gains):
            truth, residual = clean[i::2, j::2], matched[i::2, j::2] - clean[i::2, j::2]
            mean, spread = blocks(truth).mean(axis=1), blocks(truth).std(axis=1)
            # Flat blocks only, where nothing of the scene can pass for noise.
            flat = (spread < 0.02 * mean + 0.0005) & (mean > 0.002) & (mean < 0.6)
            slope, floor = np.polyfit(mean[flat], blocks(residual).var(axis=1)[flat], 1)
            floor = max(floor, 1e-8)
            unit = residual / np.sqrt(slope * truth.clip(min=0) + floor)
            # A clipped highlight has no noise to keep.
            clipped = (truth > 0.95) | (noisy[i::2, j::2] > 0.95 * noisy.max())
            unit[clipped] = rng.standard_normal(int(clipped.sum()))
            field[i::2, j::2] = unit
            shot.append(slope / gain)
            read.append(float(np.sqrt(floor)) / gain)
            textured = ~flat & (mean > 0.002) & (mean < 0.6)
            print(
                f"ISO {iso} plane {i}{j}: shot {shot[-1]:.3g} read {read[-1]:.3g}, unit mean {unit.mean():+.3f},"
                f" variance {blocks(unit)[flat].var():.3f} flat, {blocks(unit)[textured].var():.3f} textured,"
                f" clipped {clipped.mean():.4f}",
                flush=True,
            )
        fields.append(field)
        levels[iso] = {"shot": shot, "read": read}
    np.save(OUT / "gr3.npy", np.stack(fields).astype(np.float16))
    (OUT / "gr3.json").write_text(json.dumps({"camera": "Ricoh GR III", "iso": levels}, indent=1))


if __name__ == "__main__":
    main()
