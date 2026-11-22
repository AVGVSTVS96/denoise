"""The app's Detail step (src/raw.ts) in numpy, and how far the app's own picture is from it.

    uv run detail.py RESULT.ppm INPUT.ppm DETAIL.ppm          raw_check.ts's three pictures of one photo
    uv run detail.py RESULT.ppm INPUT.ppm DETAIL.ppm NOISE    the same, under the noise raw_check.ts measured

The step gives a photo back its camera's own finest detail where the picture is busy and nowhere else:
the finest band of the developed input's brightness (what a 3 px blur takes away) is added to the
result where the result's own brightness moves by more than QUIET levels within 9 px, in full from
BUSY levels up. It makes nothing up and touches no colour, and what it adds is the input's noise in
that band as much as its detail. So the picture the app's Detail slider fades to holds MOST of that
band and no more, and less of it the noisier the photo is: the photo's own noise is the strength of
that band where the result is flat, in 8-bit levels, and the picture holds all of MOST up to CALM
levels of it and none from GRAINY levels, where the band is all grain. No ISO and no camera is looked up.

The app measures that noise tile by tile as it cleans, this over the whole picture, so the two differ by
about 1 part in 100. Between CALM and GRAINY that is enough to move the picture, so give the app's NOISE to
compare the step alone.
"""

import sys

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114])
REACH, QUIET, BUSY, MOST, CALM, GRAINY = 4, 4.0, 8.0, 0.5, 8.2, 9.0
SOFT = np.array([0.25, 0.5, 0.25])


def box(plane):
    """Each pixel's mean over the 9 px around it, as far as the picture reaches."""
    blur = lambda of: cv2.blur(of, (2 * REACH + 1,) * 2, borderType=cv2.BORDER_CONSTANT)
    return blur(plane) / blur(np.ones_like(plane))


def busy(result):
    """How far a picture's brightness moves within 9 px, pooled over 9 px, in 8-bit levels (H, W)."""
    bright = result @ LUMA
    return np.sqrt(box((bright - box(bright)) ** 2).clip(0))


def noise(result, shot):
    """A photo's own noise: the strength of its developed input's finest band of brightness where its result is flat, in 8-bit levels."""
    inner = (slice(1, -1),) * 2
    flat = (busy(result) <= QUIET)[inner]
    return float(np.sqrt((fine(shot)[inner][flat] ** 2).sum() / max(flat.sum(), 1)))


def band(grainy):
    """How much of the noisy frame's finest band the Detail picture holds: MOST up to CALM levels of noise, none from GRAINY."""
    return MOST * float(np.clip((GRAINY - grainy) / (GRAINY - CALM), 0, 1))


def fine(picture):
    """A picture's finest band of brightness, what a 3 px blur takes away, in 8-bit levels (H, W), none on its rim."""
    bright = picture @ LUMA
    return np.pad((bright - cv2.sepFilter2D(bright, -1, SOFT, SOFT))[1:-1, 1:-1], 1)


def detailed(result, shot, share):
    """A result (H, W, 3, 8 bits) with that share of its developed input's finest band of brightness back where it is busy."""
    back = share * np.clip((busy(result) - QUIET) / (BUSY - QUIET), 0, 1) * fine(shot)
    return np.clip(np.rint(result + back[..., None]), 0, 255).astype(np.uint8)


def main(result, shot, detail, measured=None):
    result, shot, theirs = (cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in (result, shot, detail))
    grainy = noise(result, shot)
    share = band(float(measured or grainy))
    ours = detailed(result, shot, share)
    apart = np.abs(theirs.astype(np.int32) - ours)
    lively = busy(result) > QUIET
    print(
        f"{detail}: {ours.shape[1]}x{ours.shape[0]}, noise {grainy:.3f} of 255"
        f"{f' here and {float(measured):.3f} in the app' if measured else ''}, so the picture holds {share:.4f} of the band; "
        f"the app's Detail picture against this: "
        f"mean {apart.mean():.4f} and worst {apart.max()} of 255, {(apart == 0).mean():.3%} of values the same; "
        f"busy on {lively.mean():.1%} of the picture, which the step moves by "
        f"{np.abs(ours.astype(np.int32) - result)[lively].mean():.2f} of 255 there"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
