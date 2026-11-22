"""The app's Detail step (src/raw.ts) in numpy, and how far the app's own picture is from it.

    uv run detail.py RESULT.ppm INPUT.ppm DETAIL.ppm      raw_check.ts's three pictures of one photo

The step gives a photo back its camera's own finest detail where the picture is busy and nowhere else:
the finest band of the developed input's brightness (what a 3 px blur takes away) is added to the
result where the result's own brightness moves by more than QUIET levels within 17 px, in full from
BUSY levels up. It makes nothing up and touches no colour, and what it adds is the input's noise in
that band as much as its detail. The app's Detail slider fades from the result to this picture.
"""

import sys

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114])
REACH, QUIET, BUSY = 8, 4.0, 8.0
SOFT = np.array([0.25, 0.5, 0.25])


def box(plane):
    """Each pixel's mean over the 17 px around it, as far as the picture reaches."""
    blur = lambda of: cv2.blur(of, (2 * REACH + 1,) * 2, borderType=cv2.BORDER_CONSTANT)
    return blur(plane) / blur(np.ones_like(plane))


def busy(result):
    """How far a picture's brightness moves within 17 px, pooled over 17 px, in 8-bit levels (H, W)."""
    bright = result @ LUMA
    return np.sqrt(box((bright - box(bright)) ** 2).clip(0))


def detailed(result, shot):
    """A result (H, W, 3, 8 bits) with its developed input's finest band of brightness back where it is busy."""
    noisy = shot @ LUMA
    fine = np.pad((noisy - cv2.sepFilter2D(noisy, -1, SOFT, SOFT))[1:-1, 1:-1], 1)
    back = np.clip((busy(result) - QUIET) / (BUSY - QUIET), 0, 1) * fine
    return np.clip(np.rint(result + back[..., None]), 0, 255).astype(np.uint8)


def main(result, shot, detail):
    result, shot, theirs = (cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in (result, shot, detail))
    ours = detailed(result, shot)
    apart = np.abs(theirs.astype(np.int32) - ours)
    lively = busy(result) > QUIET
    print(
        f"{detail}: {ours.shape[1]}x{ours.shape[0]}, the app's Detail picture against this: "
        f"mean {apart.mean():.4f} and worst {apart.max()} of 255, {(apart == 0).mean():.3%} of values the same; "
        f"busy on {lively.mean():.1%} of the picture, which the step moves by "
        f"{np.abs(ours.astype(np.int32) - result)[lively].mean():.2f} of 255 there"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
