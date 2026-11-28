"""Two dials for a denoised picture against its truth, by the size of what is looked at.

A picture (N, 3, H, W) from 0 to 1, developed as bench_raw.score_one develops it, is split into
brightness and two colour differences, and each into five bands of detail, fine to coarse: what each
blur of SIGMAS takes away that the last did not. MIDDLES is the middle of each band in cycles per
pixel, the finest being everything above the first blur: detail 2 to 3 px across.

Where the truth is flat a band holds only what should not be there: noise left. Where the truth is
busy the share of the truth's band a result holds is detail kept, and what else the result's band
holds there is invented. Each is then weighed by how well the eye sees that size at a viewing
distance, which gives one number for each. VIEWS is that distance in pixels per degree: a photo at
100% on a laptop at arm's length, and zoomed in to 2x, the blind sheet's view and the one read.

Noise left is in 8-bit levels, and reads best as stops cleaner than the noisy frame: twice the log2
of the noisy frame's noise over the result's, since a stop of ISO is 1.4 times the noise. Detail kept
is 1 when all the truth's detail is there and over 1 in a band that was sharpened; a bilinear demosaic
of a frame nobody denoised keeps 0.8, so read it with invented, which sharpening and left-over noise
both raise. Detail is only worth reading against a truth sharper than the result, the full-colour
tests'; against a real noisy frame's clean twin it says little.

Detail kept is carried by edges and print, and a model can smooth cloth, grain or foliage flat while
it stays high. Texture kept and texture wrong are the finest band's kept and invented where the truth
is textured instead: where that band is FINE times its strength in the flat region or more, which is
the truth's own noise, and nothing near steps by more than STEP 8-bit levels. They are read against a
real noisy frame's clean twin too, between models on one test: a truth's noise and how well it sits on
the noisy frame lower them for every model alike.

A clean twin is a mosaic, so as a picture it has been through a demosaic, and a cleaned mosaic put
through the same one shares its errors where a model's own picture does not. on_mosaic() scores with
none on either side: a picture's own colour at each pixel against the twin's mosaic, as PSNR and as
kept and wrong in the finest band one colour's own samples can hold. Green's lie 1.4 px apart along
the diagonals, so its band is detail 3 to 4 px across; red's and blue's lie 2 px apart, detail 4 to
6 px. It cannot see the 2 to 3 px band a picture has, nor how well a model fills in the two colours a
pixel did not measure.
"""

import itertools
import math

import torch

SIGMAS = (0.7, 1.4, 2.8, 5.6, 11.2)
MIDDLES = (0.4, 0.219, 0.109, 0.055, 0.027)
VIEWS = {"100%": 45.0, "2x": 22.0}
FLATTEST, BUSIEST = 0.3, 0.2
FINE, STEP = 2.0, 20.0


def planes(picture):
    """sRGB (N, 3, H, W) as brightness and two colour differences (N, 3, H, W), in 8-bit levels."""
    red, green, blue = picture.unbind(1)
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    return torch.stack([luma, (blue - luma) * 0.564, (red - luma) * 0.713], 1) * 255


def blurred(picture, sigma):
    reach = max(1, int(3 * sigma + 0.5))
    taps = torch.arange(-reach, reach + 1, dtype=picture.dtype, device=picture.device)
    bell = torch.exp(-(taps**2) / (2 * sigma**2))
    bell = (bell / bell.sum()).expand(picture.shape[1], 1, -1)
    padded = torch.nn.functional.pad(picture, (reach,) * 4, mode="reflect")
    rows = torch.nn.functional.conv2d(padded, bell[:, :, None], groups=picture.shape[1])
    return torch.nn.functional.conv2d(rows, bell[:, :, :, None], groups=picture.shape[1])


def bands(picture):
    """A picture's five bands (5, N, C, H, W), finest first."""
    steps = [picture, *(blurred(picture, sigma) for sigma in SIGMAS)]
    return torch.stack([sharper - softer for sharper, softer in itertools.pairwise(steps)])


def regions(truth, flattest=FLATTEST):
    """Where the truth (N, 3, H, W) is flattest and where it is busiest, as masks (N, 1, H, W).

    By how much its brightness moves within 12 px or so, pooled over 17 px: the flattest and the
    busiest shares of all the windows' pixels together.
    """
    luma = planes(truth)[:, :1]
    moves = luma - blurred(luma, SIGMAS[2])
    padded = torch.nn.functional.pad(moves * moves, (8,) * 4, mode="reflect")
    busy = torch.nn.functional.avg_pool2d(padded, 17, 1).sqrt()
    ranked = busy.flatten().sort().values
    return busy <= ranked[int(flattest * (len(ranked) - 1))], busy >= ranked[int((1 - BUSIEST) * (len(ranked) - 1))]


def texture(truth, flat):
    """Where the truth (N, 3, H, W) has fine detail and no hard edge, as a mask (N, 1, H, W).

    Fine detail: over the 17 px round a pixel its finest band is FINE times as strong as in the flat
    region, or stronger. No hard edge: with detail under 6 px or so blurred away, its brightness moves
    STEP 8-bit levels or less across any 5 px of those 17.
    """
    luma = planes(truth)[:, :1]
    fine = luma - blurred(luma, SIGMAS[0])
    around = lambda plane, side, pool: pool(torch.nn.functional.pad(plane, (side // 2,) * 4, mode="reflect"), side, 1)
    strength = around(fine * fine, 17, torch.nn.functional.avg_pool2d).sqrt()
    soft, highest = blurred(luma, SIGMAS[1]), torch.nn.functional.max_pool2d
    step = around(around(soft, 5, highest) + around(-soft, 5, highest), 17, highest)
    return (strength >= FINE * fine[flat].pow(2).mean().sqrt()) & (step <= STEP)


def eye(cycles_per_degree, colour=False):
    """How well the eye sees a stripe pattern, 1 at its best: Mannos and Sakrison's curve for brightness,
    and for colour a curve that only falls, to half by 3 cycles a degree, as colour vision does."""
    f = cycles_per_degree
    if colour:
        return math.exp(-math.log(2) * (f / 3) ** 2)
    return min(1.0, 2.6 * (0.0192 + 0.114 * f) * math.exp(-((0.114 * f) ** 1.1)) / 0.98)


def weights(view, colour=False):
    return torch.tensor([eye(middle * VIEWS[view], colour) for middle in MIDDLES])


def sums(made, real, where):
    """What fitting one band to another over a region takes: the region's size, and the sums of each
    band there, of their squares and of their product, in that order."""
    x, y = made[where].double(), real[where].double()
    return torch.stack([x.new_tensor(x.numel()), x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), (y * y).sum()])


def held(fitted):
    """From sums() of a region: the share of the truth's band the result holds there, and the spread of
    what else the result's band holds, over the truth's."""
    _, _, _, xx, xy, yy = fitted.unbind(-1)
    return xy / yy, ((xx - xy**2 / yy) / yy).clamp(min=0).sqrt()


def finest(mosaic, where, apart):
    """A developed RGGB mosaic's (N, H, W) finest band at one colour's sites, in 8-bit levels, as (N, 1, H, W).

    where (H, W) is that colour's sites and apart how far its samples lie from each other in pixels.
    The band is what a blur of SIGMAS[0] of the colour's own samples takes away from each, as bands()
    takes a picture's finest: no other colour's sample is read, so no demosaic is. It is nothing at
    the other colours' sites.
    """
    own = where.to(mosaic.dtype).expand(1, 1, -1, -1)
    planes = mosaic[:, None]
    soft = blurred(planes * own, SIGMAS[0] * apart) / blurred(own, SIGMAS[0] * apart)
    return (planes - soft) * own * 255


def on_mosaic(result, truth, busy, textured):
    """What a result is against its truth with no demosaic on either side: developed RGGB mosaics (N, H, W), 0 to 1.

    psnr_mosaic: the windows' mean PSNR over every site;
    mosaic_kept, mosaic_wrong (3,): held() of the finest band a colour's own samples can hold, in the
    busy region, at the green sites, the red and the blue;
    mosaic_texture_kept, mosaic_texture_wrong (3,): the same in the textured region.
    """
    red = torch.zeros(result.shape[1:], dtype=torch.bool, device=result.device)
    blue = red.clone()
    red[0::2, 0::2] = blue[1::2, 1::2] = True
    found = {name: [] for name in ("mosaic_kept", "mosaic_wrong", "mosaic_texture_kept", "mosaic_texture_wrong")}
    for where, apart in ((~(red | blue), 2**0.5), (red, 2.0), (blue, 2.0)):
        made, real = finest(result, where, apart), finest(truth, where, apart)
        for region, name in ((busy, "mosaic"), (textured, "mosaic_texture")):
            share, wrong = held(sums(made, real, region & where))
            found[f"{name}_kept"].append(share.item())
            found[f"{name}_wrong"].append(wrong.item())
    psnr = -10 * torch.log10(((result - truth) ** 2).mean((1, 2)))
    return {"psnr_mosaic": psnr.mean().item()} | found


def measure(result, truth, flat, busy, textured=None):
    """What a result (N, 3, H, W) is against its truth, band by band:

    left (5, 3): in the flat region, the spread of what a band holds that does not follow the truth's
    band, in 8-bit levels, for brightness and the two colours;
    kept (5,): in the busy region, the share of the truth's brightness band the result holds;
    invented (5,): the spread of what else the result's band holds there, over the truth's;
    texture_kept, texture_wrong: the finest band's kept and invented in the textured region, texture()'s,
    when one is given.

    The windows go through eight at a time and only sums() of them are kept, so a test of any size
    costs a training run's card no more than eight windows do.
    """
    smooth, rough, woven = 0, 0, 0
    for at in range(0, len(result), 8):
        part = slice(at, at + 8)
        pairs = list(zip(bands(planes(result[part])), bands(planes(truth[part]))))
        smooth = smooth + torch.stack(
            [torch.stack([sums(a[:, k : k + 1], b[:, k : k + 1], flat[part]) for k in range(3)]) for a, b in pairs]
        )
        rough = rough + torch.stack([sums(a[:, :1], b[:, :1], busy[part]) for a, b in pairs])
        if textured is not None:
            woven = woven + sums(pairs[0][0][:, :1], pairs[0][1][:, :1], textured[part])
    size, x, y, xx, xy, yy = smooth.unbind(-1)
    together = xy / size - x * y / size**2
    left = (xx / size - (x / size) ** 2 - together**2 / (yy / size - (y / size) ** 2)).clamp(min=0).sqrt()
    kept, invented = held(rough)
    found = {"left": left.tolist(), "kept": kept.tolist(), "invented": invented.tolist()}
    if textured is None:
        return found
    texture_kept, texture_wrong = held(woven)
    return found | {"texture_kept": texture_kept.item(), "texture_wrong": texture_wrong.item()}


def dials(found, view="2x"):
    """Noise left, detail kept and invented as one number each from measure()'s bands, as the eye weighs them at a view.

    Noise left: the bands' left-over spreads added as energies, brightness by the eye's curve for
    brightness and the two colours by its curve for colour, in 8-bit levels. Detail kept and
    invented: the bands' means under the brightness curve.
    """
    left = torch.tensor(found["left"])
    bright, tint = weights(view), weights(view, colour=True)
    noise = ((left[:, 0] * bright) ** 2).sum() + ((left[:, 1:] * tint[:, None]) ** 2).sum()
    weighed = lambda band: ((torch.tensor(found[band]) * bright).sum() / bright.sum()).item()
    return noise.sqrt().item(), weighed("kept"), weighed("invented")
