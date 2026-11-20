"""A good classical demosaic, for scoring a mosaic-out denoiser as a picture (bench_raw.py's colour tests).

menon() is Menon, Andriani and Calvagno's "Demosaicing with directional filtering and a posteriori
decision" (2007) with its refining step, the same sums as colour-demosaicing's
demosaicing_CFA_Bayer_Menon2007, in torch so the machine that scores needs nothing more. It
interpolates green along rows and along columns, keeps at each pixel the direction in which colour
changes less, and fills red and blue from their difference to green.
"""

import torch

convolve = torch.nn.functional.conv2d


def along(planes, taps, axis):
    """Planes (N, H, W) filtered by taps along rows (axis -1) or columns (axis -2), mirrored at the edges."""
    reach = len(taps) // 2
    pad = (reach, reach, 0, 0) if axis == -1 else (0, 0, reach, reach)
    kernel = planes.new_tensor(taps).view((1, 1, 1, -1) if axis == -1 else (1, 1, -1, 1))
    return convolve(torch.nn.functional.pad(planes[:, None], pad, mode="reflect"), kernel)[:, 0]


def menon(mosaics):
    """RGGB mosaics (N, H, W) as camera RGB (N, 3, H, W); a picture comes back as it is."""
    if mosaics.dim() == 4:
        return mosaics
    rows, columns = (lambda planes, taps: along(planes, taps, -1)), (lambda planes, taps: along(planes, taps, -2))
    site = lambda i, j: torch.zeros_like(mosaics[0], dtype=torch.bool).index_put_(
        (torch.arange(i, mosaics.shape[1], 2)[:, None], torch.arange(j, mosaics.shape[2], 2)[None]),
        torch.tensor(True),
    )
    at_red, at_blue = site(0, 0), site(1, 1)
    at_green = ~(at_red | at_blue)
    red_row, blue_row = site(0, 0) | site(0, 1), site(1, 0) | site(1, 1)
    red_column, blue_column = site(0, 0) | site(1, 0), site(0, 1) | site(1, 1)

    guess = lambda way: way(mosaics, [0, 0.5, 0, 0.5, 0]) + way(mosaics, [-0.25, 0, 0.5, 0, -0.25])
    green_h = torch.where(at_green, mosaics, guess(rows))
    green_v = torch.where(at_green, mosaics, guess(columns))
    chroma_h = torch.where(at_green, torch.zeros_like(mosaics), mosaics - green_h)
    chroma_v = torch.where(at_green, torch.zeros_like(mosaics), mosaics - green_v)
    pad = torch.nn.functional.pad
    change_h = (chroma_h - pad(chroma_h[:, None], (0, 2, 0, 0), mode="reflect")[:, 0, :, 2:]).abs()
    change_v = (chroma_v - pad(chroma_v[:, None], (0, 0, 0, 2), mode="reflect")[:, 0, 2:, :]).abs()
    around = mosaics.new_tensor([[0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 3, 0, 3], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
    summed = lambda change, kernel: convolve(change[:, None], kernel.flip(0, 1)[None, None], padding=2)[:, 0]
    level = summed(change_v, around.T) >= summed(change_h, around)
    green = torch.where(level, green_h, green_v)

    red = torch.where(at_red, mosaics, torch.zeros_like(mosaics))
    blue = torch.where(at_blue, mosaics, torch.zeros_like(mosaics))
    beside = [0.5, 0, 0.5]
    from_green = lambda colour, way: green + way(colour, beside) - way(green, beside)
    red = torch.where(at_green & red_row, from_green(red, rows), red)
    red = torch.where(at_green & blue_row, from_green(red, columns), red)
    blue = torch.where(at_green & blue_row, from_green(blue, rows), blue)
    blue = torch.where(at_green & red_row, from_green(blue, columns), blue)
    across = lambda own, other: torch.where(
        level, own + rows(other, beside) - rows(own, beside), own + columns(other, beside) - columns(own, beside)
    )
    red, blue = torch.where(at_blue, across(blue, red), red), torch.where(at_red, across(red, blue), blue)

    third = [1 / 3, 1 / 3, 1 / 3]
    smooth = lambda difference: torch.where(level, rows(difference, third), columns(difference, third))
    green = torch.where(at_red, red - smooth(red - green), green)
    green = torch.where(at_blue, blue - smooth(blue - green), green)
    red_green, blue_green = red - green, blue - green
    red = torch.where(at_green & blue_row, green + columns(red_green, beside), red)
    red = torch.where(at_green & blue_column, green + rows(red_green, beside), red)
    blue = torch.where(at_green & red_row, green + columns(blue_green, beside), blue)
    blue = torch.where(at_green & red_column, green + rows(blue_green, beside), blue)
    red_blue = red - blue
    red = torch.where(at_blue, blue + smooth(red_blue), red)
    blue = torch.where(at_red, red - smooth(red_blue), blue)
    return torch.stack([red, green, blue], 1)
