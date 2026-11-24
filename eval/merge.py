"""Burst v2's align and merge: several raw frames of one scene lined up to a fraction of a pixel and
merged, outside any network, for a one-frame model to clean.

    total, count = merge.merged(frames, levels)       frames (N, frames, H, W), the reference first
    mosaic = merge.mosaic(total, count)                the merge as a mosaic (N, H, W)
    first = merge.sharpest(frames, levels)             which frame of each burst to put first

It is three steps, each a few passes of sums over small neighbourhoods, so a shader can do them on a
whole photo, and nothing in it is learned.

Align (flows()), on the lines of HDR+ and Handheld Multi-Frame Super-Resolution. Each frame becomes
a grey picture at half size, the mean of every 2x2 colour cell, kept to the band between FINE and
COARSE samples wide: finer is mostly noise, and coarser is brightness, which two exposures need not
share. That picture is halved again and again into a pyramid whose coarsest level is still a tile
of TILE samples across. On the coarsest level every tile is matched (matched()): it tries each
whole-sample move up to FAR either way and takes the one with the smallest mean difference, where
a move has to earn its distance, PULL of the difference for each sample moved, so that a tile with
nothing in it stays put. Then the fraction, on every level from the coarsest down (stepped()):
ROUNDS Lucas-Kanade steps, in which the frame is laid on the reference by the offsets so far and
each tile's offset moves by what the gradients ask, summed over the tile and its neighbours WIDE
tiles out, the nearer the more, with DAMP of the picture's mean gradient added so that a flat tile
does not move. A sample is left out of those sums where the two frames differ by more than SPARE
times their noise and what sitting STRAY samples off would explain: something that moved between
the frames is not where the camera went. Tiles are not matched again on the finer levels as HDR+
does: at high ISO a tile's own best whole move is often noise or moire, and nothing after it could
take that out, where steps summed over neighbours keep the offsets smooth, as a hand's shake is.
What comes out is one offset for each tile of 2 * TILE px, (dx, dy) in sensor px: what the
reference holds at a place the frame holds that far on. Between tile middles the offset is read
bilinearly, which covers a small turn. Where a thing as large as a tile moved, the offsets follow
it and come back to the camera's over a tile or two.

Trust (trusted()). A later frame is believed at a colour cell as far as it agrees there with the
reference, once laid on it: the two grey pictures without what is finer than FINE, their
difference squared and summed over NEAR cells a side, against what that sum should be, the two
frames' noise (variance = shot * brightness + read, never less than one photon's) and what sitting
SLACK cells off would make of the reference's gradients. It is believed fully up to LOW times that,
not at all from HIGH, and no more than the least trusted cell its sum was taken over. So print a
fraction of a pixel off is believed, its difference being what a small offset explains, and a
thing that moved, a tile the line-up missed or a change in brightness is not. A frame's weight is
its trust times the reference's noise over its own, so a noisier frame counts less: the weight is
in reference frames' worth of light.

Merge (gathered()). A mosaic cannot be moved by a fraction of a pixel without resampling each colour
from samples two pixels apart, which costs fine print its letters. So nothing is resampled: each
pixel of the reference looks up where it sits in a later frame and takes the four samples round
that place, each as the colour its own pixel measured, weighted by how near it is (a tent one pixel
wide) and by the frame's weight there. That gives, for each of red, green and blue at every pixel,
a sum and a weight (N, 3, H, W). The reference's own sample is always in at weight one, so a still
burst is the plain average, and a frame one pixel off hands over the two colours each pixel lacks.
The weight says how much was measured at each place: one frame's noise over its root is the noise
left there. mosaic() reads the merge at the colour each of the reference's pixels measured.

sharpest() picks the reference: the frame with the most detail finer than FINE samples, less what
its noise adds to that.
"""

import math

import torch

TILE = 32
FINE, COARSE = 1.0, 4.0
FAR, PULL = 4, 0.4
ROUNDS, DAMP, WIDE, SPARE, STRAY = 3, 0.02, 4, 3.0, 1.0
NEAR, LOW, HIGH, SLACK = 5, 3.0, 6.0, 0.25


def grey(mosaics):
    return torch.nn.functional.avg_pool2d(mosaics, 2)


def blurred(planes, width):
    reach = math.ceil(3 * width)
    taps = torch.exp(-0.5 * (torch.arange(-reach, reach + 1, dtype=planes.dtype, device=planes.device) / width) ** 2)
    taps = taps / taps.sum()
    wide = torch.nn.functional.pad(planes.flatten(0, 1)[:, None], (reach,) * 4, mode="reflect")
    across = torch.nn.functional.conv2d(wide, taps.view(1, 1, 1, -1))
    return torch.nn.functional.conv2d(across, taps.view(1, 1, -1, 1)).view_as(planes)


def band(planes):
    return blurred(planes, FINE) - blurred(planes, COARSE)


def places(height, width, like):
    y = torch.arange(height, dtype=like.dtype, device=like.device).view(-1, 1).expand(height, width)
    x = torch.arange(width, dtype=like.dtype, device=like.device).view(1, -1).expand(height, width)
    return x, y


def spread(tiles, side, height, width):
    """Tiles' values (N, C, tiles down, tiles across) at every sample (N, C, height, width), read
    bilinearly between the middles of tiles of that side."""
    return torch.nn.functional.interpolate(tiles, scale_factor=side, mode="bilinear")[..., :height, :width]


def laid(planes, offsets):
    """Planes (N, C, H, W) read at each place plus its offset (N, 2, H, W), bilinearly, and which
    places (N, 1, H, W) that lands inside the planes."""
    height, width = planes.shape[-2:]
    x, y = places(height, width, planes)
    at_x, at_y = x + offsets[:, 0], y + offsets[:, 1]
    grid = torch.stack([(at_x + 0.5) / width * 2 - 1, (at_y + 0.5) / height * 2 - 1], -1)
    there = torch.nn.functional.grid_sample(planes, grid, mode="bilinear", padding_mode="border", align_corners=False)
    inside = (at_x >= -0.5) & (at_x <= width - 0.5) & (at_y >= -0.5) & (at_y <= height - 0.5)
    return there, inside[:, None]


def pooled(planes):
    """Each tile's mean with its neighbours' WIDE tiles out, the nearer the more: sums for a tile's own offset."""
    tiles = torch.nn.functional.avg_pool2d(planes, TILE, ceil_mode=True)
    taps = torch.tensor([0.25, 0.5, 0.25], dtype=planes.dtype, device=planes.device)
    for _ in range(WIDE):
        wide = torch.nn.functional.pad(tiles.flatten(0, 1)[:, None], (1,) * 4, mode="replicate")
        across = torch.nn.functional.conv2d(wide, taps.view(1, 1, 1, 3))
        tiles = torch.nn.functional.conv2d(across, taps.view(1, 1, 3, 1)).view_as(tiles)
    return tiles


def matched(reference, frame, start, reach):
    """Each tile's whole-sample offset on one level of the pyramid (N, 2, tiles down, tiles across),
    from where it starts: the move up to reach either way with the smallest mean difference, each
    move's difference raised by PULL for every sample it goes."""
    height, width = reference.shape[-2:]
    each = start.repeat_interleave(TILE, 2).repeat_interleave(TILE, 3)[..., :height, :width].long()
    x, y = (place.long() for place in places(height, width, reference))
    steps = torch.arange(-reach, reach + 1, device=reference.device)
    moves = torch.stack([steps.repeat(len(steps)), steps.repeat_interleave(len(steps))], 1)
    across = (x + each[:, :1] + moves[:, 0].view(1, -1, 1, 1)).clamp(0, width - 1)
    down = (y + each[:, 1:] + moves[:, 1].view(1, -1, 1, 1)).clamp(0, height - 1)
    there = frame.flatten(1).gather(1, (down * width + across).flatten(1)).view_as(across)
    cost = torch.nn.functional.avg_pool2d((reference - there).abs(), TILE, ceil_mode=True)
    best = (cost * (1 + PULL * moves.to(cost).norm(dim=1).view(1, -1, 1, 1))).argmin(1)
    return start + moves.to(start)[best].permute(0, 3, 1, 2)


def passed(sieve, like):
    """How much of white noise's variance a filter lets through."""
    spike = like.new_zeros(1, 1, 65, 65)
    spike[..., 32, 32] = 1
    return sieve(spike).pow(2).sum()


def stepped(reference, frame, offsets, spare):
    """One Lucas-Kanade step on a level of the pyramid: each tile's offset (N, 2, tiles down, tiles
    across), in samples, moved by what the gradients of the frame laid on the reference ask, over
    the tile and its neighbours (pooled()). A sample counts where the frame has something to lay
    and the two differ by no more than SPARE times their noise there, spare (N, 1, H, W) its
    variance, and what sitting STRAY samples off would make of the gradient."""
    height, width = reference.shape[-2:]
    there, inside = laid(frame, spread(offsets, TILE, height, width))
    wide = torch.nn.functional.pad(there, (1,) * 4, mode="replicate")
    across, down = (wide[..., 1:-1, 2:] - wide[..., 1:-1, :-2]) / 2, (wide[..., 2:, 1:-1] - wide[..., :-2, 1:-1]) / 2
    rest = reference - there
    counted = inside & (rest * rest <= SPARE**2 * spare + STRAY**2 * (across * across + down * down))
    sums = pooled(torch.cat([across * across, across * down, down * down, across * rest, down * rest], 1) * counted)
    xx, xy, yy, xr, yr = sums.unbind(1)
    damp = DAMP * (xx + yy).mean((-2, -1), keepdim=True) + 1e-20
    xx, yy = xx + damp, yy + damp
    found = torch.stack([yy * xr - xy * yr, xx * yr - xy * xr], 1) / (xx * yy - xy * xy)[:, None]
    return offsets + found


def noise(levels, frames):
    """Each frame's shot and read noise (N, frames, 1, 1) each, from levels (frames, 2) or (N, frames, 2)."""
    levels = torch.as_tensor(levels, dtype=frames.dtype, device=frames.device)
    levels = levels.expand(len(frames), *levels.shape[-2:])
    return levels[..., 0, None, None], levels[..., 1, None, None]


def trusted(soft, offsets, spare, share):
    """How much each later frame counts at every colour cell (N, later frames, 1, H / 2, W / 2), in
    reference frames' worth of light: its trust times share, the reference's noise over its own.

    soft is each frame's grey picture without what is finer than FINE (N, frames, H / 2, W / 2),
    offsets each later frame's tiles' (N, later frames, 2, tiles down, tiles across) in sensor px,
    and spare the variance of the two frames' noise in such a picture (N, later frames, H / 2, W / 2).
    """
    n, frames, height, width = soft.shape
    later = frames - 1
    there, inside = laid(
        soft[:, 1:].reshape(n * later, 1, height, width), spread(offsets.flatten(0, 1), TILE, height, width) / 2
    )
    first = torch.nn.functional.pad(soft[:, :1], (1,) * 4, mode="replicate")
    across, down = (
        (first[..., 1:-1, 2:] - first[..., 1:-1, :-2]) / 2,
        (first[..., 2:, 1:-1] - first[..., :-2, 1:-1]) / 2,
    )
    box = lambda planes: torch.nn.functional.avg_pool2d(planes, NEAR, 1, NEAR // 2, count_include_pad=False)
    apart = box((there.view(n, later, height, width) - soft[:, :1]) ** 2)
    allowed = box(spare + SLACK**2 * (across * across + down * down))
    trust = ((HIGH - apart / allowed) / (HIGH - LOW)).clamp(0, 1) * inside.view(n, later, height, width)
    least = -torch.nn.functional.max_pool2d(-trust, NEAR, 1, NEAR // 2)
    return (least * share)[:, :, None]


def flows(frames, levels):
    """Where each later frame of bursts (N, frames, H, W) sits from the first, for each tile of
    2 * TILE px (N, later frames, 2, tiles down, tiles across), (dx, dy) in sensor px, and how much
    each later frame counts at every colour cell (trusted())."""
    n, later = len(frames), frames.shape[1] - 1
    shot, read = noise(levels, frames)
    fine = grey(frames)
    light = torch.nn.functional.avg_pool2d(fine[:, :1], NEAR, 1, NEAR // 2, count_include_pad=False)
    varied = shot * torch.maximum(light, shot) + read
    both = (varied[:, :1] + varied[:, 1:]) / 4
    depth = max(1, int(math.log2(min(fine.shape[-2:]) / TILE)) + 1)
    pyramid = [band(torch.nn.functional.avg_pool2d(fine, 2**level)) for level in range(depth)]
    pair = lambda level: (
        level[:, :1].expand(-1, later, -1, -1).reshape(n * later, 1, *level.shape[-2:]),
        level[:, 1:].reshape(n * later, 1, *level.shape[-2:]),
    )
    tiles = lambda level: [-(-side // TILE) for side in level.shape[-2:]]
    offsets = matched(*pair(pyramid[-1]), fine.new_zeros(n * later, 2, *tiles(pyramid[-1])), FAR)
    for at in range(depth - 1, -1, -1):
        reference, frame = pair(pyramid[at])
        spare = passed(band, fine) * torch.nn.functional.avg_pool2d(both, 2**at).reshape(
            n * later, 1, *reference.shape[-2:]
        )
        if at < depth - 1:
            offsets = 2 * torch.nn.functional.interpolate(offsets, size=tiles(pyramid[at]), mode="bilinear")
        for _ in range(ROUNDS):
            offsets = stepped(reference, frame, offsets, spare / 4**at)
    sat = 2 * offsets.view(n, later, 2, *offsets.shape[-2:])
    soften = lambda planes: blurred(planes, FINE)
    return sat, trusted(soften(fine), sat, passed(soften, fine) * both, varied[:, :1] / varied[:, 1:])


def gathered(frames, offsets, weights):
    """The merge of bursts (N, frames, H, W): for red, green and blue at every pixel, the sum of the
    samples that sit there and the sum of their weights, (N, 3, H, W) each.

    Each pixel of the first frame takes its own sample at weight one, and from each later frame the
    four samples round where it sits there by offsets (flows()), each as the colour its pixel
    measured, weighted by a tent one pixel wide and by the frame's weight at that cell (trusted()).
    """
    n, count, height, width = frames.shape
    x, y = places(height, width, frames)
    colour = lambda across, down: across.long() % 2 + down.long() % 2
    total, weight = frames.new_zeros(2, n, 3, height, width).unbind(0)
    own = colour(x, y).expand(n, 1, height, width)
    total.scatter_add_(1, own, frames[:, :1])
    weight.scatter_add_(1, own, torch.ones_like(frames[:, :1]))
    if count == 1:
        return total, weight
    later = frames[:, 1:].reshape(-1, height * width)
    each = spread(offsets.flatten(0, 1), 2 * TILE, height, width)
    share = weights.flatten(0, 2).repeat_interleave(2, 1).repeat_interleave(2, 2)
    at_x, at_y = x + each[:, 0], y + each[:, 1]
    sums = frames.new_zeros(2, len(later), 3, height, width)
    for down in (0, 1):
        for across in (0, 1):
            tap_x, tap_y = at_x.floor() + across, at_y.floor() + down
            inside = (tap_x >= 0) & (tap_x < width) & (tap_y >= 0) & (tap_y < height)
            counted = (1 - (at_x - tap_x).abs()) * (1 - (at_y - tap_y).abs()) * inside * share
            tap_x, tap_y = tap_x.clamp(0, width - 1), tap_y.clamp(0, height - 1)
            sample = later.gather(1, (tap_y * width + tap_x).long().flatten(1)).view_as(counted)
            sums[0].scatter_add_(1, colour(tap_x, tap_y)[:, None], (counted * sample)[:, None])
            sums[1].scatter_add_(1, colour(tap_x, tap_y)[:, None], counted[:, None])
    sums = sums.view(2, n, count - 1, 3, height, width).sum(2)
    return total + sums[0], weight + sums[1]


def merged(frames, levels):
    """Bursts (N, frames, H, W), the reference first, lined up and merged: gathered()'s sums and
    weights. levels is each frame's shot and read noise, (frames, 2) or (N, frames, 2), on the
    frames' own scale: variance = shot * brightness + read."""
    return gathered(frames, *(flows(frames, levels) if frames.shape[1] > 1 else (None, None)))


def mosaic(total, weight):
    """A merge as the mosaic (N, H, W) its reference is: each pixel the merged colour it measured."""
    n, _, height, width = total.shape
    x, y = places(height, width, total)
    colour = (x.long() % 2 + y.long() % 2).expand(n, height, width)[:, None]
    return (total.gather(1, colour) / weight.gather(1, colour))[:, 0]


def sharpest(frames, levels):
    """Which frame of each burst (N, frames, H, W) holds the most fine detail (N,): the energy of its
    grey picture finer than FINE samples, less what its noise puts there."""
    shot, read = noise(levels, frames)
    fine = grey(frames)
    finest = lambda planes: planes - blurred(planes, FINE)
    laid_on = passed(finest, fine) * (shot * fine.mean((-2, -1), keepdim=True).clamp(min=0) + read) / 4
    return (finest(fine).pow(2).mean((-2, -1)) - laid_on.flatten(1)).argmax(1)
