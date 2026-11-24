"""The frozen text network train_raw.py --reader compares laid print through: uv run text_reader.py

CRAFT (Baek et al. 2019, github.com/clovaai/CRAFT-pytorch, MIT licence, Copyright (c) 2019-present
NAVER Corp.) finds text in photos: VGG-16 with batch normalisation under a decoder that marks
letters and the gaps that join them. Its general model was trained on SynthText, ICDAR 2013 and
ICDAR 2017, which is nine languages' print in photos. This keeps its backbone alone, as far as the
third of the four depths the decoder reads it at, and nothing of the decoder: what is left turns a
picture into planes that follow strokes and letters' shapes and never says whether a mark is a
letter. It is no reader that small_print.py scores with.

The authors' own link is a Google Drive file that scripts are turned away from, so the weights come
from EasyOCR's copy of that file, craft_mlt_25k.pth, in its releases on GitHub.

Writes data/noise/reader.pth, 21 MB: the layers of torchvision's vgg16_bn().features up to TAPS'
last, under torchvision's names.
"""

import io
import urllib.request
import zipfile

import torch
import torchvision

import raw_pairs

OUT = raw_pairs.DATA / "noise" / "reader.pth"
ZIP = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
#: Where CRAFT's decoder reads the backbone, as layers of vgg16_bn().features: conv2_2, conv3_2 and
#: conv4_2, each normalised and through its ReLU, at a half, a quarter and an eighth of the picture's size.
TAPS = (12, 19, 29)
#: What CRAFT takes from a picture of 0 to 1 before anything else, colour by colour.
MEAN, SPREAD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def layers():
    return torchvision.models.vgg16_bn().features[: TAPS[-1] + 1]


def network():
    """The backbone with its weights, frozen."""
    found = layers()
    found.load_state_dict(torch.load(OUT))
    return found.cuda().eval().requires_grad_(False)


def fetch():
    with urllib.request.urlopen(ZIP) as packed:
        whole = torch.load(io.BytesIO(zipfile.ZipFile(io.BytesIO(packed.read())).read("craft_mlt_25k.pth")), "cpu")
    first = tuple(f"module.basenet.slice{part}." for part in (1, 2, 3))
    found = layers()
    found.load_state_dict({name.split(".", 3)[3]: values for name, values in whole.items() if name.startswith(first)})
    torch.save(found.state_dict(), OUT)


if __name__ == "__main__":
    fetch()
