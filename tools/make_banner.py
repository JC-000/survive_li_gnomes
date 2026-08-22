#!/usr/bin/env python3
"""Redraw docs/assets/banner.png: e-paper aesthetic, 2560x1280 (2:1).

Used as the README header and the GitHub social preview (the latter is
uploaded by hand: Settings -> General -> Social preview; there is no API).
Run with `uv run tools/make_banner.py`; needs macOS for Menlo.

Everything is drawn as ink-on-paper: black #101010 on e-paper grey-white,
with Bayer ordered dithering for the 8-ball's sphere shading.
"""
# /// script
# dependencies = ["pillow", "numpy"]
# ///
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 2560, 1280
PAPER = (233, 231, 224)   # e-paper "white": slightly grey, slightly warm
INK = (16, 16, 16)

img = Image.new("RGB", (W, H), PAPER)
d = ImageDraw.Draw(img)

MENLO = "/System/Library/Fonts/Menlo.ttc"


def font(size, bold=False):
    return ImageFont.truetype(MENLO, size, index=1 if bold else 0)


# ---------------------------------------------------------------- bezel frame
# Double frame like the panel's bezel: outer thick, inner thin.
d.rectangle([0, 0, W - 1, H - 1], outline=INK, width=10)
d.rectangle([26, 26, W - 27, H - 27], outline=INK, width=3)

# ---------------------------------------------------------------- 8-ball (right)
# Dithered sphere: radial gradient -> Bayer 8x8 ordered dither -> 1-bit.
CX, CY, R = 1985, 555, 330

BAYER8 = (1 / 64) * np.array(
    [[ 0, 32,  8, 40,  2, 34, 10, 42],
     [48, 16, 56, 24, 50, 18, 58, 26],
     [12, 44,  4, 36, 14, 46,  6, 38],
     [60, 28, 52, 20, 62, 30, 54, 22],
     [ 3, 35, 11, 43,  1, 33,  9, 41],
     [51, 19, 59, 27, 49, 17, 57, 25],
     [15, 47,  7, 39, 13, 45,  5, 37],
     [63, 31, 55, 23, 61, 29, 53, 21]])

yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
inside = dist <= R
# Sphere shading: highlight up-left, dark rim. value = brightness 0..1
lx, ly = CX - R * 0.42, CY - R * 0.45
hdist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / (2.1 * R)
val = np.clip(0.92 - hdist * 1.55, 0.0, 0.55)  # mostly dark ball, soft highlight
thr = np.tile(BAYER8, (H // 8 + 1, W // 8 + 1))[:H, :W]
ballmask = inside & (val <= thr)  # True -> ink

arr = np.array(img)
arr[ballmask] = INK
img = Image.fromarray(arr)
d = ImageDraw.Draw(img)

# Ball outline, crisp
d.ellipse([CX - R, CY - R, CX + R, CY + R], outline=INK, width=8)

# The white window with the 8
wr = 150
d.ellipse([CX - wr, CY - wr, CX + wr, CY + wr], fill=PAPER, outline=INK, width=8)
f8 = font(230, bold=True)
d.text((CX, CY - 8), "8", font=f8, fill=INK, anchor="mm")

# ------------------------------------------------------- speech bubble (ELIZA)
# The 8-ball answers with DOCTOR's fallback line.
bx0, by0, bx1, by1 = 1555, 935, 2445, 1105
# tail first (paper fill, ink outline), then the bubble covers its base
d.polygon([(2000, by0 + 20), (2110, by0 + 20), (2085, by0 - 72)],
          fill=PAPER, outline=INK, width=8)
d.rounded_rectangle([bx0, by0, bx1, by1], radius=18, fill=PAPER, outline=INK, width=8)
fb = font(58)
d.text(((bx0 + bx1) // 2, (by0 + by1) // 2), "PLEASE GO ON.", font=fb, fill=INK, anchor="mm")

# ---------------------------------------------------------------- text (left)
LX = 120
ft = font(148, bold=True)
d.text((LX, 210), "survive_li_gnomes", font=ft, fill=INK, anchor="lm")

# rule
d.line([(LX, 330), (1635, 330)], fill=INK, width=6)

f1 = font(62, bold=True)
d.text((LX, 440), "The Lithium Gnomes stole the", font=f1, fill=INK, anchor="lm")
d.text((LX, 525), "world's electrolyte.", font=f1, fill=INK, anchor="lm")

f2 = font(46)
d.text((LX, 680), "A Magic 8-Ball for leadership,", font=f2, fill=INK, anchor="lm")
d.text((LX, 745), "a talking ELIZA for companionship,", font=f2, fill=INK, anchor="lm")
d.text((LX, 810), "on e-paper that keeps the answer", font=f2, fill=INK, anchor="lm")
d.text((LX, 875), "when the power dies.", font=f2, fill=INK, anchor="lm")

# ---------------------------------------------------------------- spec strip
sy = 1130
d.line([(LX, sy - 60), (1440, sy - 60)], fill=INK, width=4)
f3 = font(40)
d.text((LX, sy), "RP2350 · MicroPython · 200×200 e-paper", font=f3, fill=INK, anchor="lm")
d.text((LX, sy + 58), "30 KB keyword CNN · zero flash writes", font=f3, fill=INK, anchor="lm")

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets" / "banner.png"
img.save(OUT, optimize=True)
print("wrote", OUT, img.size)
