"""Generate AlphaPulse icon (.png + .ico) and login-window background.

Theme: a stylized lightning bolt over a candlestick wave — represents the
sudden directional pulses (WINNER / LOSER) that the strategy hunts.

Run once before building the executable; the produced files are committed to
``crypto_trend/desktop/assets`` so PyInstaller can embed them.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).resolve().parents[1] / "crypto_trend" / "desktop" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
def _gradient(size: tuple[int, int], top: tuple[int, int, int],
              bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img


# --------------------------------------------------------------------------- #
def make_icon(size: int = 512) -> Image.Image:
    """Round-rect indigo gradient with white waveform + golden lightning."""
    bg = _gradient((size, size), (38, 84, 240), (12, 36, 140)).convert("RGBA")

    # Round-rect mask
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, size - 1, size - 1),
                         radius=int(size * 0.22), fill=255)
    bg.putalpha(mask)

    draw = ImageDraw.Draw(bg)

    # Subtle horizontal candlestick band
    band_y = int(size * 0.62)
    for x in range(int(size * 0.08), int(size * 0.92), int(size * 0.06)):
        h = int(size * (0.06 + 0.04 * (math.sin(x * 0.05) + 1)))
        color = (31, 200, 130, 220) if (x // int(size * 0.06)) % 2 == 0 \
                else (235, 90, 100, 220)
        draw.rectangle((x, band_y - h, x + int(size * 0.025), band_y + h),
                       fill=color)
        draw.line((x + int(size * 0.012), band_y - int(h * 1.3),
                   x + int(size * 0.012), band_y + int(h * 1.3)),
                  fill=color, width=max(1, size // 256))

    # White waveform on top
    pts = []
    for i in range(0, size, max(1, size // 200)):
        y = int(size * 0.40
                + math.sin(i * 0.025) * size * 0.05
                + math.sin(i * 0.07) * size * 0.02)
        pts.append((i, y))
    draw.line(pts, fill=(255, 255, 255, 230), width=max(2, size // 96))

    # Lightning bolt — golden, centered, slight tilt
    cx, cy = size * 0.50, size * 0.50
    s = size * 0.36
    bolt = [
        (cx - 0.22 * s, cy - 0.95 * s),
        (cx - 0.55 * s, cy + 0.10 * s),
        (cx - 0.10 * s, cy + 0.10 * s),
        (cx - 0.40 * s, cy + 0.95 * s),
        (cx + 0.55 * s, cy - 0.20 * s),
        (cx + 0.05 * s, cy - 0.20 * s),
        (cx + 0.30 * s, cy - 0.95 * s),
    ]
    glow = bg.copy()
    gd = ImageDraw.Draw(glow)
    gd.polygon(bolt, fill=(255, 220, 70, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(size // 96))
    bg.alpha_composite(glow)

    draw = ImageDraw.Draw(bg)
    draw.polygon(bolt, fill=(255, 230, 90, 255),
                 outline=(255, 255, 255, 255))

    return bg


# --------------------------------------------------------------------------- #
def make_background(size: tuple[int, int] = (1600, 1000)) -> Image.Image:
    """Light gradient with subtle wave dot overlay."""
    bg = _gradient(size, (246, 249, 252), (235, 240, 248))
    draw = ImageDraw.Draw(bg)
    w, h = size
    # decorative low-alpha sine bands
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for amplitude, freq, phase, alpha in [
        (60, 0.0035, 0.0, 18),
        (45, 0.0060, 1.4, 14),
        (30, 0.0090, 2.7, 10),
    ]:
        pts = []
        for x in range(0, w, 4):
            y = h * 0.55 + math.sin(x * freq + phase) * amplitude
            pts.append((x, y))
        od.line(pts, fill=(13, 110, 253, alpha), width=8)

    overlay = overlay.filter(ImageFilter.GaussianBlur(6))
    bg.paste(overlay, (0, 0), overlay)

    # tiny dot grid
    dd = ImageDraw.Draw(bg)
    for y in range(0, h, 28):
        for x in range(0, w, 28):
            dd.ellipse((x, y, x + 1, y + 1), fill=(210, 218, 230))
    return bg


# --------------------------------------------------------------------------- #
def main() -> None:
    icon = make_icon(512)
    icon.save(ASSETS / "icon.png", "PNG", optimize=True)

    # Multi-resolution .ico
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    icons = [icon.resize(s, Image.LANCZOS) for s in sizes]
    icons[0].save(ASSETS / "icon.ico", format="ICO",
                  sizes=sizes, append_images=icons[1:])

    bg = make_background()
    bg.save(ASSETS / "background.png", "PNG", optimize=True)

    print(f"wrote {ASSETS / 'icon.png'}  ({icon.size})")
    print(f"wrote {ASSETS / 'icon.ico'}")
    print(f"wrote {ASSETS / 'background.png'}  ({bg.size})")


if __name__ == "__main__":
    main()
