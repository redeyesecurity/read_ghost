"""
ghostreader: read a "ghost font" that only machines (and motion-sensitive
human vision) can see.

A ghost-font clip looks like pure static in any single frame, and its
time-average is also flat noise. The letters exist ONLY in coherent pixel
motion: inside a glyph every pixel drifts the same way, in the background the
drift is random. Human vision integrates common motion and "sees" the words;
a still image cannot.

We recover them the machine way: estimate the per-pixel optical-flow vector for
every consecutive frame pair (Lucas-Kanade), then average those vectors over the
whole clip. Random background motion cancels toward zero; coherent glyph motion
survives. The magnitude of the averaged flow field is the hidden text.

No OpenCV required, no AI required. numpy + scipy + ffmpeg only.
"""

import os
import subprocess
import tempfile
import glob

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, gaussian_filter


def extract_frames(source, out_dir, fps=30, max_seconds=None):
    """Pull frames from a local path OR a URL (ffmpeg reads http(s) directly)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", source, "-vf", f"fps={fps}", os.path.join(out_dir, "f_%05d.png")]
    subprocess.run(cmd, check=True)
    frames = sorted(glob.glob(os.path.join(out_dir, "f_*.png")))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames. Is the source a valid video?")
    return frames


def load_stack(frames, downscale=1):
    """Load frames as a grayscale float32 stack (N, H, W)."""
    imgs = []
    for f in frames:
        im = Image.open(f).convert("L")
        if downscale > 1:
            im = im.resize((im.width // downscale, im.height // downscale))
        imgs.append(np.asarray(im, dtype=np.float32))
    return np.stack(imgs)


def coherent_motion_map(imgs, win=11):
    """
    Average Lucas-Kanade optical flow over all frame pairs, return the
    magnitude of the mean flow vector per pixel. Glyphs survive the average;
    incoherent background cancels.
    """
    N, H, W = imgs.shape
    vx_sum = np.zeros((H, W), dtype=np.float32)
    vy_sum = np.zeros((H, W), dtype=np.float32)
    n = 0
    for i in range(N - 1):
        I0, I1 = imgs[i], imgs[i + 1]
        Ix = np.gradient(I0, axis=1)
        Iy = np.gradient(I0, axis=0)
        It = I1 - I0
        Axx = uniform_filter(Ix * Ix, win)
        Ayy = uniform_filter(Iy * Iy, win)
        Axy = uniform_filter(Ix * Iy, win)
        Axt = uniform_filter(Ix * It, win)
        Ayt = uniform_filter(Iy * It, win)
        det = Axx * Ayy - Axy * Axy + 1e-3
        vx_sum += -(Ayy * Axt - Axy * Ayt) / det
        vy_sum += -(Axx * Ayt - Axy * Axt) / det
        n += 1
    return np.hypot(vx_sum / n, vy_sum / n)


def to_image(mag, invert=True, denoise=1.0):
    """Percentile-normalize the motion map into a clean readable PNG array."""
    if denoise:
        mag = gaussian_filter(mag, denoise)
    lo, hi = np.percentile(mag, [2, 98])
    m = np.clip((mag - lo) / (hi - lo + 1e-9), 0, 1)
    if invert:  # text is dark on white, easier to read
        m = 1 - m
    return (m * 255).astype("uint8")


def decode(source, out_path=None, fps=30, win=11, downscale=1,
           max_seconds=None, invert=True, denoise=1.0):
    """
    End-to-end: video path OR url -> decoded PNG (numpy uint8 array).
    Returns the array; also writes to out_path if given.
    """
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(source, tmp, fps=fps, max_seconds=max_seconds)
        imgs = load_stack(frames, downscale=downscale)
        mag = coherent_motion_map(imgs, win=win)
    out = to_image(mag, invert=invert, denoise=denoise)
    if out_path:
        Image.fromarray(out).save(out_path)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Read a ghost font from a video (path or URL).")
    p.add_argument("source", help="video file path or http(s) URL")
    p.add_argument("-o", "--out", default="decoded.png", help="output PNG path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--win", type=int, default=11, help="flow smoothing window")
    p.add_argument("--downscale", type=int, default=1, help="speed: 2 = half size")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--no-invert", action="store_true", help="light text on dark")
    args = p.parse_args()

    decode(args.source, out_path=args.out, fps=args.fps, win=args.win,
           downscale=args.downscale, max_seconds=args.max_seconds,
           invert=not args.no_invert)
    print(f"wrote {args.out}")
