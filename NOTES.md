# Build notes: how we taught a machine to read a ghost font

Date: 2026-07-14. Session notes from the actual investigation, kept honest, dead ends included.

## The challenge

[mixfont.com/ghost-font](https://www.mixfont.com/ghost-font) generates videos where text is
"written" purely in moving noise. The pitch: humans can read it, machines can't, because
there is nothing to OCR. Screenshot it and the text is gone.

We were handed a 6-second, 1280x720, ~30fps clip and told it said something. Goal: make a
machine read it, then package the method so anyone can run it.

## What the signal actually is

Three checks established the encoding before any decoding was attempted:

1. **Single frame:** full-range noise (min 4, max 252, mean ~182, std ~94). Looks like TV
   static with a faint vertical stripe texture. Nothing readable.
2. **Time-average of all 187 frames:** converges to the stripe texture. Only the faintest
   ghost of letterforms, unreadable. So the text is NOT a brightness bias, or averaging
   would reveal it cleanly.
3. **Per-frame motion energy** (squared frame difference, box-filtered): uniform across the
   whole image. Everything flickers equally hard, letters and background alike. So the text
   is NOT "moving regions vs still regions" either.

Conclusion: the glyphs are encoded in **motion direction coherence**. Inside a letter,
pixels drift in a consistent direction over time. In the background, the motion direction
is random per pixel per frame. The *amount* of motion is identical everywhere, which is what
kills naive motion detection. Human vision reads it because the visual system integrates
common motion across space and time (the same mechanism that lets you spot a camouflaged
animal the instant it walks).

## Dead ends, in order

- **Temporal mean image:** noise. (Expected, confirmed.)
- **Temporal std image:** flat. Every pixel varies equally.
- **Frame-pair motion energy:** flat. Flicker is uniform by design.
- **Averaged 1-D horizontal flow:** first flow attempt assumed the glyph drift was
  horizontal and constant. The average came out near zero. The drift direction is not
  constant over the clip (it wanders), so a straight directional average partially cancels.

## What worked

Full 2-D Lucas-Kanade optical flow per consecutive frame pair, then average the flow
*vectors* over all 186 pairs, then take the **magnitude** of the mean vector per pixel:

```
for each frame pair (I0, I1):
    Ix, Iy = spatial gradients of I0
    It     = I1 - I0
    solve the 2x2 LK system per pixel over an 11px window:
        [ Σ Ix²   Σ IxIy ] [vx]   [ Σ IxIt ]
        [ Σ IxIy  Σ Iy²  ] [vy] = -[ Σ IyIt ]
    accumulate (vx, vy)

result = |mean flow vector|      # hypot(mean vx, mean vy)
```

Background pixels get a different random direction each pair, so their vector sum shrinks
toward zero. Glyph pixels share direction with their neighbors and drift consistently enough
that their mean vector stays large. Percentile-clip the magnitude map to [2, 98], normalize,
invert, and the words are simply legible in the output PNG.

First clip decoded to **HELLO HUMAN**. Second clip decoded to **ETAIROS CAN SEE**.

Key implementation notes:

- The 11px uniform window on the LK structure tensor is what couples neighboring pixels; it
  is the spatial half of "common fate." Bigger windows give smoother, fatter strokes.
- No OpenCV. `np.gradient` + `scipy.ndimage.uniform_filter` implement LK in ~20 lines.
- A small Gaussian blur (sigma 1) on the final map plus percentile normalization is all the
  cleanup required.
- Runs on CPU in seconds for a 6s 720p clip.

## Does it use AI?

**No.** This is the part worth teaching.

- The decoder is Lucas-Kanade optical flow, published in **1981**, plus vector averaging.
  It is deterministic linear algebra: same input video, same output image, every time, on
  any machine. There is no model, no weights, no training, no inference.
- An AI assistant *designed and debugged* the decoder: formed hypotheses about the
  encoding, ran the three signal checks above, discarded the dead ends, and wrote the code.
  It also read the final decoded image, but so can any human, the output is a plain picture
  of the words.
- If you want a fully unattended pipeline (video in, string out), bolt any OCR onto the
  output PNG. That step could use ML, but it is optional and boring; the ghost-breaking
  itself is classical.

So the honest summary: **AI figured out the method; the method itself needs no AI.** The
"only humans can read this" property lasted about twenty minutes of directed analysis. Any
scheme whose security rests on "machines don't have this human perceptual channel" should
assume the channel can be mechanized, usually with decades-old math.

## Files

- `ghostreader.py` - decoder library + CLI (file path or URL)
- `app.py` - Flask one-pager: paste a URL or drop a file, see the decoded text
- `samples/` - decoded outputs from the two test clips
