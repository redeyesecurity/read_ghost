"""
Ghost Font Reader: a tiny web page.

Paste a URL to a ghost-font video (or drop a local file), and the machine
reads what only motion-sensitive eyes can see.

    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000
"""

import base64
import io
import os
import tempfile

from flask import Flask, request, render_template_string
from PIL import Image

import ghostreader

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghost Font Reader</title>
<style>
  :root { --ink:#141414; --paper:#f7f5f0; --red:#c8102e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink);
         font-family:'Space Grotesk',ui-sans-serif,system-ui,sans-serif;
         display:flex; min-height:100vh; align-items:flex-start; justify-content:center; }
  main { width:min(780px,92vw); padding:56px 0 80px; }
  h1 { font-size:15px; letter-spacing:.28em; text-transform:uppercase; font-weight:600; margin:0 0 6px; }
  .sub { color:#6b6b6b; font-size:14px; line-height:1.5; margin:0 0 28px; max-width:60ch; }
  form { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  input[type=url], input[type=file] { flex:1 1 320px; padding:12px 14px; border:1px solid #d8d4cc;
         background:#fff; font-size:14px; border-radius:2px; }
  button { padding:12px 22px; border:none; background:var(--ink); color:var(--paper);
         text-transform:uppercase; letter-spacing:.18em; font-size:12px; cursor:pointer; border-radius:2px; }
  button:hover { background:var(--red); }
  .row { display:flex; gap:24px; align-items:center; margin-top:14px; color:#6b6b6b; font-size:13px; }
  .err { color:var(--red); margin-top:20px; font-size:14px; }
  figure { margin:34px 0 0; }
  figure img { width:100%; border:1px solid #e2ded6; border-radius:2px; display:block; }
  figcaption { color:#6b6b6b; font-size:12px; letter-spacing:.14em; text-transform:uppercase; margin-top:10px; }
  .note { margin-top:44px; padding-top:22px; border-top:1px solid #e2ded6; color:#8a8a8a; font-size:12.5px; line-height:1.6; }
  a { color:var(--red); }
</style>
</head>
<body>
<main>
  <h1>Ghost Font Reader</h1>
  <p class="sub">A ghost font hides text in pixel motion. Any single frame is static; the
  time-average is flat noise. Only coherent motion carries the letters, so eyes read it and a
  still image cannot. This machine estimates per-pixel optical flow across the clip and keeps
  what moves together. Give it a video URL or file.</p>

  <form method="post" enctype="multipart/form-data">
    <input type="url" name="url" placeholder="https://example.com/ghost.mp4" value="{{url or ''}}">
    <input type="file" name="file" accept="video/*">
    <button type="submit">Read it</button>
  </form>

  {% if error %}<p class="err">{{error}}</p>{% endif %}

  {% if result %}
  <figure>
    <img src="data:image/png;base64,{{result}}" alt="decoded ghost text">
    <figcaption>Decoded: what the machine sees</figcaption>
  </figure>
  {% endif %}

  <p class="note">Method: Lucas-Kanade optical flow per frame pair, averaged over the whole
  clip. Random background motion cancels toward zero; coherent glyph motion survives. The
  magnitude of the averaged flow field is the hidden text. Machines can be taught to read.</p>
</main>
</body>
</html>
"""


def _encode(arr):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)

    url = (request.form.get("url") or "").strip()
    upload = request.files.get("file")
    source, tmp_path = None, None
    try:
        if upload and upload.filename:
            suffix = os.path.splitext(upload.filename)[1] or ".mp4"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            upload.save(tmp_path)
            source = tmp_path
        elif url:
            source = url
        else:
            return render_template_string(PAGE, error="Give a URL or choose a file.")

        arr = ghostreader.decode(source)
        return render_template_string(PAGE, result=_encode(arr), url=url)
    except Exception as e:  # noqa: BLE001, surface any failure to the page
        return render_template_string(PAGE, error=f"Could not read that: {e}", url=url)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
