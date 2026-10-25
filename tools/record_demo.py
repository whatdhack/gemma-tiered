"""Record docs/demo.gif: a terminal session driving the live device panel.

Phase 1 runs every demo prompt through the real TieredRouter (Tier 1 FunctionGemma,
Tier 2 Ollama) and caches the results. Phase 2 replays those results into the device
panel and a mock terminal in headless Chrome, capturing frames into a GIF. The
replay compresses wall-clock waits (e.g. a ~27 s Tier 2 report) but every output
and latency shown is the real one.

Usage:
    python tools/record_demo.py            # run router, then record
    python tools/record_demo.py --reuse    # re-record from cached router results

Requires: pip install playwright pillow  (uses the installed Chrome; no browser download)
"""

import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from playwright.sync_api import sync_playwright

from device_panel import DevicePanel
from tiered_router import RouteResult, TieredRouter, format_result

PROMPTS = [
    ("Set primary motor to 1400 RPM", 3.0),
    ("Read sensor temp_01", 1.8),
    ("Adjust motor speed to 2500 RPM", 3.0),
    ("Read sensor pressure_valve_B", 1.8),
    ("Emergency halt right now", 2.6),
    ("give me a status report", 4.5),
]
CACHE = Path(tempfile.gettempdir()) / "gemma_tiered_demo_results.json"
OUT = ROOT / "docs" / "demo.gif"
VIEWPORT = {"width": 1600, "height": 900}
GIF_WIDTH = 960
MIN_FRAME_MS = 140
TYPE_MS = 45

STAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { margin: 0; background: #05080c; display: flex; height: 100vh; overflow: hidden;
         font-family: "Segoe UI", system-ui, sans-serif; }
  .term { width: 600px; flex: none; margin: 16px 0 16px 16px; border-radius: 10px; overflow: hidden;
          background: #0c0c0c; border: 1px solid #2a2a2a; display: flex; flex-direction: column; }
  .bar { background: #1f1f1f; color: #aaa; font-size: 12px; padding: 8px 12px; display: flex; gap: 8px; align-items: center; }
  .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
  #term { flex: 1; padding: 12px 14px; overflow: hidden; color: #d4d4d4;
          font: 13px/1.45 "Cascadia Mono", Consolas, monospace; white-space: pre-wrap; }
  .ps { color: #6cb6ff; }
  .typed { color: #fff; }
  .cursor { color: #fff; animation: blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .t1 { color: #2fe08b; }
  .t2 { color: #ffb547; }
  .dim { color: #777; }
  .wrap { flex: 1; overflow: hidden; position: relative; }
  iframe { border: 0; width: 1300px; height: 1180px; transform: scale(0.75); transform-origin: 0 0;
           position: absolute; left: 8px; top: 0; }
</style></head><body>
  <div class="term">
    <div class="bar"><span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span>
      <span class="dot" style="background:#28c840"></span>&nbsp; PowerShell · python tiered_router.py --panel</div>
    <div id="term"></div>
  </div>
  <div class="wrap"><iframe src="__PANEL_URL__"></iframe></div>
<script>
  const term = document.getElementById("term");
  let cur = null;
  function println(s, cls) {
    const d = document.createElement("div");
    if (cls) d.className = cls;
    d.textContent = s;
    term.appendChild(d);
    term.scrollTop = term.scrollHeight;
  }
  function startPrompt() {
    cur = document.createElement("div");
    cur.innerHTML = '<span class="ps">Enter prompt &gt; </span><span class="typed"></span><span class="cursor">█</span>';
    term.appendChild(cur);
    term.scrollTop = term.scrollHeight;
  }
  function typeText(s, ms) {
    const t = cur.querySelector(".typed");
    let i = 0;
    const iv = setInterval(() => {
      t.textContent += s[i++];
      if (i >= s.length) clearInterval(iv);
    }, ms);
  }
  function endPrompt() {
    const c = cur.querySelector(".cursor");
    if (c) c.remove();
  }
</script></body></html>"""


def collect() -> dict:
    router = TieredRouter()
    data = {"status": router.tier_status(), "steps": []}
    for prompt, _ in PROMPTS:
        print(f"routing: {prompt}")
        res = router.route(prompt)
        data["steps"].append({"result": res.model_dump(), "text": format_result(res)})
    CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def record(data: dict) -> None:
    panel = DevicePanel(port=8799)
    panel.start()
    frames, stamps = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport=VIEWPORT)
        page.set_content(STAGE_HTML.replace("__PANEL_URL__", panel.url))
        page.frame_locator("iframe").locator("#conn.ok").wait_for(timeout=10000)
        page.wait_for_timeout(500)

        def capture(seconds: float) -> None:
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                stamps.append(time.perf_counter())
                shot = Image.open(io.BytesIO(page.screenshot(type="jpeg", quality=90)))
                h = round(shot.height * GIF_WIDTH / shot.width)
                frames.append(shot.convert("RGB").resize((GIF_WIDTH, h), Image.LANCZOS))

        page.evaluate("s => println(s, 'dim')", "--- Interactive Tiered Routing Console (Type 'exit' to quit) ---")
        page.evaluate("s => println(s)", data["status"])
        page.evaluate("s => println(s, 'dim')", f"Device panel: http://127.0.0.1:8765/")
        capture(1.2)

        for (prompt, hold), step in zip(PROMPTS, data["steps"]):
            res = RouteResult(**step["result"])
            page.evaluate("startPrompt()")
            page.evaluate("([s, ms]) => typeText(s, ms)", [prompt, TYPE_MS])
            capture(len(prompt) * TYPE_MS / 1000 + 0.35)
            page.evaluate("endPrompt()")

            is_t1 = res.tier_selected == "Tier 1 (Small Brain)"
            if not is_t1:
                page.evaluate("s => println(s, 'dim')",
                              f"(Tier 2 thinking... real latency {res.latency_ms / 1000:.1f} s, shortened in this GIF)")
                capture(1.0)
            panel.publish(res)
            page.evaluate("([s, c]) => println(s, c)", [step["text"], "t1" if is_t1 else "t2"])
            page.evaluate("s => println(s)", "")
            capture(hold)

        browser.close()

    stamps.append(time.perf_counter())
    durations = [max(20, round((b - a) * 1000)) for a, b in zip(stamps, stamps[1:])]
    durations[-1] = 2500  # linger on the final frame before looping
    write_gif(frames, durations)


def write_gif(frames: list, durations: list) -> None:
    # Cap at ~7 fps by folding frames shown < MIN_FRAME_MS after the last kept one
    kept, durs = [], []
    for f, d in zip(frames, durations):
        if kept and durs[-1] < MIN_FRAME_MS:
            durs[-1] += d
            continue
        kept.append(f)
        durs.append(d)

    # One shared palette sampled across the run -> stable colors and small inter-frame deltas
    h = kept[0].height
    sample = Image.new("RGB", (GIF_WIDTH, h * 4))
    for k, idx in enumerate([0, len(kept) // 3, 2 * len(kept) // 3, len(kept) - 1]):
        sample.paste(kept[idx], (0, k * h))
    pal = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in kept]

    OUT.parent.mkdir(exist_ok=True)
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
    print(f"wrote {OUT} ({len(q)} frames, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if "--reuse" in sys.argv and CACHE.exists():
        results = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        results = collect()
    record(results)
