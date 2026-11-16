"""Measure each slide of a day 2 deck and write <session>/build/fit.json: the zoom that lets the slide body fill the frame.

    python3 build/fit_session.py nlp        (or: llm; needs playwright)

Run after assemble_session.py, then run assemble_session.py again, which applies the zooms to each slide's .wrap.
A slide gets the largest zoom from 1.50 down to 1.00 at which nothing overflows and the lowest painted box
stays above 93% of the frame.
"""
import asyncio, importlib, json, os, sys, urllib.parse
from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = sys.argv[1]
B = os.path.join(ROOT, SESS, 'build')
sys.path.insert(0, os.path.join(ROOT, 'build')); sys.path.insert(0, B)
DECK = os.path.join(ROOT, SESS, importlib.import_module('slides').OUT)

JS = """
z => {
  const sl = document.querySelector('.slide.on'), w = sl.querySelector(':scope > .wrap');
  if (!w) return null;
  w.style.zoom = z;
  const r = sl.getBoundingClientRect(), rail = r.top + r.height - 30;
  let low = 0, over = w.scrollHeight - w.clientHeight > 2 || w.scrollWidth - w.clientWidth > 2;
  w.querySelectorAll('.cols,.colstack,pre,table,.card,.callout,.out,.fig,ol.n,ul.b').forEach(e => {
    if (e.scrollHeight - e.clientHeight > 3 || e.scrollWidth - e.clientWidth > 3) over = true;
  });
  w.querySelectorAll('.card,.callout,pre,table,.out,.fig,ol.n,ul.b,svg,.kv,.wctl').forEach(e => {
    if (!e.getClientRects().length) return;
    const b = e.getBoundingClientRect();
    if (b.bottom > low) low = b.bottom;
    if (b.right > r.right - 20) over = true;
  });
  return {over, fill: (low - r.top) / (rail - r.top)};
}
"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width': 1400, 'height': 820})
        await pg.goto('file://' + urllib.parse.quote(DECK)); await pg.wait_for_timeout(800)
        n = await pg.evaluate("document.querySelectorAll('.slide').length")
        fit = {}
        for i in range(n):
            await pg.evaluate("i => { const s=[...document.querySelectorAll('.slide')]; s.forEach((x,k)=>x.classList.toggle('on',k===i));"
                              "s[i].dispatchEvent(new CustomEvent('shown')); }", i)
            await pg.wait_for_timeout(120)
            base = await pg.evaluate(JS, 1)
            if base is None:
                continue
            best = 1.0
            if not base['over']:
                for k in range(50, 0, -2):
                    z = 1 + k / 100
                    r = await pg.evaluate(JS, z)
                    await pg.wait_for_timeout(20)
                    r = await pg.evaluate(JS, z)
                    if not r['over'] and r['fill'] <= 0.93:
                        best = z; break
            await pg.evaluate(JS, 1)
            if best > 1.0:
                fit[str(i)] = best
            print(f's{i + 1:02d} fill {base["fill"]:.0%} zoom {best:.2f}')
        await b.close()
    json.dump(fit, open(os.path.join(B, 'fit.json'), 'w'), indent=1)
    print(len(fit), 'slides zoomed')
asyncio.run(main())
