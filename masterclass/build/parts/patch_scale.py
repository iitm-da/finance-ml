"""Insert the five 'how we got to LLMs' slides into _slides.html (safe to re-run), then run build.py.

    python3 parts/patch_scale.py

Chart numbers:
- Scaling law: Kaplan et al. 2020 (arXiv 2001.08361), eq. 1.1, L(N) = (8.8e13 / N) ** 0.076.
- Three-digit addition: BIG-bench results files, task modified_arithmetic, subtask
  three_digit_addition_control, zero-shot, exact_str_match
  (github.com/google/BIG-bench, bigbench/benchmark_tasks/modified_arithmetic/results).
"""
import math, pathlib, re

d = pathlib.Path(__file__).resolve().parent.parent
p = d / "_slides.html"
s = p.read_text(encoding="utf-8")
s = re.sub(r"<!-- SCALE-BEGIN (\w) -->.*?<!-- SCALE-END \1 -->\n\n", "", s, flags=re.S)

# ------------------------------------------------------------------ data
ALPHA, NC = 0.076, 8.8e13
loss = lambda n: (NC / n) ** ALPHA
DROP = 1 - 10 ** -ALPHA                       # fall in loss per tenfold increase in parameters

# (parameters, accuracy in %)
GPT3 = [(125e6, 0.6), (350e6, 0.0), (760e6, 0.0), (1.3e9, 0.0), (3e9, 0.0), (6e9, 0.6), (13e9, 16.9), (175e9, 75.9)]
BIGG = [(166996992, 0.0), (303121280, 0.0), (502161408, 0.0), (1139320832, 0.0), (2179138560, 0.0),
        (3722277888, 0.0), (8721174528, 0.0), (29188012032, 0.0), (137702416384, 51.8)]

# GPT-1, GPT-2, GPT-3 parameter counts; circle area in proportion
G = [117e6, 1.5e9, 175e9]
R3 = 150
rad = [R3 * math.sqrt(g / G[2]) for g in G]
RATIO = G[2] / G[0]

# ------------------------------------------------------------------ chart: scaling law
def law_chart():
    X0, X1, Y0, Y1 = 110, 940, 470, 30
    fx = lambda n: X0 + (math.log10(n) - 5) * (X1 - X0) / 4
    fy = lambda l: Y0 - (math.log(l) - math.log(2)) / (math.log(6) - math.log(2)) * (Y0 - Y1)
    o = '<svg class="abs" style="left:120px;top:300px" width="1000" height="570" viewBox="0 0 1000 570" role="img" aria-label="Line chart on two logarithmic axes: test loss falls in a straight line from 4.78 at 100 thousand parameters to 2.38 at 1 billion">'
    for l in (2, 3, 4, 5, 6):
        o += f'<line x1="{X0}" x2="{X1}" y1="{fy(l):.1f}" y2="{fy(l):.1f}" stroke="{"#0E1330" if l == 2 else "#E2E5F0"}" stroke-width="{2 if l == 2 else 1}"/>'
        o += f'<text x="{X0 - 14}" y="{fy(l) + 7:.1f}" text-anchor="end" font-size="22" fill="#8A90B0">{l}</text>'
    for e, lab in zip(range(5, 10), ("100 thousand", "1 million", "10 million", "100 million", "1 billion")):
        o += f'<text x="{fx(10 ** e):.1f}" y="506" text-anchor="middle" font-size="22" fill="#8A90B0">{lab}</text>'
    o += f'<text x="{(X0 + X1) / 2}" y="552" text-anchor="middle" font-size="22" fill="#5D6488">Parameters in the model (log scale)</text>'
    o += f'<path class="draw" style="--len:1000;--d:.4s" pathLength="1000" fill="none" stroke="#2B4BFF" stroke-width="5" stroke-linecap="round" d="M{fx(1e5):.1f},{fy(loss(1e5)):.1f} L{fx(1e9):.1f},{fy(loss(1e9)):.1f}"/>'
    for i, e in enumerate(range(5, 10)):
        n = 10 ** e
        o += f'<g class="a" style="--d:{.7 + i * .3:.1f}s"><circle cx="{fx(n):.1f}" cy="{fy(loss(n)):.1f}" r="9" fill="#2B4BFF" stroke="#fff" stroke-width="3"/>'
        o += f'<text x="{fx(n) + 6:.1f}" y="{fy(loss(n)) - 20:.1f}" font-size="26" font-weight="600" fill="#0E1330">{loss(n):.2f}</text></g>'
    return o + "</svg>"

# ------------------------------------------------------------------ chart: three-digit addition
def add_chart():
    X0, X1, Y0, Y1 = 90, 940, 470, 30
    fx = lambda n: X0 + (math.log10(n) - 8) * (X1 - X0) / 4
    fy = lambda a: Y0 - a / 100 * (Y0 - Y1)
    path = lambda pts: "M" + " L".join(f"{fx(n):.1f},{fy(a):.1f}" for n, a in pts)
    o = '<svg class="abs" style="left:120px;top:300px" width="1000" height="570" viewBox="0 0 1000 570" role="img" aria-label="Line chart: accuracy on three-digit addition stays near zero for GPT-3 up to 6 billion parameters, then reaches 17% at 13 billion and 76% at 175 billion. Google BIG-G stays at zero up to 27 billion and reaches 52% at 128 billion">'
    for a in (0, 25, 50, 75, 100):
        o += f'<line x1="{X0}" x2="{X1}" y1="{fy(a):.1f}" y2="{fy(a):.1f}" stroke="{"#0E1330" if a == 0 else "#E2E5F0"}" stroke-width="{2 if a == 0 else 1}"/>'
        o += f'<text x="{X0 - 14}" y="{fy(a) + 7:.1f}" text-anchor="end" font-size="22" fill="#8A90B0">{a}%</text>'
    for e, lab in zip(range(8, 13), ("100 million", "1 billion", "10 billion", "100 billion", "1 trillion")):
        o += f'<text x="{fx(10 ** e):.1f}" y="506" text-anchor="middle" font-size="22" fill="#8A90B0">{lab}</text>'
    o += f'<text x="{(X0 + X1) / 2}" y="552" text-anchor="middle" font-size="22" fill="#5D6488">Parameters in the model (log scale)</text>'
    o += f'<path class="draw" style="--len:1000;--d:1.5s" pathLength="1000" fill="none" stroke="#E8382D" stroke-width="4" stroke-linejoin="round" d="{path(BIGG)}"/>'
    o += f'<path class="draw" style="--len:1000;--d:.4s" pathLength="1000" fill="none" stroke="#2B4BFF" stroke-width="5" stroke-linejoin="round" d="{path(GPT3)}"/>'
    o += '<g class="a" style="--d:2.6s">' + "".join(f'<rect x="{fx(n) - 6:.1f}" y="{fy(a) - 6:.1f}" width="12" height="12" fill="#E8382D" stroke="#fff" stroke-width="2"/>' for n, a in BIGG)
    n, a = BIGG[-1]
    o += f'<text x="{fx(n) + 18:.1f}" y="{fy(a) + 36:.1f}" font-size="24" font-weight="600" fill="#B8231A">Google BIG-G {a:.0f}%</text></g>'
    o += '<g class="a" style="--d:1.5s">' + "".join(f'<circle cx="{fx(n):.1f}" cy="{fy(a):.1f}" r="8" fill="#2B4BFF" stroke="#fff" stroke-width="2"/>' for n, a in GPT3)
    n, a = GPT3[-1]
    o += f'<text x="{fx(n) + 16:.1f}" y="{fy(a) + 8:.1f}" font-size="24" font-weight="600" fill="#1B33C9">GPT-3 {a:.0f}%</text>'
    n, a = GPT3[-2]
    o += f'<text x="{fx(n) - 16:.1f}" y="{fy(a) - 4:.1f}" text-anchor="end" font-size="24" font-weight="600" fill="#1B33C9">{a:.0f}% at 13 billion</text></g>'
    return o + "</svg>"

# ------------------------------------------------------------------ slides
ROWS = [("News headlines marked positive or negative", "Sentiment model", "Scores the tone of market news"),
        ("Filings with every company name tagged", "Entity model", "Finds company names in filings"),
        ("Sentence pairs in English and Hindi", "Translation model", "Translates a factsheet"),
        ("Past investor emails sorted by hand", "Routing model", "Sends an email to the right desk")]
rows = "".join(f"""
    <div class="row a" style="--d:{.25 + i * .15:.2f}s;margin-bottom:16px"><div class="st r rule">{a}</div><div class="st r hum">{b}</div><div class="st r llm">{c}</div></div>""" for i, (a, b, c) in enumerate(ROWS))

A = f"""<!-- SCALE-BEGIN A -->
<section class="slide" data-sec="Before LLMs">
  <h2 class="title">Before large language models, every language task needed its own model and its own labelled data</h2>
  <div class="fl" style="top:270px">
    <div class="row a" style="--d:.15s;margin-bottom:14px"><p class="k" style="flex:1">Labelled data</p><p class="k" style="flex:1">Model</p><p class="k" style="flex:1">The one task it does</p></div>{rows}
  </div>
  <div class="abs" style="left:1340px;top:270px;width:460px">
    <div class="a" style="--d:.9s;border-top:2px solid var(--ink);padding-top:18px"><p style="font-size:30px;font-weight:600">Thousands of labelled examples for each task</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">OpenAI put the need at "thousands or tens of thousands of examples" per task.</p></div>
    <div class="a" style="--d:1.05s;border-top:2px solid var(--ink);padding-top:18px;margin-top:28px"><p style="font-size:30px;font-weight:600">A model did not transfer</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">The sentiment model could not translate. A new task was a new project.</p></div>
    <div class="a" style="--d:1.2s;border-top:2px solid var(--ink);padding-top:18px;margin-top:28px"><p style="font-size:30px;font-weight:600">Cost grew with the task list</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">Each model had its own data, its own team and its own upkeep.</p></div>
  </div>
  <div class="foot"><span>The four tasks are illustrative. Quote: Brown et al., Language Models are Few-Shot Learners (OpenAI, 2020)</span><span class="pn"></span></div>
</section>
<!-- SCALE-END A -->

"""

def gpt(i, name, year, params, head, body, dly):
    r = rad[i]
    return f"""
    <div class="a" style="--d:{dly}s">
      <div style="display:flex;align-items:center;gap:26px;height:300px"><svg width="{2 * r:.1f}" height="{2 * r:.1f}" viewBox="0 0 {2 * r:.1f} {2 * r:.1f}" aria-hidden="true"><circle cx="{r:.2f}" cy="{r:.2f}" r="{r:.2f}" fill="#2B4BFF"/></svg>
        <div><p class="big" style="font-size:52px">{name}</p><p style="font-size:26px;color:var(--muted);margin-top:8px">{year} &#183; {params} parameters</p></div></div>
      <div style="border-top:2px solid var(--ink);padding-top:16px;margin-top:14px"><p style="font-size:29px;font-weight:600;line-height:1.2">{head}</p><p class="body" style="font-size:25px;line-height:1.34;color:var(--muted);margin-top:8px">{body}</p></div>
    </div>"""

B = f"""<!-- SCALE-BEGIN B -->
<section class="slide" data-sec="Scale">
  <h2 class="title">OpenAI kept one design and made it about {round(RATIO, -2):,.0f} times larger, and one model began to handle many tasks</h2>
  <div class="abs" style="left:120px;top:236px;width:1680px;display:grid;grid-template-columns:repeat(3,1fr);gap:56px">{gpt(0, "GPT-1", "2018", "117 million", "Trained again for each task", "It first learned to predict the next token from books. Each task then needed a second round of training on labelled examples.", ".2")}{gpt(1, "GPT-2", "2019", "1.5 billion", "Tested on tasks it was never trained for", 'Reading comprehension, translation and summarising, with no task training. The paper was titled "Language Models are Unsupervised Multitask Learners".', ".45")}{gpt(2, "GPT-3", "2020", "175 billion", "Given the task in the prompt", "A few examples in the prompt were enough for translation, question answering and three-digit arithmetic. The model itself was not changed.", ".7")}
  </div>
  <div class="abs a" style="left:120px;top:852px;width:1680px;background:var(--soft);border-radius:14px;padding:26px 40px;--d:1.1s"><p class="body" style="font-size:28px"><b>The training goal was the same in all three: predict the next token.</b> What changed was the size, and the task moved out of the training data and into the prompt.</p></div>
  <div class="foot"><span>Circle area in proportion to parameter count. Sources: Radford et al. (OpenAI, 2018 and 2019); Brown et al. (OpenAI, 2020)</span><span class="pn"></span></div>
</section>
<!-- SCALE-END B -->

"""

C = f"""<!-- SCALE-BEGIN C -->
<section class="slide" data-sec="Scale">
  <h2 class="title">Scaling laws: each tenfold increase in parameters cut the test loss by about {DROP * 100:.0f}%</h2>
  <p class="abs a k" style="left:120px;top:250px;--d:.2s">Test loss, the error in predicting the next token (log scale, lower is better)</p>
  {law_chart()}
  <div class="abs" style="left:1240px;top:250px;width:560px">
    <div class="a" style="--d:.5s;border-top:2px solid var(--ink);padding-top:18px"><p class="big" style="font-size:60px">7+</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:6px">orders of magnitude over which some of the trends held</p></div>
    <div class="a" style="--d:.7s;border-top:2px solid var(--ink);padding-top:18px;margin-top:26px"><p style="font-size:30px;font-weight:600">Parameters, data and compute</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">The loss followed the same kind of law in each of the three.</p></div>
    <div class="a" style="--d:.9s;border-top:2px solid var(--ink);padding-top:18px;margin-top:26px"><p style="font-size:30px;font-weight:600">A larger model became a forecast</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">A lab could estimate the gain from a bigger training run before paying for it.</p></div>
    <div class="a" style="--d:1.1s;border-top:2px solid var(--ink);padding-top:18px;margin-top:26px"><p style="font-size:30px;font-weight:600">Data has to grow with size</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">In 2022 DeepMind's 70B Chinchilla, trained on 4x more data, beat the 175B GPT-3.</p></div>
  </div>
  <div class="foot"><span>Line is the fitted law L = (8.8&#215;10&#185;&#179; / N)^0.076 from Kaplan et al. (OpenAI, 2020). {DROP * 100:.0f}% is our calculation. Hoffmann et al. (DeepMind, 2022)</span><span class="pn"></span></div>
</section>
<!-- SCALE-END C -->

"""

D = f"""<!-- SCALE-BEGIN D -->
<section class="slide" data-sec="Emergence">
  <h2 class="title">Emergent abilities: GPT-3 scored {GPT3[4][1]:.0f}% on three-digit addition at 3 billion parameters and {GPT3[-1][1]:.0f}% at full size</h2>
  <p class="abs a k" style="left:120px;top:250px;--d:.2s">Three-digit additions answered exactly right</p>
  {add_chart()}
  <div class="abs" style="left:1240px;top:250px;width:560px">
    <div class="a" style="--d:.5s;background:var(--bluesoft);border-radius:14px;padding:24px 28px"><p style="font-size:28px;line-height:1.32;font-weight:600">"An ability is emergent if it is not present in smaller models but is present in larger models."</p><p style="font-size:23px;color:var(--muted);margin-top:10px">Wei et al., Google, 2022</p></div>
    <div class="a" style="--d:1.6s;border-top:2px solid var(--ink);padding-top:18px;margin-top:28px"><p style="font-size:30px;font-weight:600">Addition was not a training goal</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">Every size was trained only to predict the next token.</p></div>
    <div class="a" style="--d:2.7s;border-top:2px solid var(--ink);padding-top:18px;margin-top:26px"><p style="font-size:30px;font-weight:600">The size at which it appears differs</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">Google's models scored {BIGG[-2][1]:.0f}% at 27 billion parameters and {BIGG[-1][1]:.0f}% at 128 billion.</p></div>
    <div class="a" style="--d:2.9s;border-top:2px solid var(--ink);padding-top:18px;margin-top:26px"><p style="font-size:30px;font-weight:600">The loss curve gave no sign of it</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">Loss falls smoothly with size while single tasks move in steps.</p></div>
  </div>
  <div class="foot"><span>Data: BIG-bench results, three-digit addition, zero-shot, exact match; largest GPT-3 plotted at 175B. Definition: Wei et al. (2022)</span><span class="pn"></span></div>
</section>
<!-- SCALE-END D -->

"""

# whole-answer accuracy when each of four digits is right with probability q
def pair(q):
    w = q ** 4
    return f"""<div class="lr" style="grid-template-columns:140px 190px 1fr;column-gap:14px;margin-bottom:10px"><p style="font-size:24px;white-space:nowrap">{q * 100:.0f}% a digit</p><div class="track" style="position:relative;width:190px;height:34px"><div class="bar" style="--w:{190 * w:.0f}px;height:34px;background:var(--blue);--d:.9s"></div></div><b style="font-size:24px;white-space:nowrap">{w * 100:.0f}% of sums</b></div>"""

E = f"""<!-- SCALE-BEGIN E -->
<section class="slide" data-sec="Emergence">
  <h2 class="title">Emergence is a limit on prediction, so each new model version needs a test on your own work</h2>
  <div class="abs" style="left:120px;top:262px;width:1680px;display:grid;grid-template-columns:repeat(3,1fr);gap:56px">
    <div class="a" style="--d:.2s;border-top:4px solid var(--ink);padding-top:22px">
      <p style="font-size:33px;font-weight:600;line-height:1.2">The jump depends on how you score</p>
      <p class="body" style="font-size:26px;margin-top:12px">Stanford researchers scored each digit separately in 2023 and found a smooth rise. The step appears when all four digits of the answer must be right together.</p>
      <div style="margin-top:22px">{pair(.5)}{pair(.7)}{pair(.9)}{pair(.97)}</div>
    </div>
    <div class="a" style="--d:.4s;border-top:4px solid var(--ink);padding-top:22px">
      <p style="font-size:33px;font-weight:600;line-height:1.2">A business scores all-or-nothing</p>
      <p class="body" style="font-size:26px;margin-top:12px">A NAV or a PAN with one wrong digit is a wrong NAV or PAN. The strict measure is the one that matters in operations.</p>
      <p class="body" style="font-size:26px;color:var(--muted);margin-top:22px">"A language model that can almost perform a task is not much more useful than one that can't perform it at all."</p>
    </div>
    <div class="a" style="--d:.6s;border-top:4px solid var(--ink);padding-top:22px">
      <p style="font-size:33px;font-weight:600;line-height:1.2">A test confirms an ability; it cannot rule one out</p>
      <p class="body" style="font-size:26px;margin-top:12px">Asking a model to "think step by step" raised its maths scores with no retraining. Abilities can surface after release.</p>
      <p class="body" style="font-size:26px;margin-top:22px"><b>For the AMC:</b> treat a vendor upgrade as a new model and rerun your own test set before it reaches a process.</p>
    </div>
  </div>
  <div class="abs a" style="left:120px;top:846px;width:1680px;background:var(--soft);border-radius:14px;padding:24px 40px;--d:1.1s"><p class="body" style="font-size:28px;font-weight:600">"These developments are neither magical nor inexplicable: we just have limited ability to predict them."</p><p style="font-size:24px;color:var(--muted);margin-top:6px">Thomas Woodside, CSET, Georgetown University, April 2024</p></div>
  <div class="foot"><span>Sources: Woodside, CSET explainer (April 2024); Schaeffer, Miranda and Koyejo (Stanford, 2023). Bars are our calculation.</span><span class="pn"></span></div>
</section>
<!-- SCALE-END E -->

"""

m1, m2 = "<!-- HOW 1: LLM -->", "<!-- HOW 2: VLM -->"
assert s.count(m1) == 1 and s.count(m2) == 1
s = s.replace(m1, A + m1).replace(m2, B + C + D + E + m2)
p.write_text(s, encoding="utf-8")
print(f"ratio {RATIO:.0f}x, radii {[round(r, 1) for r in rad]}, drop per decade {DROP * 100:.1f}%, loss {[round(loss(10 ** e), 2) for e in range(5, 10)]}")
print("whole-answer", [round(q ** 4 * 100, 1) for q in (.5, .7, .9, .97)])

import runpy
runpy.run_path(str(d / "build.py"))
