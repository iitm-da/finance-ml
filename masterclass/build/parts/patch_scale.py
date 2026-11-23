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
# slide A: four live lanes, one model per task (plain string: holds CSS and JS braces)
A = r'''<!-- SCALE-BEGIN A -->
<section class="slide b4" data-sec="Before LLMs">
  <style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@600&display=swap');
  .b4 .hd{display:grid;grid-template-columns:380px 44px 250px 44px 422px;margin-bottom:12px}
  .b4 .ln{display:grid;grid-template-columns:380px 44px 250px 44px 422px;height:128px;margin-bottom:14px}
  .b4 .in{border:2px solid var(--line);border-radius:12px;padding:0 20px;background:#fff;display:flex;align-items:center}
  .b4 .tx{font-size:25px;line-height:1.28;font-weight:500;min-height:32px}
  .b4 .tx.ty::after{content:"";display:inline-block;width:3px;height:26px;background:var(--blue);margin-left:3px;vertical-align:-4px;animation:b4bl .5s steps(2) infinite}
  @keyframes b4bl{50%{opacity:0}}
  .b4 mark{background:transparent;color:inherit;border-radius:5px;padding:1px 3px;margin:0 -3px;transition:background .45s}
  .b4 mark.hit{background:var(--yellow)}
  .b4 .cn{position:relative}
  .b4 .cn::before{content:"";position:absolute;left:5px;right:5px;top:50%;height:2px;margin-top:-1px;background:#C9CEE3}
  .b4 .cn i{position:absolute;top:50%;left:0;width:14px;height:14px;margin-top:-7px;border-radius:50%;background:var(--blue);opacity:0}
  .b4 .cn i.go{animation:b4go .5s linear forwards}
  @keyframes b4go{0%{left:0;opacity:1}85%{opacity:1}100%{left:calc(100% - 14px);opacity:0}}
  .b4 .md{background:var(--ink);color:#fff;border-radius:12px;padding:0 18px;display:flex;flex-direction:column;justify-content:center;transition:box-shadow .3s,transform .3s}
  .b4 .md b{font-size:25px;font-weight:600;white-space:nowrap}
  .b4 .md span{font-size:19px;line-height:1.25;color:rgba(255,255,255,.62);margin-top:5px}
  .b4 .md.on{box-shadow:0 0 0 4px var(--blue),0 0 36px rgba(43,75,255,.6);transform:scale(1.04)}
  .b4 .ou{background:var(--bluesoft);border-radius:12px;padding:0 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap;align-content:center}
  .b4 .trk{position:relative;width:170px;height:16px;border-radius:8px;background:#fff}
  .b4 .trk::after{content:"";position:absolute;left:50%;top:-6px;width:2px;height:28px;margin-left:-1px;background:var(--ink)}
  .b4 .fill{position:absolute;top:0;height:16px;width:0;border-radius:8px;transition:width .55s cubic-bezier(.2,.8,.2,1)}
  .b4 .res{font-size:28px;font-weight:600;opacity:0;transition:opacity .3s}
  .b4 .chip{font-size:22px;font-weight:600;padding:6px 14px;border-radius:999px;background:var(--yellow);color:var(--ink);opacity:0;transform:scale(.6);transition:opacity .3s,transform .35s cubic-bezier(.2,1.6,.4,1);white-space:nowrap}
  .b4 .chip.on{opacity:1;transform:none}
  .b4 .hi{font-size:25px;line-height:1.35;font-weight:600;color:#1B33C9;font-family:'Geist','Noto Sans Devanagari','Nirmala UI','Mangal',sans-serif}
  .b4 .dk{font-size:20px;font-weight:600;padding:5px 11px;border-radius:999px;border:2px solid #C9CEE3;color:var(--faint);background:#fff;transition:all .3s;white-space:nowrap}
  .b4 .ou.rt{gap:8px;flex-wrap:nowrap}
  .b4 .dk.on{background:var(--blue);border-color:var(--blue);color:#fff;transform:scale(1.1)}
  </style>
  <h2 class="title">Before large language models, every language task needed its own model and its own labelled data</h2>
  <div class="abs" style="left:120px;top:254px;width:1140px">
    <div class="hd a" style="--d:.15s"><p class="k">A sample comes in</p><span></span><p class="k">Its own model</p><span></span><p class="k">The one thing it does</p></div>
    <div class="ln a" data-k="sent" style="--d:.25s"><div class="in"><p class="tx"></p></div><div class="cn"><i></i></div><div class="md"><b>Sentiment model</b><span>trained on labelled headlines</span></div><div class="cn"><i></i></div><div class="ou"><div class="trk"><div class="fill"></div></div><b class="res"></b></div></div>
    <div class="ln a" data-k="ent" style="--d:.4s"><div class="in"><p class="tx"></p></div><div class="cn"><i></i></div><div class="md"><b>Entity model</b><span>trained on tagged filings</span></div><div class="cn"><i></i></div><div class="ou"></div></div>
    <div class="ln a" data-k="tr" style="--d:.55s"><div class="in"><p class="tx"></p></div><div class="cn"><i></i></div><div class="md"><b>Translation model</b><span>trained on sentence pairs</span></div><div class="cn"><i></i></div><div class="ou"><p class="hi"></p></div></div>
    <div class="ln a" data-k="rt" style="--d:.7s"><div class="in"><p class="tx"></p></div><div class="cn"><i></i></div><div class="md"><b>Routing model</b><span>trained on sorted emails</span></div><div class="cn"><i></i></div><div class="ou rt"><span class="dk">Payments</span><span class="dk">KYC</span><span class="dk">Redemptions</span></div></div>
  </div>
  <div class="abs" style="left:1340px;top:270px;width:460px">
    <div class="a" style="--d:.9s;border-top:2px solid var(--ink);padding-top:18px"><p style="font-size:30px;font-weight:600">Thousands of labelled examples for each task</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">OpenAI put the need at "thousands or tens of thousands of examples" per task.</p></div>
    <div class="a" style="--d:1.05s;border-top:2px solid var(--ink);padding-top:18px;margin-top:28px"><p style="font-size:30px;font-weight:600">A model did not transfer</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">The sentiment model could not translate. A new task was a new project.</p></div>
    <div class="a" style="--d:1.2s;border-top:2px solid var(--ink);padding-top:18px;margin-top:28px"><p style="font-size:30px;font-weight:600">Cost grew with the task list</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">Each model had its own data, its own team and its own upkeep.</p></div>
  </div>
  <div class="foot"><span>Samples and company names are invented. Quote: Brown et al., Language Models are Few-Shot Learners (OpenAI, 2020)</span><span class="pn"></span></div>
  <script>
  (function(){
    var sec=document.currentScript.closest('.slide'),tok=0;
    var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var S={
      sent:[{t:'Bank profits beat estimates',v:.8},{t:'Rupee slides to a record low',v:-.7},{t:'Auto sales rise for a third month',v:.55}],
      ent:[{t:'<mark>Meridian Cements</mark> will merge with <mark>Kaveri Paints</mark>'},{t:'Auditors of <mark>Northgate Finance</mark> have resigned'},{t:'<mark>Solvix Pharma</mark> buys a plant from <mark>Arden Labs</mark>'}],
      tr:[{t:'Mutual fund investments are subject to market risks',h:'म्यूचुअल फंड निवेश बाज़ार जोखिमों के अधीन हैं'},{t:'Read all scheme related documents carefully',h:'योजना संबंधी सभी दस्तावेज़ों को ध्यान से पढ़ें'}],
      rt:[{t:'My SIP was debited twice this month',d:0},{t:'My KYC status shows on hold',d:1},{t:'Where is my redemption money?',d:2}]
    };
    function sleep(ms){return new Promise(function(r){setTimeout(r,ms)})}
    function plain(h){return h.replace(/<[^>]+>/g,'')}
    function reset(ln,k){
      var ou=ln.querySelector('.ou');ln.querySelector('.tx').textContent='';ln.querySelector('.md').classList.remove('on');
      if(k==='sent'){ou.querySelector('.fill').style.width='0';ou.querySelector('.res').style.opacity=0}
      if(k==='ent')ou.innerHTML='';
      if(k==='tr')ou.querySelector('.hi').textContent='';
      if(k==='rt')[].forEach.call(ou.querySelectorAll('.dk'),function(d){d.classList.remove('on')});
    }
    async function show(ln,k,s,fast){
      var ou=ln.querySelector('.ou'),tx=ln.querySelector('.tx');
      if(k==='sent'){var f=ou.querySelector('.fill'),r=ou.querySelector('.res'),pos=s.v>0,c=pos?'#2B4BFF':'#E8382D';
        f.style.background=c;f.style.left=pos?'50%':'auto';f.style.right=pos?'auto':'50%';f.style.width=(Math.abs(s.v)*50)+'%';
        r.textContent=pos?'Positive':'Negative';r.style.color=pos?'#1B33C9':'#B8231A';r.style.opacity=1}
      if(k==='ent'){tx.innerHTML=s.t;var ms=tx.querySelectorAll('mark');
        for(var i=0;i<ms.length;i++){var c2=document.createElement('span');c2.className='chip';c2.textContent=ms[i].textContent;ou.appendChild(c2)}
        void ou.offsetWidth;
        for(i=0;i<ms.length;i++){ms[i].classList.add('hit');ou.children[i].classList.add('on');if(!fast)await sleep(320)}}
      if(k==='tr'){var w=s.h.split(' '),p=ou.querySelector('.hi');
        for(var j=1;j<=w.length;j++){p.textContent=w.slice(0,j).join(' ');if(!fast)await sleep(130)}}
      if(k==='rt')ou.querySelectorAll('.dk')[s.d].classList.add('on');
    }
    function dot(el){var i=el.querySelector('i');i.classList.remove('go');void i.offsetWidth;i.classList.add('go')}
    async function lane(ln,my,delay){
      var k=ln.dataset.k,n=0,tx=ln.querySelector('.tx'),cn=ln.querySelectorAll('.cn'),md=ln.querySelector('.md');
      await sleep(delay);
      while(tok===my){
        var s=S[k][n%S[k].length],t=plain(s.t);reset(ln,k);
        tx.classList.add('ty');
        for(var c=1;c<=t.length;c++){if(tok!==my)return;tx.textContent=t.slice(0,c);await sleep(22)}
        tx.classList.remove('ty');await sleep(250);if(tok!==my)return;
        dot(cn[0]);await sleep(480);md.classList.add('on');await sleep(520);if(tok!==my)return;
        dot(cn[1]);await sleep(480);md.classList.remove('on');
        await show(ln,k,s,false);await sleep(2600);n++;
      }
    }
    function start(){tok++;var my=tok,ls=sec.querySelectorAll('.ln');
      [].forEach.call(ls,function(ln,i){var k=ln.dataset.k;
        if(reduce){reset(ln,k);ln.querySelector('.tx').textContent=plain(S[k][0].t);show(ln,k,S[k][0],true)}
        else{reset(ln,k);lane(ln,my,900+i*650)}})}
    new MutationObserver(function(){if(sec.classList.contains('active'))start();else tok++}).observe(sec,{attributes:true,attributeFilter:['class']});
    if(sec.classList.contains('active'))start();
  })();
  </script>
</section>
<!-- SCALE-END A -->

'''

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
