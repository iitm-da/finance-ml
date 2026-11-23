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
  .b4 .tb{font:600 24px/1 'Geist',Arial,sans-serif;padding:15px 26px;border-radius:999px;border:2px solid #C9CEE3;background:#fff;color:var(--muted);cursor:pointer;margin-right:12px;transition:background .3s,color .3s,border-color .3s}
  .b4 .tb.on{background:var(--ink);border-color:var(--ink);color:#fff}
  .b4 .tb:focus-visible{outline:3px solid var(--blue);outline-offset:3px}
  .b4 .sc{position:relative;height:440px;border-radius:14px;padding:18px 20px;background:var(--soft);border:2px solid transparent;opacity:.4;transition:opacity .4s,background .4s,border-color .4s,box-shadow .4s}
  .b4 .sc.done{opacity:1}
  .b4 .sc.on{opacity:1;background:#fff;border-color:var(--blue);box-shadow:0 14px 36px rgba(43,75,255,.18)}
  .b4 .sc:not(:last-child)::after{content:"";position:absolute;right:-33px;top:50%;margin-top:-11px;border:11px solid transparent;border-left:14px solid #C9CEE3;border-right:0;transition:border-left-color .4s}
  .b4 .sc.done:not(:last-child)::after{border-left-color:var(--blue)}
  .b4 .sh{font-size:29px;font-weight:600}
  .b4 .sh b{color:var(--blue);margin-right:12px}
  .b4 .cp{font-size:21px;line-height:1.3;color:var(--muted);margin-top:6px!important;min-height:55px}
  .b4 .vz{margin-top:10px}
  .b4 .it{display:flex;align-items:center;justify-content:space-between;gap:10px;background:var(--soft);border:1.5px solid var(--line);border-radius:9px;padding:8px 12px;margin-bottom:8px;font-size:19px;line-height:1.3;font-weight:500;opacity:0;transform:translateX(-16px);transition:opacity .35s,transform .35s}
  .b4 .sc:not(.on) .it{background:#fff}
  .b4 .it.col{display:block}
  .b4 .it.in{opacity:1;transform:none}
  .b4 .lb{flex:none;font-size:16px;font-weight:600;padding:4px 10px;border-radius:999px;white-space:nowrap;opacity:0;transform:scale(.5);transition:opacity .3s,transform .35s cubic-bezier(.2,1.6,.4,1)}
  .b4 .lb.in{opacity:1;transform:none}
  .b4 .lb.p{background:var(--bluesoft);color:#1B33C9}.b4 .lb.n{background:var(--redsoft);color:#B8231A}.b4 .lb.d{background:var(--yellow);color:var(--ink)}
  .b4 .hin{display:block;font-family:'Geist','Noto Sans Devanagari','Nirmala UI','Mangal',sans-serif;font-size:18px;line-height:1.4;font-weight:600;color:#1B33C9;margin-top:4px;opacity:0;transition:opacity .4s}
  .b4 .hin.in{opacity:1}
  .b4 mark{background:transparent;color:inherit;border-radius:5px;padding:1px 3px;margin:0 -3px;transition:background .45s}
  .b4 mark.hit{background:var(--yellow)}
  .b4 .cv{fill:none;stroke:var(--blue);stroke-width:4;stroke-linecap:round;stroke-dasharray:1000;stroke-dashoffset:1000}
  .b4 .cv.go{transition:stroke-dashoffset 1.7s cubic-bezier(.4,0,.2,1);stroke-dashoffset:0}
  .b4 .pg{height:12px;border-radius:6px;background:var(--line);overflow:hidden;margin-top:10px}
  .b4 .pg div{height:12px;width:0;background:var(--blue);border-radius:6px}
  .b4 .pg.go div{transition:width 1.7s linear;width:100%}
  .b4 .pt{font-size:19px;color:var(--muted);margin-top:8px!important}
  .b4 .bd{display:inline-block;background:var(--ink);color:#fff;font-size:22px;font-weight:600;padding:11px 20px;border-radius:10px;transition:box-shadow .3s,transform .35s cubic-bezier(.2,1.6,.4,1),opacity .3s}
  .b4 .bd.hid{opacity:0;transform:scale(.5)}
  .b4 .bd.glow{box-shadow:0 0 0 4px var(--blue),0 0 30px rgba(43,75,255,.6);transform:scale(1.05)}
  .b4 .dn{position:relative;height:30px;width:2px;margin:4px 0 4px 40px;background:#C9CEE3}
  .b4 .dn i{position:absolute;left:-6px;top:0;width:14px;height:14px;border-radius:50%;background:var(--blue);opacity:0}
  .b4 .dn i.go{animation:b4dn .45s linear forwards}
  @keyframes b4dn{0%{top:-4px;opacity:1}85%{opacity:1}100%{top:20px;opacity:0}}
  .b4 .tx.ty::after{content:"";display:inline-block;width:3px;height:20px;background:var(--blue);margin-left:3px;vertical-align:-3px;animation:b4bl .5s steps(2) infinite}
  @keyframes b4bl{50%{opacity:0}}
  .b4 .out{min-height:62px;border-radius:9px;background:var(--bluesoft);padding:10px 12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;opacity:0;transition:opacity .35s}
  .b4 .out.in{opacity:1}
  .b4 .trk{position:relative;width:130px;height:14px;border-radius:7px;background:#fff}
  .b4 .trk::after{content:"";position:absolute;left:50%;top:-6px;width:2px;height:26px;margin-left:-1px;background:var(--ink)}
  .b4 .fill{position:absolute;top:0;height:14px;width:0;border-radius:7px;transition:width .55s cubic-bezier(.2,.8,.2,1)}
  .b4 .res{font-size:23px;font-weight:600}
  .b4 .out .lb{font-size:19px;padding:5px 13px}
  .b4 .out .hin{font-size:20px;margin:0}
  .b4 .lp path{fill:none;stroke:var(--blue);stroke-width:3;stroke-dasharray:1000;stroke-dashoffset:1000}
  .b4 .lp polygon{fill:var(--blue);opacity:0}
  .b4 .lp .lt{opacity:0;transition:opacity .5s}
  .b4 .lp.go path{transition:stroke-dashoffset 1.3s cubic-bezier(.4,0,.2,1);stroke-dashoffset:0}
  .b4 .lp.go polygon{opacity:1;transition:opacity .2s 1.2s}
  .b4 .lp.go .lt{opacity:1}
  </style>
  <h2 class="title">Before large language models, every language task needed its own model and its own labelled data</h2>
  <div class="abs a tabs" style="left:120px;top:246px;--d:.15s" role="tablist"><button class="tb" role="tab">Sentiment model</button><button class="tb" role="tab">Entity model</button><button class="tb" role="tab">Translation model</button><button class="tb" role="tab">Routing model</button></div>
  <div class="abs a" style="left:120px;top:324px;width:1680px;display:grid;grid-template-columns:repeat(4,1fr);gap:44px;--d:.3s">
    <div class="sc"><p class="sh"><b>1</b>Data</p><p class="cp"></p><div class="vz"></div></div>
    <div class="sc"><p class="sh"><b>2</b>Labelling</p><p class="cp"></p><div class="vz"></div></div>
    <div class="sc"><p class="sh"><b>3</b>Modelling</p><p class="cp"></p><div class="vz"></div></div>
    <div class="sc"><p class="sh"><b>4</b>Deployment</p><p class="cp"></p><div class="vz"></div></div>
  </div>
  <svg class="abs lp" style="left:120px;top:766px" width="1680" height="64" viewBox="0 0 1680 64" aria-hidden="true">
    <path pathLength="1000" d="M1486,2 V30 Q1486,42 1474,42 H206 Q194,42 194,30 V12"/><polygon points="184,14 204,14 194,0"/>
    <g class="lt"><rect x="520" y="26" width="640" height="34" fill="#fff"/><text x="840" y="51" text-anchor="middle" font-size="23" font-weight="600" fill="#1B33C9" font-family="Geist,Arial,sans-serif">The loop: language changes, so collect, label and retrain</text></g>
  </svg>
  <div class="abs a" style="left:120px;top:846px;width:1680px;background:var(--soft);border-radius:14px;padding:20px 40px;--d:.6s"><p class="body" style="font-size:27px;line-height:1.35"><b>Every new task repeated the whole loop.</b> The sentiment model could not translate, and OpenAI put the labelling need at "thousands or tens of thousands of examples" for each task.</p></div>
  <div class="foot"><span>Samples and company names are invented. Quote: Brown et al., Language Models are Few-Shot Learners (OpenAI, 2020)</span><span class="pn"></span></div>
  <script>
  (function(){
    var sec=document.currentScript.closest('.slide'),tok=0,cur=0,auto=true;
    var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var T=[
     {name:'Sentiment model',kind:'chip',
      cp:['Collect market news headlines.','Analysts mark each headline positive or negative.','Train a classifier on the marked headlines.','Each new headline gets a score.'],
      items:[['Bank profits beat estimates','Positive','p'],['Rupee slides to a record low','Negative','n'],['Auto sales rise for a third month','Positive','p'],['Fund house reports outflows','Negative','n']],
      dep:{t:'IT firms raise full-year guidance',v:.8}},
     {name:'Entity model',kind:'mark',
      cp:['Collect sentences from company filings.','Annotators tag every company name by hand.','Train a tagger on the tagged sentences.','Names are pulled out of each new filing.'],
      items:[['<mark>Meridian Cements</mark> will merge with <mark>Kaveri Paints</mark>'],['Auditors of <mark>Northgate Finance</mark> have resigned'],['<mark>Solvix Pharma</mark> buys a plant from <mark>Arden Labs</mark>']],
      dep:{t:'Board of <mark>Tarini Motors</mark> approves a buyback'}},
     {name:'Translation model',kind:'pair',
      cp:['Collect English sentences from scheme documents.','Translators write the Hindi for each sentence.','Train on the sentence pairs.','New English text comes out in Hindi.'],
      items:[['Mutual fund investments are subject to market risks','म्यूचुअल फंड निवेश बाज़ार जोखिमों के अधीन हैं'],['Read all scheme related documents carefully','योजना संबंधी सभी दस्तावेज़ों को ध्यान से पढ़ें']],
      dep:{t:'Past performance may or may not be sustained in future',h:'पिछला प्रदर्शन भविष्य में कायम रह भी सकता है और नहीं भी'}},
     {name:'Routing model',kind:'chip',
      cp:['Collect past investor emails.','Service staff sort each email to a desk.','Train a classifier on the sorted emails.','Each new email goes to its desk.'],
      items:[['My SIP was debited twice this month','Payments','d'],['My KYC status shows on hold','KYC','d'],['Where is my redemption money?','Redemptions','d'],['Please update my address','KYC','d']],
      dep:{t:'Dividend not credited to my account',d:'Payments'}}
    ];
    var tabs=sec.querySelectorAll('.tb'),sc=sec.querySelectorAll('.sc'),lp=sec.querySelector('.lp');
    function plain(h){return h.replace(/<[^>]+>/g,'')}
    function sleep(ms){return new Promise(function(r){setTimeout(r,ms)})}
    function list(t,lab){return t.items.map(function(it){
      if(t.kind==='chip')return '<div class="it"><span>'+it[0]+'</span>'+(lab?'<span class="lb '+it[2]+'">'+it[1]+'</span>':'')+'</div>';
      if(t.kind==='pair')return '<div class="it col"><span>'+it[0]+'</span>'+(lab?'<span class="hin">'+it[1]+'</span>':'')+'</div>';
      return '<div class="it"><span>'+(lab?it[0]:plain(it[0]))+'</span></div>'}).join('')}
    function render(i){
      var t=T[i];cur=i;
      [].forEach.call(tabs,function(b,j){b.classList.toggle('on',j===i);b.setAttribute('aria-selected',j===i)});
      [].forEach.call(sc,function(c,j){c.className='sc';c.querySelector('.cp').textContent=t.cp[j]});
      lp.classList.remove('go');
      sc[0].querySelector('.vz').innerHTML=list(t,false);
      sc[1].querySelector('.vz').innerHTML=list(t,true);
      sc[2].querySelector('.vz').innerHTML='<svg width="340" height="124" viewBox="0 0 340 150" preserveAspectRatio="none" aria-hidden="true"><line x1="8" y1="140" x2="336" y2="140" stroke="#0E1330" stroke-width="2"/><line x1="8" y1="6" x2="8" y2="140" stroke="#0E1330" stroke-width="2"/><path class="cv" pathLength="1000" d="M12,16 C60,96 120,118 190,124 S300,130 334,131"/></svg><p class="pt">Error on examples held back</p><div class="pg"><div></div></div><p class="pt">Training run</p><p style="margin-top:14px!important"><span class="bd hid">'+t.name+'</span></p>';
      sc[3].querySelector('.vz').innerHTML='<div class="it in"><span class="tx"></span></div><div class="dn"><i></i></div><span class="bd">'+t.name+'</span><div class="dn"><i></i></div><div class="out"></div>';
    }
    function on(j){[].forEach.call(sc,function(c,k){c.classList.toggle('on',k===j);if(k<j)c.classList.add('done')})}
    function labels(now){var v=sc[1].querySelector('.vz');
      [].forEach.call(v.querySelectorAll('.it'),function(e){e.classList.add('in')});
      return [].slice.call(v.querySelectorAll('.it')).map(function(e){return function(){[].forEach.call(e.querySelectorAll('.lb,.hin'),function(x){x.classList.add('in')});[].forEach.call(e.querySelectorAll('mark'),function(x){x.classList.add('hit')})}})}
    function result(t){var o=sc[3].querySelector('.out'),d=t.dep;
      if(d.v!==undefined){var pos=d.v>0;o.innerHTML='<div class="trk"><div class="fill" style="background:'+(pos?'#2B4BFF':'#E8382D')+';'+(pos?'left:50%':'right:50%')+'"></div></div><b class="res" style="color:'+(pos?'#1B33C9':'#B8231A')+'">'+(pos?'Positive':'Negative')+'</b>';
        void o.offsetWidth;o.querySelector('.fill').style.width=(Math.abs(d.v)*50)+'%'}
      else if(d.h)o.innerHTML='<span class="hin in">'+d.h+'</span>';
      else if(d.d)o.innerHTML='<span class="lb d in">'+d.d+'</span>';
      else{var tx=sc[3].querySelector('.tx');tx.innerHTML=d.t;var m=tx.querySelectorAll('mark'),h='';
        for(var i=0;i<m.length;i++){m[i].classList.add('hit');h+='<span class="lb d in">'+m[i].textContent+'</span>'}o.innerHTML=h}
      o.classList.add('in')}
    function still(i){render(i);var t=T[i];
      [].forEach.call(sec.querySelectorAll('.sc'),function(c){c.classList.add('done')});
      [].forEach.call(sc[0].querySelectorAll('.it'),function(e){e.classList.add('in')});
      labels().forEach(function(f){f()});
      sc[2].querySelector('.cv').classList.add('go');sc[2].querySelector('.pg').classList.add('go');sc[2].querySelector('.bd').classList.remove('hid');
      sc[3].querySelector('.tx').textContent=plain(t.dep.t);result(t);lp.classList.add('go')}
    async function play(i,my){
      function w(ms){return sleep(ms).then(function(){if(tok!==my)throw 0})}
      try{
        while(true){
          var t=T[i];render(i);await w(350);
          on(0);var a=sc[0].querySelectorAll('.it');
          for(var k=0;k<a.length;k++){a[k].classList.add('in');await w(380)}
          await w(900);
          on(1);var f=labels();await w(500);
          for(k=0;k<f.length;k++){f[k]();await w(560)}
          await w(900);
          on(2);await w(300);sc[2].querySelector('.cv').classList.add('go');sc[2].querySelector('.pg').classList.add('go');
          await w(1900);sc[2].querySelector('.bd').classList.remove('hid');await w(1100);
          on(3);await w(300);var tx=sc[3].querySelector('.tx'),s=plain(t.dep.t),dn=sc[3].querySelectorAll('.dn i'),bd=sc[3].querySelector('.bd');
          tx.classList.add('ty');for(k=1;k<=s.length;k++){tx.textContent=s.slice(0,k);await w(24)}
          tx.classList.remove('ty');await w(250);
          dn[0].classList.add('go');await w(450);bd.classList.add('glow');await w(550);
          dn[1].classList.add('go');await w(450);bd.classList.remove('glow');result(t);await w(1700);
          on(4);lp.classList.add('go');await w(3000);
          if(auto)i=(i+1)%T.length;
        }
      }catch(e){}
    }
    function start(i){tok++;if(reduce){still(i);return}play(i,tok)}
    [].forEach.call(tabs,function(b,j){b.addEventListener('click',function(e){e.stopPropagation();auto=false;start(j)})});
    new MutationObserver(function(){if(sec.classList.contains('active')){auto=true;start(0)}else tok++}).observe(sec,{attributes:true,attributeFilter:['class']});
    render(0);
    if(sec.classList.contains('active'))start(0);
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
