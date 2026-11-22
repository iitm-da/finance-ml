"""Assemble the AMC deck: original head (CSS) + extra CSS + _slides.html + original tail (JS) + extra JS."""
import pathlib
d = pathlib.Path(__file__).parent
head = (d / "_head.html").read_text(encoding="utf-8")
tail = (d / "_tail.html").read_text(encoding="utf-8")
slides = (d / "_slides.html").read_text(encoding="utf-8")

css = """
/* AMC additions */
.chev{height:92px;background:var(--ink);color:#fff;font-size:26px;font-weight:600;display:flex;align-items:center;justify-content:center;text-align:center;padding:0 30px 0 40px;margin-right:-14px;
  clip-path:polygon(0 0,calc(100% - 26px) 0,100% 50%,calc(100% - 26px) 100%,0 100%,26px 50%)}
.chev.first{clip-path:polygon(0 0,calc(100% - 26px) 0,100% 50%,calc(100% - 26px) 100%,0 100%);padding-left:24px}
.vc{padding:0 22px 0 24px;border-left:1px solid var(--line);min-height:400px}
.vc p{font-size:25px;line-height:1.3}
.vc p.k{font-size:21px;margin:18px 0 6px}
.vc p.k:first-child{margin-top:0}
.vc p.tags{margin-top:22px}
.fr{display:grid;grid-template-columns:230px 330px 1fr;align-items:center;height:90px;border-bottom:1px solid var(--line);font-size:27px}
.fr .f1{font-weight:600}
.fr .f2{font-family:'Courier New',monospace;font-weight:700;font-size:26px}
.fr i{display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:12px}
.fr i.ok{background:var(--blue)}
.fr i.no{background:var(--red);border-radius:2px}
.ar{display:grid;grid-template-columns:400px 560px 130px 1fr;align-items:center;column-gap:20px;padding:20px 0;border-top:1px solid var(--line)}
.ar:first-child{border-top:2px solid var(--ink)}
.ar .an b{display:block;font-size:27px;font-weight:600;line-height:1.2}
.ar .an span{display:block;font-size:22px;color:var(--muted);margin-top:4px}
.ar .at{position:relative;height:44px;background:var(--soft);border-radius:6px}
.ar .av{font-size:30px;font-weight:600;color:var(--blue)}
.ar .ax{font-size:24px;line-height:1.3;color:var(--ink2)}
.sr{margin-bottom:34px;display:grid;grid-template-columns:900px 1fr;column-gap:22px;align-items:center}
.sr p{grid-column:1/3;font-size:28px;font-weight:600;margin-bottom:10px!important}
.sr b{font-size:32px;font-weight:600;color:var(--muted)}
.seg{display:flex;align-items:center;padding:0 28px;font-size:28px;font-weight:500;white-space:nowrap}
.seg b{font-size:44px;font-weight:600;letter-spacing:-.03em;margin-right:14px}
.lr{display:grid;grid-template-columns:230px 640px 1fr;column-gap:18px;align-items:center;margin-bottom:14px}
.lr p{font-size:26px;font-weight:600}
.lr b{font-size:26px;font-weight:600}
.tdot{width:20px;height:20px;border-radius:50%;margin:14px 0 26px}
.th{font-size:29px;font-weight:600;line-height:1.2}
.tb{font-size:24px;line-height:1.32;color:var(--muted);margin-top:8px!important}
.tok{display:inline-flex;align-items:center;height:64px;padding:0 20px;border-radius:10px;background:var(--bluesoft);color:#1B33C9;font-size:30px;font-weight:600}
.tok.q{background:#fff;border:3px dashed var(--blue);min-width:150px;justify-content:center}
.tok.s{height:46px;padding:0 13px;font-size:23px;border-radius:8px}
.pr{display:grid;grid-template-columns:200px 640px 1fr;column-gap:18px;align-items:center;margin-bottom:12px}
.pr p{font-size:28px;font-weight:600;font-family:'Courier New',monospace}
.pr b{font-size:27px;font-weight:600}
p.li{font-size:27px;line-height:1.3;padding:14px 0;border-bottom:1px solid var(--line);margin:0}
p.li:first-of-type{margin-top:14px!important}

.role{display:grid;grid-template-columns:14px 1fr;column-gap:26px;padding:22px 0;border-top:1px solid var(--line)}
.role:first-child{border-top:2px solid var(--ink)}
.role span{border-radius:4px}
.role p:first-child{font-size:31px;font-weight:600}
.role p:last-child{font-size:26px;line-height:1.35;color:var(--muted);margin-top:8px!important}
.fl{position:absolute;left:120px;width:1140px}
.fl .lab{font-size:24px;font-weight:600;color:var(--muted);margin-bottom:12px!important}
.fl .row{display:flex;gap:30px}
.st{position:relative;flex:1;min-height:205px;border-radius:12px;padding:20px 20px;font-size:26px;line-height:1.28;background:var(--soft)}
.st:not(:last-child)::after{content:"";position:absolute;right:-22px;top:50%;margin-top:-9px;border:9px solid transparent;border-left:12px solid #B9BED6;border-right:0}
.st b{display:block;font-size:21px;font-weight:600;margin-bottom:10px}
.st.llm{background:var(--bluesoft)}.st.llm b{color:#1B33C9}
.st.vlm{background:var(--redsoft)}.st.vlm b{color:#B8231A}
.st.slm{background:var(--yellowsoft)}.st.slm b{color:#7A5B00}
.st.rule{background:#fff;border:2px solid var(--line)}.st.rule b{color:var(--muted)}
.st.hum{background:var(--ink);color:#fff}.st.hum b{color:var(--yellow)}
.dot{display:inline-block;width:22px;height:22px;border-radius:50%}
.dot.off{background:transparent;border:2px solid var(--line)}
#nav{right:14px;bottom:calc(60px + env(safe-area-inset-bottom,0px));flex-direction:column}
"""

js = """
<script>
(function(){
  // indexed revenue and cost curves (5.1% and 5.4% a year for 15 years)
  function curve(r){var d='';for(var y=0;y<=15;y++){var v=100*Math.pow(1+r,y);d+=(y?'L':'M')+(60+y*640/15).toFixed(1)+','+(440-(v-100)*400/150).toFixed(1)}return d}
  var rp=document.getElementById('revp'),cp=document.getElementById('costp');
  if(rp){rp.setAttribute('d',curve(.051));cp.setAttribute('d',curve(.054))}

  // one employee, 19,885 investors
  var c=document.getElementById('dots');if(!c)return;
  var x=c.getContext('2d'),N=19885,COLS=200,raf=0;
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function paint(n){x.clearRect(0,0,1000,500);x.fillStyle='#9AA3CC';
    for(var i=0;i<n;i++){x.fillRect((i%COLS)*5,Math.floor(i/COLS)*5,3,3)}
    x.fillStyle='#2B4BFF';if(x.roundRect){x.beginPath();x.roundRect(0,0,13,13,3);x.fill()}else x.fillRect(0,0,13,13)}
  function run(){cancelAnimationFrame(raf);if(reduce){paint(N);return}
    var t0=null;function step(t){if(!t0)t0=t;var p=Math.min(1,(t-t0-300)/1500);p=Math.max(0,p);paint(Math.round(N*(1-Math.pow(1-p,2))));if(p<1)raf=requestAnimationFrame(step)}
    raf=requestAnimationFrame(step)}
  var s=c.closest('.slide');
  new MutationObserver(function(){if(s.classList.contains('active'))run()}).observe(s,{attributes:true,attributeFilter:['class']});
  if(s.classList.contains('active'))run();
})();
</script>
"""

head = head.replace("</style>", css + "</style>")
head = head.replace("<title>Language is the ledger: a CXO briefing</title>", "<title>Language is the ledger: LLMs, VLMs and SLMs for asset management</title>")
tail = tail.replace("</body>", js + "</body>")
out = d.parent / "llm_vlm_slm_for_amc.html"
out.write_text(head + slides + tail, encoding="utf-8")
print("wrote", out, len(head + slides + tail))
