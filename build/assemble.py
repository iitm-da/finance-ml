"""Assemble the day 1 deck: DA2402 theme + slides.py + widgets.js -> one HTML file."""
import os, re, subprocess, sys
B = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.insert(0, B)
from slides import S

SRC = '/mnt/e/iitm course/DA2402/data storage/data_storage_1_file_formats.html'
OUT = os.path.join(os.path.dirname(B[:-1]), 'forecasting_and_generative_models.html')
src = open(SRC, encoding='utf-8').read()

head = src[:src.index('<style>\n/* storage L1 additions')]
head = re.sub(r'<title>.*?</title>',
              '<title>Time Series Forecasting and Generative Models for Finance</title>', head)
mech = src[src.index('/* ============================================================\n   deck mechanics'):
           src.index("(function(){\n  const el=document.getElementById('bw-data')")]
mech = mech.replace('DA2402 &#183; Data Curation and Visualization', 'Machine Learning for Finance &#183; Forecasting and Generative Models')
mech = mech.replace('Lecture 1 &#183; ', '')
assert 'DA2402' not in mech

css = '''<style>
/* deck additions */
.seg{flex-wrap:wrap}
.seg button{text-transform:none}
.wctl{display:flex;gap:12px;align-items:stretch;flex-wrap:wrap}
.wctl .out{padding:6px 10px}
.wctl label.rg{display:flex;flex-direction:column;justify-content:center;gap:3px;font-family:var(--mono);
  font-size:10px;letter-spacing:.1em;color:var(--soft);min-width:230px}
.wctl label.rg b{color:var(--ink);font-size:12px;letter-spacing:0}
input[type=range]{width:100%;accent-color:#B02F3B}
.wbtn{font-family:var(--mono);font-size:11px;padding:5px 16px;border:1px solid var(--ink);background:var(--ink);
  color:#fff;cursor:pointer;border-radius:2px;min-width:74px}
.kv{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kv .lf{font-family:var(--mono);font-size:24px;font-weight:700;padding:8px 12px;background:var(--card);
  border:1px solid var(--hair);border-left:4px solid var(--soft);line-height:1.2}
.kv .lf small{display:block;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--soft);font-weight:500}
.kv .lf.bad{color:#B02F3B;border-left-color:#B02F3B}.kv .lf.ok{color:#0E7A55;border-left-color:#0E7A55}
.kv .lf.b{color:#125F9E;border-left-color:#125F9E}
.kv.k3{grid-template-columns:repeat(3,1fr)}
ul.b.lg{font-size:15px;line-height:1.5}
ul.b.lg li{margin-bottom:11px}
table.d td.r,table.d th.r{text-align:right;font-family:var(--mono);white-space:nowrap}
table.d tr.best td{background:#F3FBF7;font-weight:700}
.cols.nat{flex:none}
.wctl .jsbadge{align-self:center}
.card.big{padding:14px 16px}
.card.big h3{font-size:16.5px;margin-bottom:8px}
.card.big p{font-size:13.6px;line-height:1.55}
table.d.lg{font-size:15px;line-height:1.45}
table.d.lg td{padding:15px 14px}
table.d.lg td:first-child{font-family:var(--body);white-space:normal;width:21%}
.refs{font-size:12.5px;line-height:1.5}
.refs p{margin-bottom:8px;padding-left:16px;text-indent:-16px}
</style>
'''
widgets = open(B + 'widgets.js', encoding='utf-8').read()
body = '<body>\n<div id="stage"><div id="deck">\n<div id="pbar"></div>\n' + ''.join(S) + '</div></div>\n'
nav = ('<div id="nav"><button id="prev" aria-label="Previous slide">&#8249;</button>'
       '<button id="next" aria-label="Next slide">&#8250;</button></div>\n'
       '<div id="hint">&#8592; &#8594; navigate &#183; F fullscreen &#183; P print</div>\n')
html = head + css + '</head>\n' + body + nav + '<script>\n' + mech + widgets + '\n</script>\n</body>\n</html>\n'
open(OUT, 'w', encoding='utf-8').write(html)

# every <script> block that is code must parse
ok = True
for i, m in enumerate(re.finditer(r'<script>(.*?)</script>', html, re.S)):
    p = f'{B}_js{i}.js'
    open(p, 'w', encoding='utf-8').write(m.group(1))
    r = subprocess.run(['node', '--check', p], capture_output=True, text=True)
    print('script', i, 'ok' if r.returncode == 0 else r.stderr)
    ok &= r.returncode == 0
n_sec = len(re.findall(r'<section class="slide', html))
n_chk = len(re.findall(r'<section class="slide[^"]*" data-sec="([^"]+)">', html))
print(len(S), 'slides,', len(html), 'bytes | checker sees', n_chk, 'of', n_sec)
