"""Assemble a day 2 deck: DA2402 theme + <session>/build/slides.py + <session>/build/widgets.js -> one HTML file.

    python3 build/assemble_session.py nlp        (or: llm)

Full build: assemble, then fit_session.py (measures the slides), then assemble again. NOFIT=1 skips the zooms.

slides.py must define S (list of slide HTML), TITLE, RAIL and OUT (file name of the deck).
"""
import os, re, subprocess, sys, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = sys.argv[1]
B = os.path.join(ROOT, SESS, 'build') + '/'
sys.path.insert(0, os.path.join(ROOT, 'build')); sys.path.insert(0, B)
os.chdir(ROOT)
sl = importlib.import_module('slides')
S = sl.S
import json
FIT = B + 'fit.json'                    # written by fit_session.py: slide index -> zoom of the slide body
if os.path.exists(FIT) and not os.environ.get('NOFIT'):
    for k, z in json.load(open(FIT)).items():
        if int(k) < len(S):
            S[int(k)] = S[int(k)].replace('<div class="wrap">', f'<div class="wrap" style="zoom:{z}">', 1)

SRC = os.path.join(os.path.dirname(ROOT), 'DA2402/data storage/data_storage_1_file_formats.html')
OUT = os.path.join(ROOT, SESS, sl.OUT)
src = open(SRC, encoding='utf-8').read()
head = src[:src.index('<style>\n/* storage L1 additions')]
head = re.sub(r'<title>.*?</title>', f'<title>{sl.TITLE}</title>', head)
mech = src[src.index('/* ============================================================\n   deck mechanics'):
           src.index("(function(){\n  const el=document.getElementById('bw-data')")]
mech = mech.replace('DA2402 &#183; Data Curation and Visualization', sl.RAIL)
mech = mech.replace('Lecture 1 &#183; ', '')
assert 'DA2402' not in mech

css = open(os.path.join(ROOT, 'build', 'deck.css'), encoding='utf-8').read() + getattr(sl, 'CSS', '')
widgets = open(B + 'widgets.js', encoding='utf-8').read()
body = '<body>\n<div id="stage"><div id="deck">\n<div id="pbar"></div>\n' + ''.join(S) + '</div></div>\n'
nav = ('<div id="nav"><button id="prev" aria-label="Previous slide">&#8249;</button>'
       '<button id="next" aria-label="Next slide">&#8250;</button></div>\n'
       '<div id="hint">&#8592; &#8594; navigate &#183; F fullscreen &#183; P print</div>\n')
html = head + '<style>\n' + css + '\n</style>\n</head>\n' + body + nav + '<script>\n' + mech + widgets + '\n</script>\n</body>\n</html>\n'
open(OUT, 'w', encoding='utf-8').write(html)

ok = True
for i, m in enumerate(re.finditer(r'<script>(.*?)</script>', html, re.S)):
    p = f'{B}_js{i}.js'
    open(p, 'w', encoding='utf-8').write(m.group(1))
    r = subprocess.run(['node', '--check', p], capture_output=True, text=True)
    print('script', i, 'ok' if r.returncode == 0 else r.stderr)
    os.remove(p)
n_sec = len(re.findall(r'<section class="slide', html))
n_chk = len(re.findall(r'<section class="slide[^"]*" data-sec="([^"]+)">', html))
print(len(S), 'slides,', len(html), 'bytes | checker sees', n_chk, 'of', n_sec, '|', OUT)
