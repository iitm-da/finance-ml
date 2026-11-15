import html, re, json, math

TS = json.load(open('/mnt/e/iitm course/lloyds/build/m_ts.json'))
GN = json.load(open('/mnt/e/iitm course/lloyds/build/m_gan.json'))

INK = '#14222B'; SOFT = '#53666F'; LINE = '#C3CFD5'; HAIR = '#DCE4E8'
BLUE = '#125F9E'; GREEN = '#0E7A55'; RED = '#B02F3B'; AMBER = '#8F5D0A'; PURP = '#9B2C7E'; SLATE = '#4B5A87'
MONO = 'JetBrains Mono,monospace'; BODY = 'Public Sans,system-ui,sans-serif'

esc = html.escape

def mb(b, d=1):
    return f"{b/1e6:.{d}f} MB"

def kb(b, d=1):
    return f"{b/1e3:.{d}f} KB"

def comma(n):
    return f"{n:,}"

KW = r'\b(import|from|as|for|in|if|else|def|return|True|False|None|with|print)\b'

def hl(line):
    """tiny python highlighter: comments, strings, keywords, numbers"""
    m = re.match(r'^(.*?)(\s*#.*)?$', line)
    body, cm = m.group(1), m.group(2) or ''
    parts = re.split(r"('[^']*'|\"[^\"]*\")", body)
    out = []
    for p in parts:
        if p.startswith(("'", '"')) and len(p) > 1:
            out.append(f'<span class="py-str">{esc(p)}</span>')
        else:
            e = esc(p)
            e = re.sub(KW, r'<span class="py-kw">\1</span>', e)
            e = re.sub(r'(?<![\w.#])(\d[\d_]*\.?\d*)', r'<span class="py-num">\1</span>', e)
            out.append(e)
    s = ''.join(out)
    if cm:
        s += f'<span class="py-cm">{esc(cm)}</span>'
    return s

def code(src, cls=''):
    lines = src.strip('\n').split('\n')
    return f'<pre class="code {cls}">' + '\n'.join(hl(l) for l in lines) + '</pre>'

def out(txt, cls='', lbl='output'):
    o = f'<div class="outlbl">{esc(lbl)}</div>' if lbl else ''
    return o + f'<div class="out {cls}">{esc(txt.strip(chr(10)))}</div>'

def slide(sec, eyebrow, title, small, body, cls='', sm_title=False):
    c = 'slide' + (' ' + cls if cls else '')
    tcls = 't sm' if sm_title else 't'
    sm = f'\n    <small>{small}</small>' if small else ''
    return f'''
<section class="{c}" data-sec="{sec}">
  <div class="eyebrow">{eyebrow}</div>
  <h1 class="{tcls}">{title}{sm}</h1>
  <div class="wrap">
{body}
  </div>
</section>
'''

def fig(inner, cap='', rd=''):
    c = f'<div class="cap">{cap}</div>' if cap else ''
    r = f'<div class="rd">{rd}</div>' if rd else ''
    return f'<div class="fig">{c}{inner}{r}</div>'

def card(h, p, k=''):
    return f'<div class="card {k}"><h3>{h}</h3><p>{p}</p></div>'

def table(head, rows, cls='d sm free'):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table class="{cls}"><tr>{h}</tr>{b}</table>'

# ---------------- svg helpers ----------------

def T(x, y, s, size=12, fill=INK, anchor='start', font=BODY, weight=400, extra=''):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(str(s))}</text>')

def Rect(x, y, w, h, fill='#FFFFFF', stroke='none', sw=1, extra=''):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'

def L(x1, y1, x2, y2, stroke=SOFT, sw=1.4, extra=''):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'

def arrow(x1, y1, x2, y2, stroke=SOFT, sw=1.6):
    # straight arrow with head at (x2,y2)
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - 8 * math.cos(ang) + 4 * math.sin(ang); hy1 = y2 - 8 * math.sin(ang) - 4 * math.cos(ang)
    hx2 = x2 - 8 * math.cos(ang) - 4 * math.sin(ang); hy2 = y2 - 8 * math.sin(ang) + 4 * math.cos(ang)
    return L(x1, y1, x2 - 6 * math.cos(ang), y2 - 6 * math.sin(ang), stroke, sw) + \
        f'<path d="M{x2:.1f},{y2:.1f} L{hx1:.1f},{hy1:.1f} L{hx2:.1f},{hy2:.1f} z" fill="{stroke}"/>'

def svg(w, h, inner, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMin meet" role="img" '
            f'aria-label="{esc(label)}">{inner}</svg>')

def hbars(items, w=620, label_w=150, bar_h=30, gap=12, unit='', maxv=None, label='bar chart', valfmt=None, top=6):
    """items: (label, value, colour, note)"""
    maxv = maxv or max(v for _, v, _, _ in items)
    span = w - label_w - 150
    h = top + len(items) * (bar_h + gap)
    s = ''
    for i, (lab, v, col, note) in enumerate(items):
        y = top + i * (bar_h + gap)
        bw = max(2, span * v / maxv)
        s += T(label_w - 10, y + bar_h / 2 + 4.5, lab, 13, INK, 'end', MONO, 500)
        s += Rect(label_w, y, bw, bar_h, col)
        txt = valfmt(v) if valfmt else f'{v}{unit}'
        nt = (f'<tspan fill="{SOFT}" font-family="{BODY}" font-weight="400" font-size="11" dx="8">{esc(note)}</tspan>'
              if note else '')
        s += (f'<text x="{label_w + bw + 8:.1f}" y="{y + bar_h / 2 + 4.5:.1f}" font-family="{MONO}" font-size="13" '
              f'fill="{INK}" font-weight="700">{esc(txt)}{nt}</text>')
    return svg(w, h, s, label)
