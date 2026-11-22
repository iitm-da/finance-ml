"""Apply the CXO-masterclass theme (Geist, navy / blue / red / yellow on white) to a built session deck.

usage: python3 retheme.py deck.html [deck2.html ...]
assemble.py and assemble_session.py call retheme() before they write, so a rebuild keeps the theme.
Idempotent: running it on an already themed deck changes nothing."""
import re, sys

MARK = '/* theme: masterclass */'

# old colour -> new colour, applied everywhere (CSS, inline SVG, widget JS)
MAP = {
    # ink and greys
    '14222B': '0E1330', '31444F': '1A2152', '26404E': '1A2152', '53666F': '5D6488',
    '7C8C95': '7A80A3', '8FA0A9': '8A90B0', '9FB0B9': '9AA3CC', '9BAAB3': '9AA3CC',
    '7E9AA8': '8A90B0', 'B9C4CA': 'B9BED6', 'C3CFD5': 'CED3E6', 'C9D3D8': 'CED3E6',
    'C9D3D9': 'CED3E6', 'DCE4E8': 'E2E5F0', 'E4EAED': 'E8EBF4', 'EEF2F4': 'F4F6FB',
    'F3F7F9': 'F4F6FB', 'F7F7F7': 'F4F6FB', 'E9EDEF': 'FFFFFF',
    # dark grounds
    '0E1A21': 'E4E7F1', '122029': '0E1330', '2C4A5C': '2B4BFF',
    # accents
    '125F9E': '2B4BFF', '2166AC': '2B4BFF', 'B02F3B': 'E8382D', 'D98A92': 'F29A93',
    '8F5D0A': 'A87900', 'D1E5F0': 'E8ECFF', 'EAF4FA': 'E8ECFF',
    'FFF5F5': 'FDE9E7', 'FFF9F9': 'FDE9E7', 'FFF3F3': 'FDE9E7', 'FBECEE': 'FDE9E7', 'FDF3F4': 'FDE9E7',
    'FFFBF2': 'FFF6D1', 'FFF6E0': 'FFF6D1', 'FBE6C0': 'FFF6D1',
}

FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300..800'
         '&family=Geist+Mono:wght@400..700&display=swap" rel="stylesheet"/>')

CSS = MARK + '''
:root{--mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --disp:"Geist",system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  --body:"Geist",system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
body{font-feature-settings:"ss01","liga" 0,"calt" 0}
#deck{box-shadow:0 24px 70px rgba(14,19,48,.16)}
h1.t{font-weight:600;letter-spacing:-.028em}
h2.s{font-weight:600}
.eyebrow{color:#2B4BFF}
.eyebrow::after{background:#E2E5F0}
#pbar{background:#2B4BFF}
.card,.fig,.out{border-radius:8px}
pre.code{border-radius:8px;border-left:0}
/* cover: navy ground, as on the masterclass decks */
.slide.cover{background:#0E1330;color:#fff}
.cover h1{font-weight:600;letter-spacing:-.035em;color:#fff}
.cover .kicker{color:#FFC800}
.cover .pt{color:#AEB6E0;font-weight:500}
.cover .rule{background:#2B4BFF}
.cover .sub{color:#D5D9EE}
.cover .by{color:#fff;font-weight:600}
.cover .by span{color:#AEB6E0}
.coverpat{opacity:.55}
#nav button{background:#0E1330;border-color:#0E1330;color:#fff;border-radius:8px}
#nav button:hover{background:#2B4BFF}
#nav button:focus-visible{outline-color:#2B4BFF}
#hint{color:#8A90B0}
'''


def retheme(html):
    if MARK in html:
        return html
    html = re.sub(r'<link href="https://fonts\.googleapis\.com/css2\?[^"]*" rel="stylesheet"/?>', FONTS, html, count=1)
    pat = re.compile('#(' + '|'.join(MAP) + r')\b', re.I)
    html = pat.sub(lambda m: '#' + MAP[m.group(1).upper()], html)
    i = html.rindex('</style>', 0, html.index('</head>'))
    return html[:i] + CSS + html[i:]


if __name__ == '__main__':
    for p in sys.argv[1:]:
        s = open(p, encoding='utf-8').read()
        t = retheme(s)
        open(p, 'w', encoding='utf-8').write(t)
        print(p, 'themed' if t != s else 'already themed')
