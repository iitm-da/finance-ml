"""Shared by the NLP session's slide scripts: measured numbers, colours, small helpers."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'build'))
from fx import *

NL = json.load(open(os.path.join(HERE, 'm_nlp.json')))
AT, SE, NE, DC, IE = NL['attn'], NL['senti'], NL['ner'], NL['doc'], NL['ie']
JSB = '<span class="jsbadge">browser JS</span>'
ECOL = {'PER': BLUE, 'ORG': PURP, 'LOC': GREEN, 'MONEY': AMBER, 'DATE': SLATE, 'ACCT': RED}

def f3(v):
    return f'{v:.3f}'

def chiprow(x, y, items, h=26, pad=9, gap=6, size=12.5, fill='#FFFFFF', stroke=HAIR, col=INK, font=MONO, cw=7.6):
    """a row of text boxes; returns (svg, x after the last box)"""
    s = ''
    for it in items:
        w = len(it) * cw + 2 * pad
        s += Rect(x, y, w, h, fill, stroke, 1) + T(x + w / 2, y + h / 2 + 4.3, it, size, col, 'middle', font, 500)
        x += w + gap
    return s, x
