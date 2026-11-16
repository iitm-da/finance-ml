"""Shared by the LLM session's slide scripts: measured numbers, colours, small helpers."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'build'))
from fx import *

LM = json.load(open(os.path.join(HERE, 'm_llm.json')))
RG, EV, PR, PF = LM['rag'], LM['eval'], LM['prompt'], LM['peft']
NLP_IE = json.load(open(os.path.join(HERE, '..', '..', 'nlp', 'build', 'm_nlp.json')))['ie']
NLP_SE = json.load(open(os.path.join(HERE, '..', '..', 'nlp', 'build', 'm_nlp.json')))['senti']
JSB = '<span class="jsbadge">browser JS</span>'
GCOL = {'correct': GREEN, 'abstained': GREY, 'wrong': RED}
mean = lambda d: sum(d.values()) / len(d)

def stacked(rows, w=700, label_w=250, bar_h=28, gap=12, label='stacked bars', note=None):
    """rows: (label, {'correct':..,'abstained':..,'wrong':..}) -> horizontal bars that add to 100%"""
    span = w - label_w - 16; s = ''
    for i, (lab, d) in enumerate(rows):
        y = 6 + i * (bar_h + gap); x = label_w
        s += T(label_w - 10, y + bar_h / 2 + 4.5, lab, 12, INK, 'end', MONO, 500)
        for k in ('correct', 'abstained', 'wrong'):
            bw = span * d[k]
            s += Rect(x, y, bw, bar_h, GCOL[k])
            if bw > 34:
                s += T(x + bw / 2, y + bar_h / 2 + 4.5, f'{d[k] * 100:.0f}%', 11.5, '#FFFFFF' if k != 'abstained' else INK, 'middle', MONO, 700)
            x += bw
    return svg(w, 6 + len(rows) * (bar_h + gap), s, label)

GLEG = (f'<span class="tag" style="background:{GREEN}">&#160;</span>correct &#160; <span class="tag" style="background:{GREY}">&#160;</span>abstained &#160; '
        f'<span class="tag" style="background:{RED}">&#160;</span>wrong')
