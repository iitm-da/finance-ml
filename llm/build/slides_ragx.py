"""LLM session, RAG block: chunking strategies, query expansion, the ontology. Imported by slides_rag.py."""
from common import *

ST_, FC, QX, ON = RG['strat'], RG['figc'], RG['qe'], RG['onto']
LT = RG['late']
REC = QX['rec']; NA = RG['n_ans']
SV = {s['strategy']: s for s in ST_}
r3 = lambda m, r, k='all': REC[m][r][k]['r3']
r1 = lambda m, r, k='all': REC[m][r][k]['r1']
f2 = lambda v: f'{v:.2f}'
f3 = lambda v: f'{v:.3f}'

# ================================================================ chunking: concept (build)
SENT = FC['sents']; NWD = sum(n for n, _ in SENT)
cum = [0]
for n, _ in SENT:
    cum.append(cum[-1] + n)
heads = [i for i, (_, h) in enumerate(SENT) if h]
EVI = heads[1] + 2                                    # second sentence of the second section: the online payment limit
EV = [cum[EVI], cum[EVI + 1]]
EV_TEXT = RG['kb_sample']['text'].split('. ')[1] + '.'
EV_Q = ON['link_ex'][1]['q']
assert len(EV_TEXT.split()) == EV[1] - EV[0], (EV_TEXT, EV)
LW = 292; X0 = LW + 10; SC = (1140 - X0 - 8) / NWD
X = lambda w: X0 + w * SC
ROWS = [('fixed, 40 words', 'fixed', 'cut every 40 words'),
        ('fixed, 40 words, 10 overlap', 'fixed + overlap', 'repeats the last 10 words'),
        ('sentence, up to 40 words', 'sentence', 'whole sentences, up to 40 words'),
        ('semantic', 'semantic', 'cut where neighbours differ most'),
        ('recursive, up to 40 words', 'recursive', 'headings first, then sentences'),
        ('by section', 'by section', 'one chunk per heading')]
RH = 35; Y0 = 50; FH = Y0 + RH * len(ROWS) + 2
s = Rect(X(EV[0]), 2, (EV[1] - EV[0]) * SC, FH - 4, GREEN, extra='fill-opacity=".13"')
s += T(LW, 17, 'the document', 12.5, INK, 'end', BODY, 700) + T(LW, 31, f'{len(SENT)} sentences, {NWD} words', 10, SOFT, 'end', MONO)
for i, (n, h) in enumerate(SENT):
    s += Rect(X(cum[i]) + .6, 8, n * SC - 1.2, 22, SLATE if h else ('#CFE9DD' if i == EVI else PALE), GREEN if i == EVI else LINE, 1)
for a, _ in FC['cuts']['by section'][1:]:
    s += L(X(a), 4, X(a), FH - 2, SOFT, 1, 'stroke-dasharray="3 4"')
for k, (name, lab, rule) in enumerate(ROWS):
    y = Y0 + k * RH; cuts = FC['cuts'][name]; lap = 'overlap' in name; g = ''
    g += T(LW, y + 12, lab, 12.5, INK, 'end', BODY, 700) + T(LW, y + 26, f'{rule} · {len(cuts)} chunks', 10, SOFT, 'end', MONO)
    for j, (a, b) in enumerate(cuts):
        yy, hh = (y + 2 + (j % 2) * 12, 10) if lap else (y + 4, 20)
        g += Rect(X(a) + 1, yy, (b - a) * SC - 2, hh, BLUE, extra=f'fill-opacity="{.9 if j % 2 == 0 else .55}"')
        if EV[0] < b < EV[1] and not lap:
            g += L(X(b), y - 1, X(b), y + 29, RED, 2.6)
    s += G_(k + 1, g)
CH_CONCEPT = slide('RAG', 'RAG &#183; Chunking &#183; Concept', 'Concept',
    'A chunk is the unit that is searched and the unit that is handed to the model. The same product guide, cut six ways.',
    fig(svg(1140, FH, s, 'One document cut by six chunking strategies'),
        f'{FC["head"]} &#183; dark blocks are headings, dashed lines are section breaks, the green band is one answering sentence',
        'A red mark is a cut that falls inside the answering sentence.') +
    '<div class="cols3">'
    '<div class="card sm" data-at="7"><h3>Add a header</h3><p>Write the document title and code at the top of every chunk before it is embedded. A chunk from the middle of a guide then still says which product it describes.</p></div>'
    '<div class="card sm g" data-at="8"><h3>Small to big</h3><p>Index single sentences, which match a question closely. Hand the model the whole section that the matched sentence came from.</p></div>'
    '<div class="card sm o" data-at="9"><h3>Late chunking</h3><p>Embed the whole document once, then average the token vectors inside each chunk. The chunk&#8217;s vector has seen the rest of the document.</p></div>'
    '</div>' + bld(9), 'build')

# ================================================================ chunking: strategies measured
SHORT = {'fixed, 40 words': 'fixed 40', 'fixed, 40 words, 10 overlap': 'fixed 40 + 10', 'sentence, up to 40 words': 'sentence', 'semantic': 'semantic',
         'recursive, up to 40 words': 'recursive', 'by section': 'by section', 'sentence + header': 'sentence + header',
         'by section + header': 'section + header', 'small to big: sentence found, section returned': 'small to big'}
COL = lambda n: GREEN if 'small' in n else (PURP if 'header' in n else (AMBER if 'fixed' in n else BLUE))
items = [(SHORT[x['strategy']], round(x['served'], 3), COL(x['strategy']),
          f"whole {pct(x['whole'], 0)} · {x['chunks']} chunks · {x['context words']:.0f} words to the model") for x in ST_]
FX, SE, SM, SEC, SECH, S2B = (SV[k] for k in ('fixed, 40 words', 'sentence, up to 40 words', 'semantic', 'by section', 'by section + header',
                                             'small to big: sentence found, section returned'))
CH_EXAMPLE = slide('RAG', 'RAG &#183; Chunking &#183; Example', 'Example continued 1',
    f"Dense retrieval, top 3. Fixed 40-word windows serve {pct(FX['served'])} of the {NA} answerable questions; a sentence index that returns sections serves {pct(S2B['served'])}.",
    '<div class="cols"><div class="colstack" style="flex:1.45">' +
    fig(hbars(items, w=700, label_w=150, bar_h=21, gap=8, maxv=2.3, valfmt=f3, label='Questions served by chunking strategy'),
        'share of answerable questions served &#183; whole = the answering sentence is uncut in some chunk') +
    '</div><div class="colstack">'
    f'<div class="callout warn sm"><b>Fixed windows cut sentences.</b> {pct(FX["whole"], 0)} of answering sentences survive whole at 40 words. Every strategy that cuts at a sentence end keeps 100%.</div>'
    f'<div class="callout sm"><b>Semantic cuts.</b> {pct(SM["served"])} served against {pct(SE["served"])} for plain sentence packing, at the price of embedding every sentence first.</div>'
    f'<div class="callout sm"><b>The header.</b> Section chunks serve {round(SEC["served"] * NA)} questions without it and {round(SECH["served"] * NA)} with it.</div>'
    f'<div class="callout good sm"><b>Small to big.</b> {S2B["chunks"]} one-sentence entries in the index, {S2B["context words"]:.0f} words handed to the model, {pct(S2B["served"])} served.</div>'
    f'<div class="callout sm"><b>Late chunking.</b> The same sentence chunks under e5-small: {round(LT[0]["served"] * NA)} questions served embedded alone, {round(LT[1]["served"] * NA)} embedded late. It pays when a chunk depends on text outside it, such as a product named only in the heading; these product guides repeat the name in the sentence.</div>'
    '</div></div>')

# ================================================================ chunking: widget
ev_w = EV_TEXT.split()
def mark(t):
    if EV_TEXT in t:
        return 'whole', esc(t).replace(esc(EV_TEXT), f'<b style="color:{GREEN}">{esc(EV_TEXT)}</b>')
    w = t.split()
    for k in range(len(ev_w) - 1, 1, -1):
        if w[-k:] == ev_w[:k]:
            return 'part', esc(' '.join(w[:-k])) + f' <b style="color:{RED}">{esc(" ".join(w[-k:]))}</b>'
        if w[:k] == ev_w[-k:]:
            return 'part', f'<b style="color:{RED}">{esc(" ".join(w[:k]))}</b> ' + esc(' '.join(w[k:]))
    return 'none', esc(t)
WNAMES = [n for n in SHORT if n in FC['texts']]
wd = {}
for n in WNAMES:
    m = [mark(t) for t in FC['texts'][n]]
    whole = [i + 1 for i, (k, _) in enumerate(m) if k == 'whole']; part = [i + 1 for i, (k, _) in enumerate(m) if k == 'part']
    wd[n] = {'c': [[h, k, len(t.split())] for (k, h), t in zip(m, FC['texts'][n])],
             'v': (f'whole in chunk {" and ".join(map(str, whole))}' if whole else f'split across chunks {" and ".join(map(str, part))}'), 'ok': bool(whole),
             'served': pct(SV[n]['served'])}
btns = ''.join(f'<button data-v="{n}"{" class=on" if i == 0 else ""}>{SHORT[n]}</button>' for i, n in enumerate(WNAMES))
CH_WIDGET = slide('RAG', 'RAG &#183; Chunking &#183; Example', 'The first six chunks of the Standard Current Account guide',
    f'Question: {esc(EV_Q)} The sentence that answers it is in colour.',
    f'<div class="wctl"><div class="seg" id="cx-seg">{btns}</div>{JSB}</div>'
    '<div class="out sm" id="cx-v" style="white-space:normal"></div>'
    '<div id="cx-list" style="display:grid;grid-template-columns:1fr 1fr;gap:8px 14px"></div>' + jdata('cx-data', wd))

# ================================================================ query expansion: concept (build)
EX = QX['explore'][0]
s = box(10, 46, 190, 58, 'question', 'as the member of staff typed it', INK)
s += G_(1, arrow(204, 75, 246, 75) + box(250, 46, 170, 58, 'expand', 'add words, or rewrite', PURP))
qy = [8, 52, 96]
for j, y in enumerate(qy):
    s += G_(1, arrow(424, 75, 476, y + 23) + box(480, y, 170, 46, f'query {j + 1}', '', PURP, tsize=12.5))
    s += G_(2, arrow(654, y + 23, 696, y + 23, BLUE) + box(700, y, 150, 46, 'search', 'top 20', BLUE, tsize=12.5))
    s += G_(3, arrow(854, y + 23, 906, 75, GREEN))
s += G_(3, box(910, 46, 210, 58, 'merge the rankings', 'reciprocal rank fusion, keep 3', GREEN))
hy = EX['hyde']; hy = hy[:hy.index('.') + 1]
QE_CONCEPT = slide('RAG', 'RAG &#183; Query expansion &#183; Concept', 'Concept',
    'Query expansion changes the question before it is searched, so that it shares words or meaning with the section that answers it.',
    fig(svg(1140, 146, s, 'Query expansion: one question becomes several searches'), 'between the question and the index') +
    f'<div class="out sm" style="white-space:normal"><b>question</b> &#160;{esc(EX["q"])}</div>'
    '<div class="cols3">'
    f'<div class="card sm o" data-at="4"><h3>Feedback terms</h3><p>Search once. Add the 8 most distinctive words of the top 3 sections and search again. No model.</p><div class="out xs" style="white-space:normal">+ {esc(", ".join(EX["prf"][:6]))}</div></div>'
    f'<div class="card sm" data-at="5"><h3>Multi-query</h3><p>The language model writes 3 rewordings. All 4 versions are searched and the rankings merged.</p><div class="out xs" style="white-space:normal">{esc(EX["rw"][1])}</div></div>'
    f'<div class="card sm g" data-at="6"><h3>HyDE</h3><p>The model writes a made-up passage that would answer the question. The passage is searched for.</p><div class="out xs" style="white-space:normal">{esc(hy)}</div></div>'
    '</div>' + bld(6), 'build')

# ================================================================ query expansion: recall
METH = ['no expansion', 'feedback terms', 'multi-query', 'HyDE']
MCOL = {'no expansion': GREY, 'feedback terms': AMBER, 'multi-query': BLUE, 'HyDE': GREEN, 'ontology synonyms': PURP, 'no expansion + filter': PURP}
def groups(gs, meths, W=1120, H=250, bw=52):
    """gs: (retriever, kind, label) -> grouped bars of recall@3"""
    s = ''; x0 = 50; ph = H - 62; gw = (W - x0 - 6) / len(gs)
    for v in (0, .25, .5, .75, 1):
        y = 18 + (1 - v) * ph
        s += L(x0, y, W - 6, y, HAIR, 1) + T(x0 - 6, y + 4, f2(v), 10, SOFT, 'end', MONO)
    for i, (r, kd, lab) in enumerate(gs):
        gx = x0 + i * gw + (gw - len(meths) * (bw + 6)) / 2
        for j, m in enumerate(meths):
            v = r3(m, r, kd); h = v * ph
            s += Rect(gx + j * (bw + 6), 18 + ph - h, bw, h, MCOL[m]) + T(gx + j * (bw + 6) + bw / 2, 18 + ph - h - 5, f2(v), 11, INK, 'middle', MONO, 700)
        s += T(x0 + i * gw + gw / 2, H - 24, lab, 12.5, INK, 'middle', MONO, 700) + T(x0 + i * gw + gw / 2, H - 8, f'{RG["kinds"][kd]} {kd} questions', 10, SOFT, 'middle', MONO)
    return s
leg = lambda ms: ' &#160; '.join(f'<span class="tag" style="background:{MCOL[m]}">&#160;</span>{m}' for m in ms)
G4 = [('BM25', 'paraphrase', 'BM25'), ('BM25', 'code', 'BM25'), ('dense', 'paraphrase', 'dense'), ('dense', 'code', 'dense')]
BG = RG['gen']['+ strict prompt']; XG = QX['gen'][QX['best']]
dq = round((XG['ans']['correct'] - BG['ans']['correct']) * NA)
QE_EXAMPLE = slide('RAG', 'RAG &#183; Query expansion &#183; Example', 'Example',
    f"Recall@3 with each expansion. HyDE lifts BM25 on paraphrase questions from {f2(r3('no expansion', 'BM25', 'paraphrase'))} to {f2(r3('HyDE', 'BM25', 'paraphrase'))}; feedback terms lower it to {f2(r3('feedback terms', 'BM25', 'paraphrase'))}.",
    fig(svg(1120, 250, groups(G4, METH), 'Recall at 3 by expansion method, retriever and question kind'), 'recall@3', leg(METH)) +
    '<div class="kv">' + ''.join(tile(f'{m}, hybrid + reranker', f3(r3(m, 'hybrid + reranker')), 'ok' if r3(m, 'hybrid + reranker') > r3('no expansion', 'hybrid + reranker') else '') for m in METH) + '</div>'
    '<div class="cols nat">'
    f'<div class="callout warn sm"><b>Feedback repeats the first search.</b> When that search missed, the added words come from the wrong sections. On code questions BM25 falls from {f2(r3("no expansion", "BM25", "code"))} to {f2(r3("feedback terms", "BM25", "code"))}.</div>'
    f'<div class="callout sm"><b>Answers.</b> With {QX["best"]} ahead of the reranker and the strict prompt, correct answers go from {pct(BG["ans"]["correct"])} to {pct(XG["ans"]["correct"])}: {dq} of {NA} questions. Retrieval was at {f3(r3("no expansion", "hybrid + reranker"))} before expansion.</div>'
    '</div>')

# ================================================================ query expansion: widget
QE_WIDGET = slide('RAG', 'RAG &#183; Ontology &#183; Example', 'Expansion, question by question',
    f"{len(QX['explore'])} questions, most of them missed by BM25. For each: what every method adds, and the rank of the answering section afterwards.",
    f'<div class="wctl"><button class="wbtn" id="qx-prev">&#8249; previous</button><button class="wbtn" id="qx-next">next &#8250;</button>{JSB}</div>'
    '<div class="cols"><div class="colstack">'
    '<div class="out" id="qx-q" style="min-height:58px;white-space:normal"></div>'
    '<div class="outlbl">rank of the answering section (1 is best; top 3 reach the model)</div><div id="qx-pos"></div>'
    '</div><div class="colstack">'
    '<div class="outlbl">feedback terms</div><div class="out xs" id="qx-prf" style="white-space:normal"></div>'
    '<div class="outlbl">rewordings by the model</div><div class="out xs" id="qx-rw" style="white-space:normal;min-height:62px"></div>'
    '<div class="outlbl">HyDE passage</div><div class="out xs" id="qx-hy" style="white-space:normal;min-height:62px"></div>'
    '<div class="outlbl">ontology synonyms</div><div class="out xs" id="qx-syn" style="white-space:normal"></div>'
    '</div></div>' + jdata('qx-data', {'q': QX['explore'], 'm': METH + ['ontology synonyms']}))


# ================================================================ ontology: what the linker finds, from the notebook's own code
_src = open(os.path.join(HERE, 'llm_part.py'), encoding='utf-8').read()
_ns = {'json': json, 're': __import__('re'), 'DATA': os.path.join(HERE, '..', 'data') + '/'}
exec(_src[_src.index("onto = json.load(open(DATA + 'ontology.json'"):_src.index('for q in (Q[3], Q[5],')], _ns)
ONT = _ns['onto']
_qs = [json.loads(l) for l in open(os.path.join(HERE, '..', 'data', 'questions.jsonl'), encoding='utf-8')]
N_PARA = sum(r['kind'] == 'paraphrase' for r in _qs)
N_PARA_NOLINK = sum(r['kind'] == 'paraphrase' and not _ns['synonyms'](r['question']) for r in _qs)
assert sum(bool(_ns['synonyms'](r['question'])) for r in _qs if r['gold']) == ON['n_syn']

# ================================================================ ontology: concept (build)
def cbox(x, y, w, name, col=SLATE):
    return Rect(x, y, w, 32, '#FFFFFF', HAIR, 1) + Rect(x, y, 4, 32, col) + T(x + w / 2 + 2, y + 21, name, 12.5, INK, 'middle', BODY, 700)
link_ = lambda x1, y1, x2, y2: f'<path d="M{x1},{y1} C{(x1 + x2) / 2},{y1} {(x1 + x2) / 2},{y2} {x2},{y2}" fill="none" stroke="{SOFT}" stroke-width="1.4"/>'
inst_of = lambda c: [x for x in ONT['instances'] if x['class'] == c]
s = T(10, 14, 'classes', 11, SOFT, 'start', MONO, 700)
s += cbox(10, 104, 100, 'product') + cbox(160, 56, 110, 'account') + cbox(160, 168, 110, 'credit card')
s += cbox(320, 26, 210, 'personal current account') + cbox(320, 100, 210, 'business account')
s += link_(110, 120, 160, 72) + link_(110, 120, 160, 184) + link_(270, 72, 320, 42) + link_(270, 72, 320, 116)
s += T(135, 92, 'is a', 10, SOFT, 'middle', MONO)
g = T(580, 14, 'instances, each with its document code', 11, SOFT, 'start', MONO, 700)
for cls, x, y0 in (('personal current account', 580, 22), ('business account', 580, 104), ('credit card', 320, 164)):
    for j, it in enumerate(inst_of(cls)):
        y = y0 + j * 19
        g += T(x + 8, y + 14, it['name'], 12, INK, 'start', BODY, 500) + T(x + 200, y + 14, it['code'], 11.5, BLUE, 'start', MONO, 700)
    n = len(inst_of(cls)); px = x - 50 if cls != 'credit card' else 270
    g += L(x, y0 + 2, x, y0 + n * 19 + 2, BLUE, 2) + L(px, (184 if cls == 'credit card' else y0 + 20), x, (184 if cls == 'credit card' else y0 + 20), BLUE, 1.4)
s += G_(2, g)
TSHOW = ['cash machine', 'early repayment charge', 'fit note', 'multi-factor authentication', 'payment holiday']
g = T(850, 14, 'terms: the document\'s word, then the others', 11, SOFT, 'start', MONO, 700)
for j, t in enumerate(TSHOW):
    c = next(c for c in ONT['concepts'] if c['pref'] == t); y = 24 + j * 42
    g += Rect(850, y, 282, 36, '#FFFFFF', HAIR, 1) + Rect(850, y, 4, 36, PURP)
    g += T(862, y + 15, t, 12.5, INK, 'start', BODY, 700) + T(862, y + 29, (lambda a: a if len(a) < 40 else ', '.join(c['alt'][:2]))(', '.join(c['alt'][:3])), 10.5, SOFT, 'start', MONO)
s += G_(3, g)
ON_CONCEPT = slide('RAG', 'RAG &#183; Ontology &#183; Concept', 'Concept',
    'An ontology is a written model of a domain: the kinds of thing in it, the things themselves, and the words people use for them.',
    fig(svg(1140, 236, s, 'The Northmere ontology: classes, instances and terms'), 'llm/data/ontology.json, in part') +
    '<div class="cols3">'
    '<div class="card sm" data-at="1"><h3>Classes</h3><p>A kind of thing and its parent. A business account is an account; an account is a product.</p></div>'
    '<div class="card sm" data-at="2"><h3>Instances</h3><p>The products and documents themselves. Each carries the code of the document that describes it.</p></div>'
    '<div class="card sm" data-at="3"><h3>Terms</h3><p>The word the documents use, with the other words staff use for the same thing.</p></div>'
    '</div>'
    '<div class="kv">' + tile('classes', ON['n_classes']) + tile('instances', ON['n_inst']) + tile('terms', ON['n_terms']) + tile('labels in all', ON['n_labels']) + '</div>'
    '<div class="callout sm">Written by hand from the 18 documents and ordinary banking vocabulary. It was not tuned on the test questions.</div>' + bld(3), 'build')

# ================================================================ ontology: mechanism (build)
CW = 8.42                                             # width of one 14 px monospace character
def qline(x, y, parts):
    """parts: (text, colour or None) -> the question in monospace with the linked spans tinted; returns svg and span centres"""
    s = ''; mid = []
    for t, col in parts:
        w = len(t) * CW
        if col:
            s += Rect(x - 2, y - 15, w + 4, 21, col, extra='fill-opacity=".16"') + L(x - 2, y + 7, x + w + 2, y + 7, col, 2)
            mid.append(x + w / 2)
        s += T(x, y, t.replace(' ', '\u00a0'), 14, INK, 'start', MONO, 700 if col else 400)
        x += w
    return s, mid
q1, m1 = qline(20, 24, [('What is the daily ', None), ('ATM', PURP), (' limit on the ', None), ('Standard account', BLUE), ('?', None)])
q2, m2 = qline(610, 24, [('Which ', None), ('account', GREEN), (' has the highest daily cash machine limit?', None)])
L1 = ON['link_ex'][0]
s = q1
s += G_(1, arrow(m1[0], 34, 125, 62, PURP) + box(20, 64, 210, 48, 'term: cash machine', 'ATM, cashpoint, cash dispenser', PURP, tsize=12.5) + arrow(125, 114, 125, 142, PURP) +
        box(20, 144, 210, 52, '1 · synonyms', 'join the query', PURP))
s += G_(2, arrow(m1[1], 34, 385, 62, BLUE) + box(260, 64, 250, 48, 'instance: ' + L1['inst'][0][0], 'document ' + L1['inst'][0][1], BLUE, tsize=12.5) + arrow(385, 114, 385, 142, BLUE) +
        box(260, 144, 250, 52, '2 · filter', 'search only ' + L1['inst'][0][1], BLUE))
s += G_(3, q2 + arrow(m2[0], 34, 735, 62, GREEN) + box(610, 64, 250, 48, 'class: account', '5 instances, PG-01 to PG-05', GREEN, tsize=12.5) + arrow(735, 114, 735, 142, GREEN) +
        box(610, 144, 510, 52, '3 · fan-out', 'one search per instance, one section kept from each', GREEN))
s += L(560, 6, 560, 200, HAIR, 1)
ON_MECH = slide('RAG', 'RAG &#183; Ontology &#183; Mechanism', 'Mechanism',
    'Linking finds the terms, the instance and the class that a question mentions. Each link changes the search in its own way.',
    fig(svg(1140, 204, s, 'Two questions linked to the ontology, and the three uses of the links'), 'two questions, three links') +
    '<div class="cols3">'
    '<div class="card sm" data-at="1"><h3>1. Synonyms</h3><p>Every other label of a linked term joins the query. This is query expansion from a dictionary the bank controls, with no call to a model.</p></div>'
    '<div class="card sm" data-at="2"><h3>2. Filter</h3><p>A named product or document code restricts the search to that document. A filter is a hard rule: a wrong link hides the right section.</p></div>'
    '<div class="card sm g" data-at="3"><h3>3. Fan-out</h3><p>A question about a class is searched once per instance. Top-k ranks sections by likeness to the question and has no notion of covering every product.</p></div>'
    '</div>' + bld(3), 'build')

# ================================================================ ontology: synonyms and filter, measured
def bars(gs, W=540, H=236, bw=62):
    """gs: (method, retriever, kind, label) -> bars of recall@3"""
    s = ''; x0 = 46; ph = H - 50
    for v in (0, .25, .5, .75, 1):
        y = 18 + (1 - v) * ph
        s += L(x0, y, W - 4, y, HAIR, 1) + T(x0 - 6, y + 4, f2(v), 10, SOFT, 'end', MONO)
    step = (W - x0 - 4) / len(gs)
    for i, (m, r, kd, lab) in enumerate(gs):
        v = r3(m, r, kd); h = v * ph; x = x0 + i * step + (step - bw) / 2
        s += Rect(x, 18 + ph - h, bw, h, MCOL[m]) + T(x + bw / 2, 18 + ph - h - 5, f3(v), 11.5, INK, 'middle', MONO, 700)
        s += T(x + bw / 2, H - 16, lab, 11, INK, 'middle', MONO, 600)
    return svg(W, H, s, 'Recall at 3')
SY = 'ontology synonyms'; FI = 'no expansion + filter'
ON_EXAMPLE = slide('RAG', 'RAG &#183; Ontology &#183; Example', 'Example',
    f"Uses 1 and 2. Synonyms lift BM25 on paraphrase questions from {f2(r3('no expansion', 'BM25', 'paraphrase'))} to {f2(r3(SY, 'BM25', 'paraphrase'))}. The filter lifts dense retrieval on code questions from {f2(r3('no expansion', 'dense', 'code'))} to {f2(r3(FI, 'dense', 'code'))}.",
    '<div class="cols"><div class="colstack">' +
    fig(bars([('no expansion', 'BM25', 'paraphrase', 'none'), ('multi-query', 'BM25', 'paraphrase', 'multi-query'), ('HyDE', 'BM25', 'paraphrase', 'HyDE'), (SY, 'BM25', 'paraphrase', 'synonyms')]),
        f'BM25, recall@3, {N_PARA} paraphrase questions') +
    f'<div class="callout sm"><b>No model call.</b> The synonyms reach {f2(r3(SY, "BM25", "paraphrase"))} from a lookup. Multi-query and HyDE reach {f2(r3("multi-query", "BM25", "paraphrase"))} and {f2(r3("HyDE", "BM25", "paraphrase"))} with one generation per question.</div>'
    f'<div class="callout warn sm"><b>Coverage.</b> {N_PARA_NOLINK} of the {N_PARA} paraphrase questions mention no term of the ontology and are searched unchanged.</div>'
    '</div><div class="colstack">' +
    fig(bars([('no expansion', 'dense', 'code', 'none'), ('HyDE', 'dense', 'code', 'HyDE'), (FI, 'dense', 'code', 'filter')], bw=74),
        f'dense, recall@3, {RG["kinds"]["code"]} questions that cite a document code') +
    f'<div class="callout sm"><b>Who is linked.</b> {ON["n_link"]} of the {NA} answerable questions name a product or a code. The answer was outside the named document for {ON["n_harm"]} of them.</div>'
    f'<div class="callout sm"><b>Recall@1 for BM25</b> goes from {f3(r1("no expansion", "BM25"))} to {f3(r1(SY, "BM25"))} with synonyms, the highest of the five BM25 variants.</div>'
    '</div></div>')
assert r1(SY, 'BM25') == max(r1(m, 'BM25') for m in METH + [SY])


# ================================================================ expansion and ontology: drawbacks
hy_fig = __import__('re').search(r'[$£]\s?\d[\d,.]*|GBP\s?\d[\d,.]*|\d+(?:\.\d+)?%', EX['hyde'])
ON_DRAW = slide('RAG', 'RAG &#183; Expansion and ontology &#183; Drawbacks', 'Drawbacks',
    'What query expansion and an ontology cost, on this knowledge base.',
    '<div class="cols3">'
    f'<div class="card big r"><h3>Someone maintains it</h3><p>{ON["n_terms"]} terms and {ON["n_inst"]} instances for 18 documents, written by hand. A product left out of the ontology is skipped by the fan-out, and no error is raised.</p></div>'
    f'<div class="card big o"><h3>A dictionary has gaps</h3><p>{N_PARA_NOLINK} of the {N_PARA} paraphrase questions link to no term. A lookup helps where an entry exists. The model-based methods reword any question.</p></div>'
    '<div class="card big"><h3>Labels collide</h3><p>"Business card" is a label of the Business Credit Card and also an ordinary object. A wrong link followed by a filter removes the right section from the search.</p></div>'
    '</div>'
    '<div class="cols nat">'
    f'<div class="callout"><b>Expansion by a model costs a generation.</b> Rewording took {QX["secs"]["rewrite"]:.0f} seconds and HyDE {QX["secs"]["hyde"]:.0f} seconds for {QX["n_q"]} questions on this GPU, before any search ran.</div>'
    f'<div class="callout warn"><b>HyDE invents figures.</b> Its passage for the overdraft question states "{esc(hy_fig.group(0).rstrip('.,'))}". The passage is a search key. It must stay out of the prompt that the answering model reads.</div>'
    '</div>')


# ================================================================ ontology: fan-out on comparison questions
MU = ON['multi']['answer directly']; MC = ON['multi_code']; NM = ON['n_multi']
PN_ = {'top 3': 'top 3', 'top n (n = products in the class)': 'top n', 'ontology fan-out': 'fan-out', 'fan-out, current sections only': 'fan-out, current only'}
FO = MU['fan-out, current sections only']; T3 = MU['top 3']
cnt = lambda v: round(v * NM)
cov_items = [(PN_[k], round(MU[k]['covered'], 3), GREEN if MU[k]['covered'] == 1 else BLUE, f"{cnt(MU[k]['covered'])} of {NM}") for k in PN_]
ans_rows = [(PN_[k] + ', model compares', MU[k]) for k in PN_] + [('fan-out, current only, code compares', MC)]
fx_ = ON['figs_ex']
ON_MULTI = slide('RAG', 'RAG &#183; Ontology &#183; Example', 'Example continued 1',
    f"Use 3, on {NM} questions that compare the products of a class. Fan-out over current sections retrieves every needed section for {pct(FO['covered'], 0)} of them; top 3 does so for {pct(T3['covered'], 0)}.",
    '<div class="cols"><div class="colstack" style="flex:1.15">' +
    fig(hbars(cov_items, w=640, label_w=200, bar_h=22, gap=8, maxv=1.0, valfmt=lambda v: pct(v, 0), label='Questions with every needed section retrieved'),
        'questions with every needed section in the prompt &#183; top n: n = products in the class') +
    fig(stacked(ans_rows, w=640, label_w=300, bar_h=22, gap=8, label='Answers to comparison questions by pipeline'), f'answers to the {NM} questions', GLEG) +
    '</div><div class="colstack">' +
    out(ON['multi_ex']['q'] + '\n' + '\n'.join(f'{n:26s}{v:>8,.0f}' for n, v in fx_[0]) + f'\nmax() -> {fx_[1]}', 'xs', 'one figure read from each section, compared in code') +
    f'<div class="callout warn sm"><b>Full coverage, same answers.</b> With every section present the model names the right product on {FO["correct, covered"]} of {NM}. Under top 3, {T3["correct, not covered"]} of its {cnt(T3["correct"])} correct answers came with a needed section missing.</div>'
    f'<div class="callout good sm"><b>The model reads, code compares.</b> One call per section returns one figure, {MC["calls"]} calls in all. max() or min() names the product: {cnt(MC["correct"])} of {NM} correct.</div>'
    f'<div class="callout sm"><b>Archived sections.</b> Fan-out alone covers {pct(MU["ontology fan-out"]["covered"], 0)}: for some products it kept a superseded section. The status filter takes it to {pct(FO["covered"], 0)}.</div>'
    '</div></div>')

CHUNK_X = [CH_EXAMPLE, CH_WIDGET]
QE_X = [QE_CONCEPT, QE_EXAMPLE]
ON_X = [ON_CONCEPT, ON_MECH, ON_EXAMPLE, ON_MULTI, QE_WIDGET, ON_DRAW]
