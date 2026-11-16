"""NLP session, part 3: named entity recognition."""
from common import *

C = []
add = C.append
NM = NE['m']; TYPES = NE['types']; GENRES = ['narrative', 'note', 'news', 'complaint']
GNAME = {'narrative': 'payment narrative', 'note': 'case note', 'news': 'news line', 'complaint': 'complaint'}
EX = NE['examples']
DEVN = 'GPU'

def ent_html(tokens, spans, size=13.5):
    """tokens with entity spans drawn as coloured boxes"""
    start = {a: (ty, b) for ty, a, b in spans}
    out, i = [], 0
    while i < len(tokens):
        if i in start:
            ty, b = start[i]
            out.append(f'<span style="background:{ECOL[ty]}1F;border-bottom:2px solid {ECOL[ty]};padding:1px 3px;white-space:nowrap">{esc(" ".join(tokens[i:b + 1]))}'
                       f'<small style="font-family:var(--mono);font-size:9px;color:{ECOL[ty]};font-weight:700;margin-left:4px">{ty}</small></span>')
            i = b + 1
        else:
            out.append(esc(tokens[i])); i += 1
    return f'<div style="font-size:{size}px;line-height:1.9">' + ' '.join(out) + '</div>'

def prf_table(res, types, extra=None):
    rows = [[f'<span class="tag" style="background:{ECOL[t]}">{t}</span>', res[t]['n'], f"{res[t]['p']:.3f}", f"{res[t]['r']:.3f}", f"<b>{res[t]['f1']:.3f}</b>"] for t in types]
    rows.append(['<b>all</b>', res['all']['n'], f"{res['all']['p']:.3f}", f"{res['all']['r']:.3f}", f"<b>{res['all']['f1']:.3f}</b>"])
    return table(['type', 'entities', 'precision', 'recall', 'F1'], rows, 'd sm free')

# ================================================================ concept: BIO tags
toks, tags = NE['align_tokens'], NE['align_tags']
s = ''; x = 8; cw = []
for t in toks:
    cw.append(max(46, len(t) * 9.2 + 18))
scale = min(1.0, 1100 / sum(cw))
for t, g in zip(toks, tags):
    w = cw[toks.index(t)] * scale if False else max(46, len(t) * 9.2 + 18) * scale
    ty = g[2:] if g != 'O' else None
    s += Rect(x, 8, w - 4, 30, '#FFFFFF', LINE, 1) + T(x + (w - 4) / 2, 28, t, 12.5, INK, 'middle', MONO, 600)
    s += Rect(x, 44, w - 4, 26, ECOL[ty] if ty else '#EEF2F4') + T(x + (w - 4) / 2, 61.5, g, 11, '#FFFFFF' if ty else SOFT, 'middle', MONO, 700)
    x += w
legend = ' &#160; '.join(f'<span class="tag" style="background:{ECOL[t]}">{t}</span>{d}' for t, d in
                         (('PER', 'person'), ('ORG', 'company'), ('LOC', 'place'), ('MONEY', 'amount'), ('DATE', 'date'), ('ACCT', 'account number')))
add(slide('Entities', 'Entities &#183; Concept', 'Concept',
          'Named entity recognition gives every token a tag. B- opens an entity, I- continues it, O is everything else.',
          fig(svg(1120, 78, s, 'A sentence with BIO tags'), 'one test sentence, token by token', legend) +
          '<div class="cols3">'
          '<div class="card sm"><h3>Why a bank wants it</h3><p>Screen payment text against sanctions lists. Find the parties in a contract. Remove personal data from a complaint before it is used for training.</p></div>'
          '<div class="card sm"><h3>Why tags, not a list</h3><p>The tagger works on any sentence length and returns where each entity is, so the entity can be highlighted, redacted or linked to a customer record.</p></div>'
          '<div class="card sm o"><h3>Two kinds of entity</h3><p>Amounts, dates and account numbers have a shape. Names of people, companies and places have no shape and depend on context.</p></div>'
          '</div>'))

# ================================================================ data: four kinds of text
cards = ''
for g in GENRES:
    e = EX[g][0]['ft']
    cards += (f'<div class="card sm"><h3>{GNAME[g].capitalize()} <small style="font-family:var(--mono);font-weight:400;color:var(--soft)">{NE["genre_n"][g]} test sentences</small></h3>'
              + ent_html(e['tokens'], e['gold'], 13) + '</div>')
add(slide('Entities', 'Entities &#183; Data', 'The tagged sentences, by text type',
          f"{comma(NE['n_train'])} training and {NE['n_test']} test sentences, simulated, with {comma(sum(NE['n_ent'].values()))} tagged entities in the test set.",
          f'<div class="cols"><div class="colstack">{"".join(cards.split("</div></div>")[0:1])}</div></div>' if False else
          '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">' + cards + '</div>'
          '<div class="cols nat"><div class="callout"><b>The test set is new text.</b> Its sentence patterns, people, companies and places do not appear in the training set. A tagger that memorised the training names scores nothing on them.</div>'
          '<div class="callout warn"><b>Payment narratives are in capitals</b> and have no grammar. They are the text a bank holds most of.</div></div>'))

# ================================================================ test: how an entity is scored
gold = [('ORG', 4, 6)]
tk = ['I', 'was', 'charged', 'by', 'Thornfield', 'Capital', 'Ltd', 'on', '3', 'May', '2025']
def rowsv(y, label, spans, okc):
    s = T(8, y + 20, label, 11.5, INK, 'start', MONO, 700); x = 150
    for i, t in enumerate(tk):
        w = len(t) * 9 + 18
        sp = next(((ty, a, b) for ty, a, b in spans if a <= i <= b), None)
        s += Rect(x, y, w - 3, 30, (ECOL[sp[0]] if sp else '#FFFFFF'), LINE, 1) + T(x + (w - 3) / 2, y + 20, t, 12.5, '#FFFFFF' if sp else INK, 'middle', MONO, 600)
        x += w
    s += T(x + 14, y + 20, okc[0], 12, okc[1], 'start', MONO, 700)
    return s
s = rowsv(6, 'label', [('ORG', 4, 6), ('DATE', 8, 10)], ('2 entities', INK))
s += rowsv(50, 'prediction A', [('ORG', 4, 6), ('DATE', 8, 10)], ('2 found, 0 false', GREEN))
s += rowsv(94, 'prediction B', [('ORG', 4, 5), ('DATE', 8, 10)], ('1 found, 1 false, 1 missed', RED))
s += rowsv(138, 'prediction C', [('PER', 4, 6), ('DATE', 8, 10)], ('1 found, 1 false, 1 missed', RED))
add(slide('Entities', 'Entities &#183; Test', 'Test',
          'An entity counts as found only when the first token, the last token and the type all match the label. Most of a company name earns nothing.',
          fig(svg(1120, 176, s, 'Three predictions scored against a label'), 'entity-level scoring') +
          '<div class="cols"><div class="colstack">' +
          out("""precision = found / predicted      how much of the output is right
recall    = found / labelled       how much of the truth was found
F1        = 2 x precision x recall / (precision + recall)""", '', 'the three scores') +
          '</div><div class="colstack">'
          '<div class="callout"><b>B drops "Ltd".</b> Its span is counted once as a false entity and once as a missed one. A sanctions screen that matches on the full legal name fails on it.</div>'
          '<div class="callout sm"><b>Token accuracy hides this.</b> Prediction B has 10 of 11 tokens right, and most tokens in any text are O.</div>'
          '</div></div>'))

# ================================================================ example: rules
R = NM['rules']
add(slide('Entities', 'Entities &#183; Example', 'Example',
          f"Three regular expressions for the entities that have a shape. Over the three types they score F1 {R['all']['f1']:.3f} with no training data.",
          '<div class="cols"><div class="colstack" style="flex:1.25">' +
          code(r"""RULES = {
 'MONEY': r'(?:£|(?:GBP|EUR)\s)\d[\d,]*(?:\.\d+)?(?:\s?(?:m|million))?\b'
          r'|\b\d[\d,]*\.\d\d\sGBP\b|\b\d+(?:\.\d+)?\smillion\spounds\b',
 'DATE':  rf'\b\d{1,2}\s{MON}\s\d{4}\b|\b\d{2}/\d{2}/\d{4}\b'
          rf'|\b\d{4}-\d{2}-\d{2}\b|\b{MON}\s\d{1,2},\s\d{4}\b',
 'ACCT':  r'\bGB\d{2}\s[A-Z]{4}(?:\s\d{4}){3}\s\d{2}\b'
          r'|\b\d{2}-\d{2}-\d{2}\s\d{8}\b',
}""", 'xs') + ent_html(EX['narrative'][1]['rule']['tokens'], EX['narrative'][1]['rule']['pred'], 13) +
          '</div><div class="colstack">' + prf_table(R, ['MONEY', 'DATE', 'ACCT']) +
          '<div class="callout good sm"><b>Cheap, fast and explainable.</b> A rule runs in microseconds and a failed match can be read off the pattern.</div>'
          '<div class="callout warn sm"><b>A rule knows only the formats it was written for.</b> Each miss here is a format the author did not list, and each new format is another edit.</div>'
          '</div></div>'))

# ================================================================ example continued 1: general model
G = NM['general']; GG = NM['general_genre']
items = [(GNAME[g], round(GG[g]['f1'], 3), RED if g == 'narrative' else BLUE, f"{GG[g]['n']} entities") for g in GENRES]
e = next(x for x in EX['narrative'] if any(s_[0] in ('PER', 'ORG', 'LOC') for s_ in x['gen']['gold']))
add(slide('Entities', 'Entities &#183; Example', 'Example continued 1',
          f"A general NER model trained on news, used as it comes on people, companies and places: F1 {G['all']['f1']:.3f} overall, {GG['narrative']['f1']:.3f} on payment narratives.",
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          fig(hbars(items, w=620, label_w=170, bar_h=28, gap=12, maxv=1.0, valfmt=lambda v: f'{v:.3f}', label='General model F1 by kind of text'), 'F1 on PER, ORG and LOC, by kind of text') +
          '<div class="outlbl">a payment narrative: label, then the general model</div>' +
          f'<div class="card sm">{ent_html(e["gen"]["tokens"], [s_ for s_ in e["gen"]["gold"] if s_[0] in ("PER", "ORG", "LOC")], 13)}'
          f'{ent_html(e["gen"]["tokens"], e["gen"]["pred"], 13)}</div>' +
          '</div><div class="colstack">' + prf_table(G, ['PER', 'ORG', 'LOC']) +
          '<div class="callout warn sm"><b>Capitals carry the signal in news.</b> A model that learned "capitalised word in mid-sentence is a name" has nothing to go on when every word is in capitals.</div>'
          '<div class="callout sm"><b>Its label set is fixed.</b> It has no tag for an amount or an account number, and cannot be asked for one.</div>'
          '</div></div>'))

# ================================================================ mechanism: aligning labels with word pieces (build)
AL = NE['align']
s = ''; x = 8
words_x = []
# words row
wx = 8
for t, g in zip(toks, tags):
    w = max(52, len(t) * 9 + 18)
    words_x.append((wx, w)); wx += w
scale = min(1.0, 1104 / wx)
pi = 0; pieces_per_word = []
tmp = []
for p, l in AL:
    if p.startswith('##') and tmp:
        tmp[-1].append((p, l))
    else:
        tmp.append([(p, l)])
row1 = row2 = row3 = ''
x = 8
for (t, g), grp in zip(zip(toks, tags), tmp):
    ty = g[2:] if g != 'O' else None
    pw = [max(34, len(p) * 8.4 + 14) for p, _ in grp]
    wtot = sum(pw)
    row1 += Rect(x, 6, wtot - 4, 28, '#FFFFFF', LINE, 1) + T(x + (wtot - 4) / 2, 25, t, 12, INK, 'middle', MONO, 600)
    row1 += Rect(x, 38, wtot - 4, 22, ECOL[ty] if ty else '#EEF2F4') + T(x + (wtot - 4) / 2, 53.5, g, 10.5, '#FFFFFF' if ty else SOFT, 'middle', MONO, 700)
    px = x
    for (p, l), w in zip(grp, pw):
        row2 += Rect(px, 84, w - 4, 28, '#F3F7F9', LINE, 1) + T(px + (w - 4) / 2, 103, p, 11.5, INK, 'middle', MONO, 500)
        ign = l == 'ignored'
        lt = l[2:] if (not ign and l != 'O') else None
        row3 += Rect(px, 116, w - 4, 22, '#FFFFFF' if ign else (ECOL[lt] if lt else '#EEF2F4'), LINE if ign else 'none', 1, 'stroke-dasharray="3 2"' if ign else '')
        row3 += T(px + (w - 4) / 2, 131.5, '–' if ign else l, 10.5, SOFT if (ign or not lt) else '#FFFFFF', 'middle', MONO, 700)
        px += w
    x += wtot
tot_w = x + 8
n_pieces = len(AL); n_ign = sum(l == 'ignored' for _, l in AL)
s = row1 + G_(1, row2 + T(8, 78, 'word pieces the model reads', 10.5, SOFT, 'start', MONO)) + G_(2, row3)
add(slide('Entities', 'Entities &#183; Mechanism', 'Mechanism',
          f"Labels are per word and the model reads word pieces. This sentence has {len(toks)} words and {n_pieces} pieces.",
          fig(svg(max(1120, tot_w), 146, s, 'Aligning word labels with word pieces'), 'labels on words, then on pieces') +
          '<div class="cols3">'
          '<div class="card sm"><h3>The labels</h3><p>An annotator tags whole words. A company name of three words gets B-ORG, I-ORG, I-ORG.</p></div>'
          '<div class="card sm" data-at="1"><h3>1. Split</h3><p>The tokenizer breaks rare words, names above all, into pieces. The tagger outputs one tag per piece.</p></div>'
          f'<div class="card sm g" data-at="2"><h3>2. Label the first piece</h3><p>Each word&#8217;s label goes on its first piece. The other {n_ign} pieces are left out of the loss, and at prediction time only first pieces are read.</p></div>'
          '</div>' + bld(2), 'build'))

# ================================================================ example continued 2: fine-tuned
F = NM['ft']; CU = NM['curve']; FG = NM['ft_genre']
ns = sorted(CU, key=int)
import math as _m
ax = Ax(520, 250, (_m.log10(int(ns[0])) - .1, _m.log10(int(ns[-1])) + .1), (min(0.5, min(CU.values()) - .05), 1.0), pad=(44, 12, 14, 32))
yt = [v / 10 for v in range(int(ax.yr[0] * 10) + 1, 11)]
s = ax.frame(yticks=yt, xticks=[(_m.log10(int(n)), f'{int(n):,}') for n in ns], yfmt=lambda v: f'{v:.1f}', ylab='F1, all six types', xlab='training sentences')
s += ax.line([_m.log10(int(n)) for n in ns], [CU[n] for n in ns], GREEN, 2.6)
for n in ns:
    s += ax.dot(_m.log10(int(n)), CU[n], GREEN, 4) + T(ax.X(_m.log10(int(n))), ax.Y(CU[n]) + 18, f'{CU[n]:.3f}', 11, INK, 'middle', MONO, 700)
add(slide('Entities', 'Entities &#183; Example', 'Example continued 2',
          f"DistilBERT fine-tuned on the {comma(NE['n_train'])} training sentences tags all six types at F1 {F['all']['f1']:.3f}, in {NM['t_ft']:.0f} seconds on a {DEVN}.",
          '<div class="cols"><div class="colstack">' + prf_table(F, TYPES) +
          f'<div class="callout sm"><b>Payment narratives: {FG["narrative"]["f1"]:.3f}</b>, against {GG["narrative"]["f1"]:.3f} for the general model. The tagger has now seen capitals without grammar.</div>'
          '</div><div class="colstack">' + fig(svg(520, 250, s, 'F1 against training sentences'), 'F1 on the test set, by number of training sentences') +
          f'<div class="callout sm"><b>{int(ns[0])} labelled sentences already give {CU[ns[0]]:.3f}.</b> An afternoon of annotation moves a tagger a long way; the last points cost the most labels.</div>'
          '</div></div>'))

# ================================================================ drawbacks: what is still missed
rows = [[GNAME[g], f'<span class="tag" style="background:{ECOL[t]}">{t}</span>', esc(ent), esc(pred)] for g, t, ent, pred in NE['miss'] if len(ent) < 34 and len(pred) < 44][:8]
top = NE['miss_top'][0]
add(slide('Entities', 'Entities &#183; Drawbacks', 'Drawbacks',
          f"The fine-tuned tagger misses {NE['n_miss']} of {comma(sum(NE['n_ent'].values()))} test entities. The largest group is {top[2]} {top[1]} entities in {GNAME[top[0]]}s.",
          '<div class="cols"><div class="colstack" style="flex:1.5">' + table(['text', 'type', 'entity in the label', 'what the tagger returned'], rows, 'd sm free') +
          '</div><div class="colstack">'
          '<div class="callout warn"><b>These scores are from simulated text.</b> Real narratives are messier: truncated names, reference codes, typing errors. Expect lower numbers and the same ordering of methods.</div>'
          '<div class="callout"><b>Company or person.</b> A firm named after its founders reads as two people. Context settles it in a sentence and is absent in a narrative.</div>'
          '<div class="callout sm"><b>Labels are personal data.</b> Sentences tagged with real names and account numbers need the same controls as the source system.</div>'
          '</div></div>'))

# ================================================================ verdict: three approaches by type
s = ''; gw = 172; x0 = 60; H = 230
for v in (0, .25, .5, .75, 1):
    y = 20 + (1 - v) * (H - 60)
    s += L(x0, y, x0 + gw * 6, y, HAIR, 1) + T(x0 - 6, y + 4, f'{v:.2f}', 10, SOFT, 'end', MONO)
for i, ty in enumerate(TYPES):
    gx = x0 + i * gw + 16
    vals = [(R.get(ty, {}).get('f1'), AMBER, 'rules'), (G.get(ty, {}).get('f1'), SLATE, 'general'), (F[ty]['f1'], GREEN, 'fine-tuned')]
    for j, (v, c, nm) in enumerate(vals):
        bx_ = gx + j * 46
        if v is None:
            s += T(bx_ + 20, H - 46, 'n/a', 10, SOFT, 'middle', MONO)
            continue
        h = v * (H - 60)
        s += Rect(bx_, 20 + (H - 60) - h, 40, h, c) + T(bx_ + 20, 20 + (H - 60) - h - 5, f'{v:.2f}', 10.5, INK, 'middle', MONO, 700)
    s += T(gx + 66, H - 20, ty, 12, ECOL[ty], 'middle', MONO, 700)
leg = (f'<span class="tag" style="background:{AMBER}">&#160;</span>rules &#160; <span class="tag" style="background:{SLATE}">&#160;</span>general model, as it comes &#160; '
       f'<span class="tag" style="background:{GREEN}">&#160;</span>fine-tuned tagger')
add(slide('Entities', 'Entities &#183; Verdict', 'Verdict',
          'F1 by entity type for the three approaches. Each approach is absent where it has nothing to offer.',
          fig(svg(1120, H, s, 'F1 by entity type and approach'), 'F1 on the test set', leg) +
          '<div class="cols nat"><div class="callout"><b>Shaped entities: start with rules.</b> They need no labels and their failures can be read. Keep them as a check on the model even after a tagger exists.</div>'
          '<div class="callout"><b>Names: label the bank&#8217;s own text.</b> A few hundred sentences from the real channel did more here than a model trained on a large corpus of the wrong kind of text.</div></div>'))
