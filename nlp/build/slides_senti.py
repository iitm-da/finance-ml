"""NLP session, part 2: financial sentiment."""
from common import *

B = []
add = B.append
M = SE['m']; LAB = SE['lab']; SH = SE['share']
LCOL = [RED, GREEN, SOFT]
KEYS = ['always neutral', 'general tweet model, as is', 'FinBERT, as is', 'TF-IDF + logistic regression',
        'embeddings + logistic regression', 'DistilBERT, fine-tuned']
MCOL = {'always neutral': GREY, 'general tweet model, as is': AMBER, 'FinBERT, as is': SLATE, 'TF-IDF + logistic regression': BLUE,
        'embeddings + logistic regression': PURP, 'DistilBERT, fine-tuned': GREEN}
FT = M['DistilBERT, fine-tuned']; TF = M['TF-IDF + logistic regression']; GE = M['general tweet model, as is']
FB = M['FinBERT, as is']; EM = M['embeddings + logistic regression']
DEVN = 'GPU' if SE['dev'] == 'cuda' else 'CPU'

def tagl(k):
    return f'<span class="tag {["r", "g", "s"][k]}">{LAB[k]}</span>'

def rowtable(rows, head, widths=None):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="d sm free">{"<tr>" + h + "</tr>"}{b}</table>'

# ================================================================ data
rows = [[esc(t), tagl(k)] for t, k in SE['head'] if len(t) < 100][:10]
s = ''
for k in range(3):
    y = 8 + k * 40
    s += T(84, y + 19, LAB[k], 13, INK, 'end', MONO, 600) + Rect(94, y, 300 * SH[k] / max(SH), 28, LCOL[k])
    s += T(100 + 300 * SH[k] / max(SH), y + 19, pct(SH[k]), 13, INK, 'start', MONO, 700)
add(slide('Sentiment', 'Sentiment &#183; Data', 'Finance headlines with three labels',
          f"{comma(SE['n_train'])} training rows and {comma(SE['n_test'])} test rows from a public dataset. Bearish means bad news for the stock named, bullish means good news.",
          '<div class="cols"><div class="colstack" style="flex:1.7">' + rowtable(rows, ['headline', 'label']) +
          '</div><div class="colstack">' + fig(svg(460, 126, s, 'Share of each label in the test set'), 'share of the test rows') +
          f'<div class="callout warn"><b>Accuracy flatters.</b> A model that answers "neutral" to everything is right on {pct(SH[2])} of rows. Its macro-F1, the average of the three per-class F1 scores, is {M["always neutral"]["f1"]:.3f}.</div>'
          '<div class="callout sm">Every score from here on is macro-F1 on the same test rows.</div>'
          '</div></div>'))

# ================================================================ concept: three ways to use a pretrained model
def stack(x, title, sub, body_trained, head, head_trained, labels):
    s = T(x + 130, 18, title, 14, INK, 'middle', BODY, 700) + T(x + 130, 36, sub, 11, SOFT, 'middle', MONO)
    s += Rect(x + 40, 50, 180, 34, RED if head_trained else GREY) + T(x + 130, 72, head, 12, '#FFFFFF', 'middle', MONO, 700)
    s += Rect(x + 40, 90, 180, 96, RED if body_trained else GREY) + T(x + 130, 134, 'transformer', 12.5, '#FFFFFF', 'middle', MONO, 700)
    s += T(x + 130, 152, '66 million weights', 10.5, '#FFFFFF', 'middle', MONO)
    s += T(x + 130, 208, labels, 11.5, INK, 'middle', MONO, 600)
    return s
s = stack(0, 'As it comes', 'someone else trained it', False, 'their label set', False, 'labels needed: 0')
s += stack(290, 'Frozen embeddings', 'train a linear model on top', False, 'logistic regression', True, 'labels needed: hundreds')
s += stack(580, 'Fine-tuning', 'update every weight', True, 'new head', True, 'labels needed: thousands')
s += Rect(880, 60, 16, 16, RED) + T(902, 73, 'trained on the bank&#8217;s labels'.replace('&#8217;', '’'), 11.5, INK, 'start', MONO)
s += Rect(880, 86, 16, 16, GREY) + T(902, 99, 'left as downloaded', 11.5, INK, 'start', MONO)
add(slide('Sentiment', 'Sentiment &#183; Concept', 'Concept',
          'A pretrained encoder can be used three ways. They differ in which weights the bank&#8217;s own labels are allowed to change.',
          fig(svg(1120, 218, s, 'Three ways to use a pretrained model'), 'which weights are trained') +
          '<div class="cols3">'
          '<div class="card sm o"><h3>As it comes</h3><p>Download a model that already has a sentiment head. Nothing is trained. Its labels and its idea of "negative" come from its own training data.</p></div>'
          '<div class="card sm"><h3>Frozen embeddings</h3><p>Run each text through the model once and keep the vector. Train a small linear model on the vectors. Seconds on a CPU once the vectors exist.</p></div>'
          '<div class="card sm g"><h3>Fine-tuning</h3><p>Put a new head on the model and train all the weights on the labelled rows for a few passes. Needs a GPU and the most labels.</p></div>'
          '</div>'))

# ================================================================ example: models as they come
def bars(keys, w=700):
    return hbars([(k.replace(', as is', '').replace('logistic regression', 'LR'), round(M[k]['f1'], 3), MCOL[k], f"accuracy {M[k]['acc']:.3f}") for k in keys],
                 w=w, label_w=250, bar_h=30, gap=12, maxv=1.0, label='Macro-F1 by model', valfmt=lambda v: f'{v:.3f}')
k4 = KEYS[:4]
best_asis = max(GE['f1'], FB['f1'])
add(slide('Sentiment', 'Sentiment &#183; Example', 'Example',
          f"Two models used as they come, against a bag-of-words baseline trained on the {comma(SE['n_train'])} labelled rows. "
          f"The baseline scores {TF['f1']:.3f}; the better of the two downloads scores {best_asis:.3f}.",
          '<div class="cols"><div class="colstack" style="flex:1.35">' + fig(bars(k4), 'macro-F1 on the test rows') +
          '</div><div class="colstack">'
          '<div class="callout"><b>General tweet model.</b> RoBERTa trained on 124 million general tweets. Its "negative" is an unhappy author, which a profit warning written in neutral words is not.</div>'
          f'<div class="callout"><b>FinBERT.</b> BERT trained on financial news sentences. Right domain, different text: full sentences from news wires, where these rows are clipped headlines with tickers.</div>'
          f'<div class="callout {"warn" if TF["f1"] > best_asis else "good"}"><b>TF-IDF and logistic regression</b> has no idea what a word means, and has read {comma(SE["n_train"])} rows of exactly this text. Here that is worth {TF["f1"] - best_asis:+.3f} macro-F1 over the better download.</div>'
          '</div></div>'))

# ================================================================ example continued 1: rows the general model gets wrong
rows = [[esc(t), f'<span class="tag {["r", "g", "s"][LAB.index(tr)]}">{tr}</span>', f'<span style="font-family:var(--mono);font-size:11.5px">{g}</span>']
        for t, tr, g in SE['ex_fix'] if len(t) < 88][:9]
add(slide('Sentiment', 'Sentiment &#183; Example', 'Example continued 1',
          'Rows the general model gets wrong. The words that carry the label are market vocabulary: cut, downgrade, raises target, beats.',
          '<div class="cols"><div class="colstack" style="flex:2.1">' + rowtable(rows, ['headline', 'label', 'general model said']) +
          '</div><div class="colstack">'
          '<div class="callout"><b>Analyst actions.</b> "Cuts to Neutral" holds no emotional word. To an investor it is a downgrade.</div>'
          '<div class="callout"><b>Direction words.</b> "Falls", "slides" and "jumps" describe a price. A general model learned them from text that was not about prices.</div>'
          '<div class="callout sm">These rows are from the test set, picked where the fine-tuned model is right and the general model is wrong.</div>'
          '</div></div>'))

# ================================================================ example continued 2: all models
add(slide('Sentiment', 'Sentiment &#183; Example', 'Example continued 2',
          f"The two methods that train on the labelled rows. Fine-tuning DistilBERT reaches {FT['f1']:.3f} macro-F1 in {SE['t_ft']:.0f} seconds on a {DEVN}.",
          '<div class="cols"><div class="colstack" style="flex:1.35">' + fig(bars(KEYS), 'macro-F1 on the test rows') +
          '</div><div class="colstack">' +
          f'<div class="callout"><b>Embeddings and a linear model: {EM["f1"]:.3f}.</b> One pass of a 33-million-weight embedding model over all rows took {SE["t_embed"]:.0f} seconds. The vectors can be reused for search and clustering.</div>'
          f'<div class="callout good"><b>Fine-tuning: {FT["f1"]:.3f}.</b> Three passes over the training rows, every weight updated. {FT["f1"] - TF["f1"]:+.3f} over the bag-of-words baseline and {FT["f1"] - best_asis:+.3f} over the better download.</div>'
          + (f'<div class="callout warn sm"><b>Frozen vectors lose to word counts here</b> ({EM["f1"]:.3f} against {TF["f1"]:.3f}). A general-purpose embedding has no dimension for "downgrade".</div>' if EM['f1'] < TF['f1'] else
             f'<div class="callout sm">Frozen vectors beat word counts by {EM["f1"] - TF["f1"]:+.3f} with no GPU training.</div>') +
          '</div></div>'))

# ================================================================ test: confusion matrix
cm = FT['cm']; tot = [sum(r) for r in cm]
s = ''; cell = 86; x0, y0 = 150, 50
for j in range(3):
    s += T(x0 + j * cell + cell / 2, y0 - 10, LAB[j], 12, LCOL[j], 'middle', MONO, 700)
s += T(x0 + 1.5 * cell, 14, 'model says', 11, SOFT, 'middle', MONO)
for i in range(3):
    s += T(x0 - 10, y0 + i * cell + cell / 2 + 4, LAB[i], 12, LCOL[i], 'end', MONO, 700)
    s += T(x0 - 10, y0 + i * cell + cell / 2 + 20, f'{tot[i]} rows', 10, SOFT, 'end', MONO)
    for j in range(3):
        v = cm[i][j]; sh = v / tot[i]
        s += Rect(x0 + j * cell, y0 + i * cell, cell - 3, cell - 3, f'rgba(14,122,85,{sh:.2f})' if i == j else f'rgba(176,47,59,{min(1, sh * 2.5):.2f})')
        s += T(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 2, str(v), 17, '#FFFFFF' if (sh > .5 and i == j) else INK, 'middle', MONO, 700)
        s += T(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 20, pct(sh, 0), 10.5, '#FFFFFF' if (sh > .5 and i == j) else SOFT, 'middle', MONO)
s += T(20, y0 + 1.5 * cell, 'label', 11, SOFT, 'start', MONO)
conf = max(((cm[i][j], i, j) for i in range(3) for j in range(3) if i != j))
f1c = FT['f1c']
add(slide('Sentiment', 'Sentiment &#183; Test', 'Test',
          f"The fine-tuned model on the {comma(SE['n_test'])} test rows. The largest error cell is {conf[0]} {LAB[conf[1]]} rows read as {LAB[conf[2]]}.",
          '<div class="cols"><div class="colstack" style="flex:1.1">' + fig(svg(430, 316, s, 'Confusion matrix of the fine-tuned model'), 'rows: label, columns: prediction') +
          '</div><div class="colstack">' + '<div class="kv k3">' + ''.join(tile(f'F1, {LAB[k]}', f'{f1c[k]:.3f}', ['bad', 'ok', ''][k]) for k in range(3)) + '</div>' +
          f'<div class="callout"><b>Read the rows.</b> Of {tot[0]} bearish rows the model finds {cm[0][0]}. Of {tot[1]} bullish rows it finds {cm[1][1]}. Neutral, the class with the most training rows, takes {cm[0][2]} of the {tot[0] - cm[0][0]} bearish misses and {cm[1][2]} of the {tot[1] - cm[1][1]} bullish misses.</div>'
          '<div class="callout"><b>Which error costs more depends on the use.</b> A news-alert feed can live with a missed bearish headline. A model that feeds a trading signal treats bearish read as bullish as the expensive cell.</div>'
          '</div></div>'))

# ================================================================ learning curve
SZ = SE['sizes']; CU = SE['curve']
import math as _m
ax = Ax(680, 330, (_m.log10(SZ[0]) - .08, _m.log10(SZ[-1]) + .08), (0.3, 0.95), pad=(46, 14, 14, 34))
s = ax.frame(yticks=[.4, .5, .6, .7, .8, .9], xticks=[(_m.log10(n), f'{n:,}') for n in SZ], yfmt=lambda v: f'{v:.1f}', ylab='macro-F1', xlab='labelled rows used for training')
for k in ('TF-IDF + logistic regression', 'embeddings + logistic regression', 'DistilBERT, fine-tuned'):
    ys = [CU[k][str(n)] for n in SZ]
    s += ax.line([_m.log10(n) for n in SZ], ys, MCOL[k], 2.6)
    for n, v in zip(SZ, ys):
        s += ax.dot(_m.log10(n), v, MCOL[k], 4)
    s += T(ax.X(_m.log10(SZ[-1])) - 8, ax.Y(ys[-1]) + (-10 if k.startswith('Distil') else 16), k.replace('logistic regression', 'LR'), 11, MCOL[k], 'end', MONO, 700)
for v, lab in ((GE['f1'], 'general model, as is'), (FB['f1'], 'FinBERT, as is')):
    s += L(ax.l, ax.Y(v), ax.l + ax.pw, ax.Y(v), SOFT, 1, 'stroke-dasharray="4 3"') + T(ax.l + 8, ax.Y(v) - 5, f'{lab} {v:.3f}', 10.5, SOFT, 'start', MONO)
c100 = {k: CU[k][str(SZ[0])] for k in CU}; b100 = max(c100, key=c100.get)
cross = next((n for n in SZ if CU['DistilBERT, fine-tuned'][str(n)] > max(GE['f1'], FB['f1'])), None)
crossT = next((n for n in SZ if CU['DistilBERT, fine-tuned'][str(n)] > CU['TF-IDF + logistic regression'][str(n)] and CU['DistilBERT, fine-tuned'][str(n)] > CU['embeddings + logistic regression'][str(n)]), None)
add(slide('Sentiment', 'Sentiment &#183; Example', 'Example continued 3',
          f"The same three methods trained on fewer labels. With {SZ[0]} labelled rows the best is {b100.split(',')[0].split(' +')[0]} at {c100[b100]:.3f}"
          + (f"; fine-tuning passes the downloaded models at {cross:,} rows." if cross else '.'),
          '<div class="cols"><div class="colstack" style="flex:1.5">' + fig(svg(680, 330, s, 'Macro-F1 against number of labelled rows'), 'macro-F1 on the full test set, by training size (log scale)') +
          '</div><div class="colstack">' +
          table(['labels'] + ['TF-IDF', 'embed.', 'fine-tune'],
                [[f'{n:,}'] + [f"{CU[k][str(n)]:.3f}" for k in CU] for n in SZ], 'd sm free') +
          '<div class="callout sm"><b>Labels are the budget.</b> Each row here was read and labelled by a person. The curve says what the next thousand labels buy.</div>'
          + (f'<div class="callout sm"><b>Fine-tuning leads from {crossT:,} rows.</b> Below that, a linear model on fixed features is the safer choice.</div>' if crossT and crossT > SZ[0] else '') +
          '</div></div>'))

# ================================================================ widget: row browser
short = {'TF-IDF + logistic regression': 'TF-IDF', 'general tweet model, as is': 'general', 'FinBERT, as is': 'FinBERT',
         'embeddings + logistic regression': 'embed.', 'DistilBERT, fine-tuned': 'fine-tuned'}
models = [k for k in KEYS[1:]]
data = {'rows': [{'t': r['t'], 'y': r['y'], 'p': r['p']} for r in SE['browse']], 'models': models, 'short': [short[m] for m in models],
        'lab': LAB, 'ft': 'DistilBERT, fine-tuned', 'gen': 'general tweet model, as is'}
add(slide('Sentiment', 'Sentiment &#183; Example', 'Test rows, model by model',
          '400 test rows drawn at random. A coloured cell is a wrong prediction; the filters keep the rows where a given model fails.',
          '<div class="wctl"><div class="seg" id="sb-filter"><button data-v="all" class="on">all rows</button><button data-v="gen">general wrong, fine-tuned right</button>'
          '<button data-v="ft">fine-tuned wrong</button><button data-v="every">every model wrong</button></div>'
          f'<button class="wbtn" id="sb-next">next page</button><div class="out" id="sb-out" style="min-width:330px"></div>{JSB}</div>'
          '<table class="d sm free" id="sb-body"></table>' + jdata('sb-data', data)))

# ================================================================ drawbacks
rows = [[esc(t), f'<span class="tag {["r", "g", "s"][LAB.index(tr)]}">{tr}</span>', f'<span style="font-family:var(--mono);font-size:11.5px">{g}</span>']
        for t, tr, g in SE['ex_wrong'] if len(t) < 86][:6]
add(slide('Sentiment', 'Sentiment &#183; Drawbacks', 'Drawbacks',
          f"The fine-tuned model is still wrong on {pct(1 - FT['acc'])} of test rows. Six of them, with the label and what the model said.",
          '<div class="cols"><div class="colstack" style="flex:1.9">' + rowtable(rows, ['headline', 'label', 'fine-tuned said']) +
          '</div><div class="colstack">'
          '<div class="callout warn"><b>One label per headline.</b> "Downgraded at one broker, top pick at another" names two views of one stock. A single label cannot hold both.</div>'
          '<div class="callout"><b>Labels are judgements.</b> Whether a routine price-target change is neutral or bullish is a choice the annotators made. A model trained on their labels inherits it.</div>'
          '<div class="callout"><b>Vocabulary moves.</b> The training rows come from one period. New tickers, new products and new slang arrive after it, and the score on last year&#8217;s test set does not show that.</div>'
          '</div></div>'))
