"""NLP session, part 1: cover, plan, tokens and attention."""
from common import *

A = []
add = A.append
TOK = AT['tokens']; W = AT['weights']; SC = AT['scores']; II, IB = AT['i_its'], AT['i_bank']
LY, HD = AT['layer'] + 1, AT['head'] + 1
FT = SE['m']['DistilBERT, fine-tuned']; GENM = SE['m']['general tweet model, as is']

# ================================================================ cover
add(f'''
<section class="slide cover" data-sec="Cover">
  <div class="coverpat">
    <div><span class="h-dim">"its" &#8594; "bank"   attention weight </span><span class="h-val">{W[IB]:.3f}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">sentiment, general model   macro-F1 </span><span class="h-bad">{GENM['f1']:.3f}</span></div>
    <div><span class="h-dim">sentiment, fine-tuned      macro-F1 </span><span class="h-val">{FT['f1']:.3f}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">NER, general model, 3 types  F1 </span><span class="h-bad">{NE['m']['general']['all']['f1']:.3f}</span></div>
    <div><span class="h-dim">NER, fine-tuned, 6 types     F1 </span><span class="h-val">{NE['m']['ft']['all']['f1']:.3f}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">loan fields, rules      </span><span class="h-bad">{pct(sum(IE['m']['rules'].values()) / 6)}</span></div>
    <div><span class="h-dim">loan fields, QA model   </span><span class="h-val">{pct(sum(IE['m']['qa'].values()) / 6)}</span></div>
  </div>
  <div class="kicker">Machine Learning for Finance</div>
  <h1>Advanced NLP<br/>for Finance</h1>
  <div class="pt">Attention, sentiment, entities, documents</div>
  <div class="rule"></div>
  <p class="sub">One attention head computed by hand, then three jobs a bank gives a language model: reading sentiment,
  finding names and amounts, and pulling fields out of documents.</p>
  <div class="by"><span>Instructor</span>Dr. Arun B Ayyar</div>
  <table class="meta">
    <tr><td>With</td><td>Dr. Nirav Bhatt</td></tr>
    <tr><td>Duration</td><td>90 minutes</td></tr>
    <tr><td>Data</td><td>{comma(SE['n_train'] + SE['n_test'])} finance headlines &#183; simulated sentences, pages, loan agreements</td></tr>
    <tr><td>Code</td><td>advanced_nlp_for_finance.ipynb</td></tr>
  </table>
</section>
''')

# ================================================================ plan
blocks = [('Attention', 25, BLUE, ['word pieces', 'one head by hand', 'what it costs']),
          ('Sentiment', 20, PURP, ['a model as it comes', 'embeddings', 'fine-tuning']),
          ('Named entities', 20, GREEN, ['rules', 'a general model', 'a fine-tuned tagger']),
          ('Documents', 25, RED, ['classify a page', 'extract six fields', 'route by confidence'])]
s = ''; x = 10; wid = 1100; t = 0
for name, mins, col, items in blocks:
    w = wid * mins / 90
    s += Rect(x, 30, w - 4, 54, col) + T(x + 10, 53, name, 13.5, '#FFFFFF', 'start', BODY, 700) + T(x + 10, 72, f'{mins} min', 11, '#FFFFFF', 'start', MONO)
    s += T(x, 20, f'{t}', 10.5, SOFT, 'start', MONO)
    for i, it in enumerate(items):
        s += T(x + 10, 108 + i * 19, it, 12, INK, 'start', MONO)
    x += w; t += mins
s += T(1106, 20, '90', 10.5, SOFT, 'end', MONO)
add(slide('Plan', 'Plan', 'Plan for the 90 minutes',
          'Four blocks. Every number on the slides comes from the notebook, which runs in a few minutes on a free Colab GPU.',
          fig(svg(1120, 172, s, 'Session plan'), 'minutes from the start') +
          '<div class="cols nat"><div class="callout"><b>One model family.</b> Every model today is a transformer encoder of the BERT family, between 33 and 125 million weights. The first block opens one up; the other three put it to work.</div>'
          '<div class="callout"><b>One question per block.</b> How far does a model get as it comes, and what do labelled examples from the bank&#8217;s own text add?</div></div>'
          '<div class="callout warn sm">The headlines are a public dataset. The sentences, pages and loan agreements are simulated by <code class="k">make_data.py</code>; no text comes from a real bank or a real company.</div>'))

# ================================================================ word pieces
s = ''; y = 8
for w_, pcs in AT['pieces'].items():
    s += T(150, y + 18, w_, 14, INK, 'end', MONO, 700)
    s += T(166, y + 18, '→', 14, SOFT, 'start', MONO)
    c, _ = chiprow(192, y, pcs, fill='#F3F7F9' if len(pcs) > 1 else '#F3FBF7', stroke=LINE)
    s += c
    s += T(640, y + 18, f'{len(pcs)} token' + ('s' if len(pcs) > 1 else ''), 11.5, SOFT, 'end', MONO)
    y += 36
add(slide('Tokens', 'Tokens &#183; Example', 'Example',
          f"The model reads word pieces. Its vocabulary holds {comma(AT['vocab'])} pieces, and a word that is missing from it is spelled out of smaller ones.",
          '<div class="cols"><div class="colstack">' +
          fig(svg(650, y + 4, s, 'Word pieces for eight finance terms'), 'DistilBERT tokenizer, uncased') +
          '</div><div class="colstack">' +
          code("""tok = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased')
tok.tokenize('remortgage')""", 'sm') +
          '<div class="callout"><b>No word is unknown.</b> A term the model has never seen still reaches it, in pieces. Whether the pieces mean anything together is learned from the bank&#8217;s text.</div>'
          f'<div class="callout"><b>Limits count pieces.</b> A 512-token limit is a limit on pieces. "remortgage" spends {len(AT["pieces"]["remortgage"])} of them.</div>'
          '<div class="callout warn"><b>Case is gone.</b> This tokenizer lowercases, so EBITDA and SONIA lose their capitals. A cased model keeps them, which matters for names.</div>'
          '</div></div>'))

# ================================================================ the model as a pipeline (build)
bx = [('text', 'one sentence', INK), ('word pieces', '12 tokens', SLATE), ('vectors', '768 numbers a token', BLUE),
      ('attention', 'tokens read each other', PURP), ('feed-forward', 'each token alone', AMBER), ('task head', 'label, tag or span', GREEN)]
s = ''; x = 10; bw = 160; gp = 28
for i, (a, b, c) in enumerate(bx):
    k = [0, 0, 1, 2, 3, 4][i]
    piece = box(x, 40, bw, 62, a, b, c)
    if i:
        piece += arrow(x - gp + 3, 71, x - 3, 71)
    s += G_(k, piece) if k else piece
    x += bw + gp
x3 = 10 + 3 * (bw + gp); x4 = x3 + bw + gp
s += G_(3, f'<rect x="{x3 - 10}" y="22" width="{2 * bw + gp + 20}" height="98" fill="none" stroke="{SOFT}" stroke-dasharray="5 4"/>' +
        T(x3 + bw + gp / 2, 16, 'one layer, repeated 6 times', 11.5, INK, 'middle', MONO, 700))
add(slide('Model', 'Model &#183; Concept', 'Concept',
          f"A transformer encoder turns a sentence into one vector per token. DistilBERT does it with {AT['n_params'] / 1e6:.0f} million weights in 6 layers.",
          fig(svg(1140, 128, s, 'Stages of a transformer encoder'), 'DistilBERT, left to right') +
          '<div class="cols4">'
          '<div class="card sm" data-at="1"><h3>Vectors</h3><p>Each piece is looked up in a table of 768 numbers. A second table adds the position, 1 to 512.</p></div>'
          '<div class="card sm" data-at="2"><h3>Attention</h3><p>Every token takes a weighted average of all the tokens. The weights are computed from the tokens themselves.</p></div>'
          '<div class="card sm o" data-at="3"><h3>Feed-forward</h3><p>A small network reshapes each token&#8217;s vector on its own. Attention and feed-forward make one layer.</p></div>'
          '<div class="card sm g" data-at="4"><h3>Task head</h3><p>One linear layer on top: a class for the sentence, a tag for each token, or the start and end of an answer.</p></div>'
          '</div>' + bld(4), 'build'))

# ================================================================ intuition: its -> bank
def arcs(tokens, w, src, wid=1120, y0=150, hi=None):
    n = len(tokens); cw = wid / n; s = ''
    for j, t in enumerate(tokens):
        cx = cw * j + cw / 2
        on = j == src
        s += Rect(cx - cw / 2 + 4, y0, cw - 8, 30, INK if on else '#FFFFFF', LINE, 1)
        s += T(cx, y0 + 20, t, 13, '#FFFFFF' if on else INK, 'middle', MONO, 700 if on or j == hi else 500)
        s += T(cx, y0 + 50, f'{w[j]:.3f}', 11.5, RED if j == hi else SOFT, 'middle', MONO, 700 if j == hi else 400)
    sx = cw * src + cw / 2
    for j in range(n):
        if j == src or w[j] < 0.005:
            continue
        cx = cw * j + cw / 2; h = min(130, 34 + abs(sx - cx) * 0.33)
        s += (f'<path d="M{sx:.1f},{y0} C{sx:.1f},{y0 - h:.1f} {cx:.1f},{y0 - h:.1f} {cx:.1f},{y0}" fill="none" stroke="{RED}" '
              f'stroke-width="{max(1, 16 * w[j]):.1f}" stroke-opacity="{0.35 + 0.65 * w[j]:.2f}" stroke-linecap="round"/>')
    return s
s = arcs(TOK, W, II, hi=IB)
add(slide('Attention', 'Attention &#183; Intuition', 'Intuition',
          f'To represent "its", the model has to find what "its" refers to. In layer {LY}, head {HD}, "its" puts {pct(W[IB])} of its weight on "bank".',
          fig(svg(1120, 208, s, 'Attention weights from the token its'), f'"{AT["sent"]}"  &#183;  attention from "its", layer {LY}, head {HD}',
              'The twelve weights add to 1. Line width is the weight.') +
          '<div class="cols nat"><div class="callout"><b>No labels for this.</b> The model was trained only to fill in hidden words. A head that links a pronoun to its noun appeared because it helps with that task.</div>'
          f'<div class="callout"><b>A weight is a share.</b> The new vector for "its" is a weighted average of all twelve token vectors, and here {pct(W[IB])} of it is the vector of "bank".</div></div>'))

# ================================================================ mechanism ladder (build)
n = len(TOK); cw = 86; x0 = 104
s = ''
for j, t in enumerate(TOK):
    cx = x0 + cw * j + cw / 2
    s += Rect(cx - cw / 2 + 3, 6, cw - 6, 26, INK if j == II else '#FFFFFF', LINE, 1)
    s += T(cx, 24, t, 12, '#FFFFFF' if j == II else INK, 'middle', MONO, 600)
r1 = T(x0 - 8, 62, 'score', 11.5, INK, 'end', MONO, 700) + T(x0 - 8, 76, 'q·k / 8', 10, SOFT, 'end', MONO)
r2 = T(x0 - 8, 178, 'weight', 11.5, INK, 'end', MONO, 700) + T(x0 - 8, 192, 'softmax', 10, SOFT, 'end', MONO)
r3 = ''
mx = max(SC); mn = min(SC)
for j in range(n):
    cx = x0 + cw * j + cw / 2
    r1 += T(cx, 68, f'{SC[j]:.2f}', 13, RED if j == IB else INK, 'middle', MONO, 700 if j == IB else 500)
    bh = 80 * W[j]
    r2 += Rect(cx - 20, 196 - bh, 40, max(bh, 1), RED if j == IB else GREY)
    r2 += T(cx, 196 - bh - 5, f'{W[j]:.3f}', 11.5, RED if j == IB else INK, 'middle', MONO, 700 if j == IB else 500)
r2 += L(x0, 196, x0 + cw * n, 196, LINE, 1)
top3 = sorted(range(n), key=lambda j: -W[j])[:3]
r3 += T(x0, 236, 'new vector for "its"  =  ' + '  +  '.join(f'{W[j]:.3f} × v({TOK[j]})' for j in top3) + '  +  …', 13.5, INK, 'start', MONO, 600)
s += G_(1, r1) + G_(2, r2) + G_(3, r3)
add(slide('Attention', 'Attention &#183; Mechanism', 'Mechanism',
          'One head, one token, three steps. The row is "its"; every number is read out of the model.',
          fig(svg(1140, 248, s, 'Scores, weights and the weighted sum for the token its'), f'layer {LY}, head {HD}') +
          '<div class="cols3">'
          f'<div class="card sm" data-at="1"><h3>1. Score</h3><p>Each token&#8217;s vector is multiplied by two learned matrices to give a query q and a key k of 64 numbers. The score is the dot product of the query of "its" with each key, divided by &#8730;64 = 8.</p></div>'
          f'<div class="card sm r" data-at="2"><h3>2. Softmax</h3><p>Exponentiate the scores and divide by their sum. A gap of {SC[IB] - sorted(SC)[-2]:.2f} in score between "bank" and the runner-up becomes a weight of {W[IB]:.3f} against {sorted(W)[-2]:.3f}.</p></div>'
          '<div class="card sm g" data-at="3"><h3>3. Weighted sum</h3><p>A third matrix gives each token a value v. The output for "its" is the sum of the values, each multiplied by its weight.</p></div>'
          '</div>' + bld(3), 'build'))

# ================================================================ mechanism continued 1: formula + code
formula = ('<div class="out" style="font-size:19px;text-align:center;padding:16px 10px">Attention(Q, K, V) &#160;=&#160; softmax( Q K<sup>T</sup> / &#8730;d ) &#183; V</div>')
add(slide('Attention', 'Attention &#183; Mechanism', 'Mechanism continued 1',
          'The three steps for all tokens at once are one line of matrix algebra. The notebook computes it by hand and gets the model&#8217;s own numbers.',
          '<div class="cols"><div class="colstack">' + formula +
          table(['symbol', 'what it is', 'shape here'],
                [['Q', 'queries: one row per token', '12 &#215; 64'], ['K', 'keys: one row per token', '12 &#215; 64'],
                 ['V', 'values: one row per token', '12 &#215; 64'], ['d', 'numbers in a query or key', '64'],
                 ['Q K<sup>T</sup>', 'every token scored against every token', '12 &#215; 12'],
                 ['softmax', 'each row turned into weights that add to 1', '12 &#215; 12']]) +
          '</div><div class="colstack">' +
          code(f"""att = enc.transformer.layer[{AT['layer']}].attention
h = hidden_states[{AT['layer']}][0]        # 12 x 768
Q = att.q_lin(h).view(-1, 12, 64)[:, {AT['head']}]
K = att.k_lin(h).view(-1, 12, 64)[:, {AT['head']}]
scores  = (Q @ K.T) / 64 ** 0.5
weights = torch.softmax(scores, dim=-1)
assert torch.allclose(weights, model_attention)""", 'sm') +
          '<div class="callout good sm">The assert passes: the hand computation and the model agree to four decimal places.</div>'
          '<div class="callout sm"><b>Why divide by &#8730;d.</b> Dot products of longer vectors are larger, and large scores push the softmax to put all the weight on one token. Dividing keeps the scores in a range where the weights can still be trained.</div>'
          '</div></div>'))

# ================================================================ widget: attention explorer
def _heads(layers, share):                 # heads putting over half their weight where share() looks, averaged over the widget sentences
    k = 0
    for L in layers:
        for H in range(12):
            k += sum(share(w['a'][L][H]) for w in AT['widget']) / len(AT['widget']) > 0.5
    return k
_adj = lambda a: sum(a[i][j] for i in range(len(a)) for j in (i - 1, i + 1) if 0 <= j < len(a)) / len(a)
_sep = lambda a: sum(r[-1] for r in a) / len(a)
N_ADJ, N_SEP = _heads((0, 1), _adj), _heads((2, 3, 4), _sep)
btn_s = ''.join(f'<button data-v="{i}"{" class=on" if i == 0 else ""}>{esc(w["tokens"][1])} {esc(w["tokens"][2])} &#8230;</button>' for i, w in enumerate(AT['widget']))
add(slide('Attention', 'Attention &#183; Mechanism', 'Mechanism continued 2',
          'A layer runs 12 heads side by side, each with its own Q, K and V, and the model stacks 6 layers: 72 heads, each attending to something different.',
          f'<div class="wctl"><div class="seg" id="ax-sent">{btn_s}</div>'
          f'<label class="rg">layer: <b id="ax-ll">{LY}</b><input type="range" id="ax-layer" min="1" max="6" step="1" value="{LY}"/></label>'
          f'<label class="rg">head: <b id="ax-hl">{HD}</b><input type="range" id="ax-head" min="1" max="12" step="1" value="{HD}"/></label>{JSB}</div>'
          '<div class="fig"><div class="cap">click a token to see where it looks</div>'
          '<svg id="ax-svg" viewBox="0 0 1120 300" width="100%" role="img" aria-label="Attention explorer"></svg></div>'
          '<div class="cols nat"><div class="out" id="ax-out" style="min-height:58px"></div>'
          f'<div class="callout sm">On these sentences, {N_ADJ} heads in layers 1 and 2 put over half their weight on the next or previous token. In layers 3 to 5, {N_SEP} of the 36 heads put over half on [SEP], where a head parks its weight when it has nothing to do.</div></div>' +
          jdata('ax-data', {'w': AT['widget'], 'layer': AT['layer'], 'head': AT['head'], 'src': II})))

# ================================================================ bank in two contexts
def heat(C, x0, y0, cell=44, title=''):
    s = T(x0 + 3 * cell, y0 - 26, title, 12, INK, 'middle', MONO, 700)
    for i in range(6):
        col = BLUE if AT['banks'][i][0] == 'money' else GREEN
        s += Rect(x0 - 16, y0 + i * cell + 2, 10, cell - 4, col) + Rect(x0 + i * cell + 2, y0 - 16, cell - 4, 10, col)
        for j in range(6):
            v = C[i][j]; sh = max(0.0, min(1.0, (v - 0.45) / 0.55))
            s += Rect(x0 + j * cell, y0 + i * cell, cell - 1, cell - 1, f'rgba(176,47,59,{sh:.2f})')
            s += T(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 4, f'{v:.2f}', 11, '#FFFFFF' if sh > .55 else INK, 'middle', MONO, 500)
    return s
s = heat(AT['C0'], 30, 44, title='input vectors') + heat(AT['C6'], 340, 44, title='after 6 layers')
CS = AT['cos']
sents = ''.join(f'<div style="display:flex;gap:8px;align-items:baseline;margin-bottom:5px"><span class="tag {"b" if k == "money" else "g"}">{i + 1}</span>'
                f'<span style="font-size:13.5px">{esc(t).replace("bank", "<b>bank</b>")}</span></div>' for i, (k, t) in enumerate(AT['banks']))
add(slide('Attention', 'Attention &#183; Example', 'Example',
          f'The word "bank" in six sentences. Going in, the six vectors are nearly the same ({CS["in_diff"]:.2f} across senses). After six layers of attention, the two senses have separated ({CS["out_diff"]:.2f}).',
          '<div class="cols"><div class="colstack" style="flex:1.25">' +
          fig(svg(620, 316, s, 'Cosine similarity between vectors for the word bank'), 'cosine similarity between the vectors for "bank"',
              'Blue rows are the money sense, green rows the river sense.') +
          '</div><div class="colstack">' + f'<div class="card">{sents}</div>' +
          '<div class="kv k2" style="grid-template-columns:1fr 1fr">' + tile('same sense, after 6 layers', f'{CS["out_same"]:.2f}', 'ok') + tile('different sense, after 6 layers', f'{CS["out_diff"]:.2f}', 'bad') + '</div>'
          '<div class="callout sm"><b>This is what a classifier reads.</b> Sentiment, entity tags and extracted spans are all computed from these context-dependent vectors.</div>'
          '</div></div>'))

# ================================================================ drawbacks: cost
CO = AT['cost']; ks = sorted(CO, key=int)
items = [(f'{k} tokens', CO[k], BLUE if int(k) < 512 else RED, f'{int(k) ** 2:,} cells') for k in ks]
add(slide('Attention', 'Attention &#183; Drawbacks', 'Drawbacks',
          f'Every token attends to every token, so a head holds n&#178; weights for n tokens. At 512 tokens that is {512 * 512:,} weights a head, {512 * 512 * 72 / 1e6:.1f} million across the 72 heads.',
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          fig(hbars(items, w=660, label_w=110, bar_h=30, gap=12, valfmt=lambda v: f'{v:.0f} ms', label='Time by sequence length'),
              f'one batch of 16 texts through DistilBERT, {"GPU" if AT["dev"] == "cuda" else "CPU"}',
              f'{int(ks[-1]) // int(ks[0])} times the tokens took {CO[ks[-1]] / CO[ks[0]]:.0f} times as long here; the attention cells grew {(int(ks[-1]) // int(ks[0])) ** 2} times.') +
          '</div><div class="colstack">'
          '<div class="callout warn"><b>A hard limit of 512 pieces.</b> The position table has 512 rows. A longer document is cut, or split into chunks that are scored one at a time.</div>'
          '<div class="callout"><b>A cut loses the end.</b> The governing-law clause of a contract sits on the last page. Truncation to the first 512 pieces never shows it to the model.</div>'
          '<div class="callout"><b>A weight is not an explanation.</b> A head that looks at "bank" shows where information came from. It does not show how the model used it.</div>'
          '</div></div>'))

# ================================================================ encoder vs decoder
def mask(x0, y0, tri, col, cell=24, n=7):
    s = ''
    for i in range(n):
        for j in range(n):
            on = (j <= i) if tri else True
            s += Rect(x0 + j * cell, y0 + i * cell, cell - 2, cell - 2, col if on else '#EEF2F4')
    return s
s = T(10, 16, 'encoder: every token sees every token', 12, INK, 'start', MONO, 700) + mask(10, 28, False, BLUE)
s += T(300, 16, 'decoder: a token sees only earlier ones', 12, INK, 'start', MONO, 700) + mask(300, 28, True, PURP)
s += T(10, 214, 'rows: the token being computed', 10.5, SOFT, 'start', MONO) + T(300, 214, 'columns: the tokens it may attend to', 10.5, SOFT, 'start', MONO)
add(slide('Model', 'Model &#183; Concept', 'Concept continued 1',
          'The same attention, with one change, gives the two families in use today. An encoder reads; a decoder writes.',
          '<div class="cols"><div class="colstack" style="flex:1.15">' + fig(svg(600, 222, s, 'Attention masks of an encoder and a decoder'), 'which weights are allowed') +
          '</div><div class="colstack">' +
          table(['', 'encoder (BERT family)', 'decoder (GPT family)'],
                [['trained to', 'fill in hidden words', 'predict the next word'],
                 ['output', 'a vector per token', 'text, one token at a time'],
                 ['size today', '0.03 to 0.4 billion weights', '0.5 to hundreds of billions'],
                 ['used for', 'classify, tag, extract, embed', 'answer, summarise, write'],
                 ['session', 'this one', 'the next one']]) +
          '<div class="callout sm"><b>Why encoders still matter.</b> A fixed label set, millions of rows a day and a latency budget in milliseconds describe most text jobs in a bank. An encoder of 66 million weights does them on a CPU.</div>'
          '</div></div>'))
