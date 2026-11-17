"""LLM session, part 4: prompting, LoRA and full fine-tuning; summary; references."""
from common import *

D = []
add = D.append
ENC = PF['enc']; RANK = PF['rank']; LL = PF['llm']; PP = PF['prompt']
HO, FU, L8 = ENC['head only'], ENC['full fine-tuning'], ENC['LoRA, rank 8']
LZ, LR_ = LL['zero-shot'], LL['LoRA, rank 8, 3,000 labels']
mbs = lambda b: f'{b / 1e6:.1f} MB' if b >= 1e5 else f'{b / 1e3:.0f} KB'

# ================================================================ concept: three ways to adapt
def col(x, title, sub, body, adapters, note):
    s = T(x + 150, 18, title, 14, INK, 'middle', BODY, 700) + T(x + 150, 36, sub, 11, SOFT, 'middle', MONO)
    s += Rect(x + 60, 52, 180, 120, RED if body else GREY) + T(x + 150, 108, 'model weights', 12.5, '#FFFFFF', 'middle', MONO, 700)
    s += T(x + 150, 126, 'trained' if body else 'frozen', 11, '#FFFFFF', 'middle', MONO)
    if adapters:
        for k in range(4):
            s += Rect(x + 246, 56 + k * 29, 22, 22, RED)
        s += T(x + 280, 110, 'adapters', 11, RED, 'start', MONO, 700) + T(x + 280, 124, 'trained', 10.5, RED, 'start', MONO)
    s += T(x + 150, 196, note, 11.5, INK, 'middle', MONO, 600)
    return s
s = col(0, 'Prompting', 'the task is in the prompt', False, False, 'labels: 0 to a few dozen')
s += col(370, 'LoRA', 'small matrices beside the weights', False, True, 'labels: hundreds to thousands')
s += col(760, 'Full fine-tuning', 'every weight moves', True, False, 'labels: thousands and more')
add(slide('Adaptation', 'Adaptation &#183; Concept', 'Concept',
          'Three ways to make a general model do the bank&#8217;s task. They differ in what is trained and in what has to be stored and served afterwards.',
          fig(svg(1140, 206, s, 'Prompting, LoRA and full fine-tuning'), 'red: trained on the bank&#8217;s data') +
          '<div class="cols3">'
          '<div class="card sm"><h3>Prompting</h3><p>Nothing is trained. One copy of the model serves every task. Behaviour is changed by editing text, and is limited to what the model can already do.</p></div>'
          '<div class="card sm r"><h3>LoRA</h3><p>The model is frozen and a small set of new weights is trained beside it. One base model in memory, one small adapter file per task.</p></div>'
          '<div class="card sm o"><h3>Full fine-tuning</h3><p>All weights are updated. The result is a complete second model, to be stored, versioned and served on its own.</p></div>'
          '</div>'))

# ================================================================ LoRA intuition (build)
d = 768
s = Rect(30, 30, 190, 190, GREY) + T(125, 120, 'W', 30, '#FFFFFF', 'middle', MONO, 700) + T(125, 146, f'{d} × {d}', 12, '#FFFFFF', 'middle', MONO)
s += T(125, 240, f'{d * d:,} weights, frozen', 12, INK, 'middle', MONO, 700)
p1 = T(262, 130, '+', 30, INK, 'middle', MONO, 700)
p1 += Rect(310, 30, 22, 190, RED) + T(321, 240, 'B', 15, RED, 'middle', MONO, 700) + T(321, 256, f'{d} × r', 11, SOFT, 'middle', MONO)
p1 += Rect(346, 30, 190, 22, RED) + T(441, 68, 'A', 15, RED, 'middle', MONO, 700) + T(441, 84, f'r × {d}', 11, SOFT, 'middle', MONO)
p2 = T(600, 60, 'r = 8', 15, INK, 'start', MONO, 700)
p2 += T(600, 88, f'B: {d} × 8 = {d * 8:,}', 13, INK, 'start', MONO) + T(600, 110, f'A: 8 × {d} = {d * 8:,}', 13, INK, 'start', MONO)
p2 += T(600, 138, f'{2 * d * 8:,} trained weights', 13, RED, 'start', MONO, 700) + T(600, 160, f'= {2 * d * 8 / (d * d):.1%} of W', 13, RED, 'start', MONO, 700)
p3 = T(860, 60, 'DistilBERT', 15, INK, 'start', MONO, 700)
p3 += T(860, 88, '6 layers × 2 matrices', 13, INK, 'start', MONO) + T(860, 110, f'= 12 × {2 * d * 8:,}', 13, INK, 'start', MONO)
p3 += T(860, 138, f'= {L8["lora"]:,} weights', 13, RED, 'start', MONO, 700) + T(860, 160, f'of {FU["total"] / 1e6:.1f} million', 13, RED, 'start', MONO, 700)
s += G_(1, p1) + G_(2, p2) + G_(3, p3)
add(slide('LoRA', 'LoRA &#183; Intuition', 'Intuition',
          'A weight matrix W is left alone. The change to it is written as the product of two thin matrices, B and A, and only those are trained.',
          fig(svg(1140, 266, s, 'A weight matrix with a low-rank update beside it'), 'one attention matrix of DistilBERT') +
          '<div class="cols3">'
          '<div class="card sm" data-at="1"><h3>1. Two thin matrices</h3><p>B has r columns and A has r rows. Their product BA has the shape of W, so it can be added to it.</p></div>'
          f'<div class="card sm r" data-at="2"><h3>2. The rank r</h3><p>r sets the size. At r = 8 the two matrices hold {2 * d * 8:,} weights where W holds {d * d:,}.</p></div>'
          '<div class="card sm g" data-at="3"><h3>3. Across the model</h3><p>The same pair is added beside the query and value matrices of every attention layer. Everything else keeps the weights it was downloaded with.</p></div>'
          '</div>' + bld(3), 'build'))

# ================================================================ LoRA mechanism
add(slide('LoRA', 'LoRA &#183; Mechanism', 'Mechanism',
          'The layer computes its usual output and adds the output of the two thin matrices, scaled.',
          '<div class="cols"><div class="colstack">' +
          '<div class="out" style="font-size:19px;text-align:center;padding:16px 10px">h &#160;=&#160; W x &#160;+&#160; (&#945; / r) &#183; B A x</div>' +
          table(['symbol', 'what it is'],
                [['W', 'the downloaded weight matrix, frozen'], ['x', 'the input to the layer'],
                 ['A', 'r &#215; d matrix, starts as small random numbers'], ['B', 'd &#215; r matrix, starts as zeros'],
                 ['r', 'the rank: 1 to 64 in practice'], ['&#945;', 'a fixed scale, often 2r']]) +
          '<div class="callout sm"><b>B starts at zero</b>, so BA is zero and the adapted model begins exactly as the downloaded one.</div>'
          '</div><div class="colstack">' +
          code("""from peft import LoraConfig, get_peft_model

cfg = LoraConfig(task_type='SEQ_CLS', r=8, lora_alpha=16,
                 lora_dropout=0.1,
                 target_modules=['q_lin', 'v_lin'])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()""", 'sm') +
          '<div class="callout sm"><b>After training.</b> Either keep the adapter as a separate file and load it over the base model, or add BA into W once and serve a model with no extra cost per request.</div>'
          '<div class="callout sm"><b>Why it works.</b> Adapting to one task moves the weights in few directions. A low-rank matrix can hold those directions.</div>'
          '</div></div>'))

# ================================================================ widget: rank
rk = sorted(RANK, key=int)
data = {'ranks': [int(r) for r in rk], 'f1': [RANK[r]['f1'] for r in rk], 'lora': [RANK[r]['lora'] for r in rk], 'full': FU['f1'], 'head': HO['f1'],
        'models': [{'name': 'DistilBERT (67 million weights)', 'd': 768, 'mats': 12, 'total': L8['total'] - L8['trainable'] + (L8['trainable'] - L8['lora'])},
                   {'name': 'a 7-billion-weight LLM (32 layers, width 4,096)', 'd': 4096, 'mats': 64, 'total': 7.2e9}]}
add(slide('LoRA', 'LoRA &#183; Example', 'The rank, from 1 to 32',
          f"The sentiment task with LoRA at six ranks. Macro-F1 moves from {RANK[rk[0]]['f1']:.3f} at rank {rk[0]} to {RANK[rk[-1]]['f1']:.3f} at rank {rk[-1]}; full fine-tuning scores {FU['f1']:.3f}.",
          '<div class="cols"><div class="colstack" style="flex:1.4"><div class="fig"><div class="cap">macro-F1 by rank, measured</div>'
          '<svg id="lr-svg" viewBox="0 0 680 290" width="100%" role="img" aria-label="Macro-F1 by LoRA rank"></svg></div>'
          f'<div class="wctl"><label class="rg">rank r: <b id="lr-rl">8</b><input type="range" id="lr-r" min="0" max="{len(rk) - 1}" step="1" value="{rk.index("8")}"/></label>{JSB}</div>'
          '</div><div class="colstack"><div class="out" id="lr-out" style="min-height:150px"></div>'
          '<div class="callout sm"><b>The adapter size is arithmetic.</b> Two matrices of r &#215; width for each adapted matrix, stored as 32-bit numbers. The second block applies the same sum to a 7-billion-weight model.</div>'
          '</div></div>' + jdata('lr-data', data)))

# ================================================================ encoder comparison
rows = [[n, f"<b>{v['f1']:.3f}</b>", f"{v['trainable']:,}", f"{v['trainable'] / v['total']:.2%}", mbs(v['bytes']), f"{v['secs']:.0f} s"]
        for n, v in (('head only', HO), ('LoRA, rank 8', L8), ('full fine-tuning', FU))]
items = [('head only', round(HO['f1'], 3), GREY, f"{HO['trainable']:,} weights trained"), ('LoRA, rank 8', round(L8['f1'], 3), RED, f"{L8['trainable']:,} weights trained"),
         ('full fine-tuning', round(FU['f1'], 3), GREEN, f"{FU['trainable']:,} weights trained")]
add(slide('LoRA', 'LoRA &#183; Example', 'Example',
          f"DistilBERT on the sentiment task with 3,000 labelled headlines, three ways. LoRA at rank 8 trains {L8['trainable'] / L8['total']:.1%} of the weights and scores {L8['f1']:.3f}; full fine-tuning scores {FU['f1']:.3f}.",
          fig(hbars(items, w=1000, label_w=180, bar_h=30, gap=12, maxv=1.0, valfmt=lambda v: f'{v:.3f}', label='Macro-F1 for three ways of training DistilBERT'), 'macro-F1 on 600 test headlines') +
          '<div class="cols"><div class="colstack" style="flex:1.5">' +
          table(['method', 'macro-F1', 'weights trained', 'share', 'file to store', 'training time'], rows, 'd sm free') +
          '</div><div class="colstack">'
          f'<div class="callout sm"><b>Of the {L8["trainable"]:,} trained weights, {L8["lora"]:,} are LoRA matrices</b> and the rest are the classification head, which every method trains.</div>'
          f'<div class="callout sm"><b>Head only: {HO["f1"]:.3f}.</b> Training the head on a frozen model is not enough; the attention layers have to move.</div>'
          '</div></div>'))

# ================================================================ LoRA on the LLM
loss = LR_['loss']
ax = Ax(520, 250, (0, len(loss) - 1), (0, max(loss) * 1.05), pad=(46, 14, 14, 32))
yt = [round(max(loss) * k / 4, 2) for k in range(5)]
s = ax.frame(yticks=yt, xticks=[(i, str(i * 10)) for i in range(0, len(loss), max(1, len(loss) // 5))], yfmt=lambda v: f'{v:.2f}', ylab='training loss', xlab='training step')
s += ax.line(range(len(loss)), loss, RED, 2.4)
items = [('zero-shot', round(LZ['f1'], 3), AMBER, 'no training'), ('LoRA, rank 8', round(LR_['f1'], 3), GREEN, f"{LR_['trainable']:,} weights, {LR_['secs']:.0f} s")]
add(slide('LoRA', 'LoRA &#183; Example', 'Example continued 1',
          f"The same adapter on a language model. A 0.5-billion-weight model scores {LZ['f1']:.3f} zero-shot and {LR_['f1']:.3f} after LoRA on 3,000 labelled headlines.",
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          fig(hbars(items, w=620, label_w=140, bar_h=34, gap=14, maxv=1.0, valfmt=lambda v: f'{v:.3f}', label='Macro-F1 before and after LoRA'), 'macro-F1 on 600 test headlines, Qwen2.5-0.5B') +
          '<div class="kv k3">' + tile('weights trained', f"{LR_['trainable'] / LR_['total']:.2%}", 'b') + tile('adapter file', mbs(LR_['bytes']), 'b') + tile('training time', f"{LR_['secs']:.0f} s", 'b') + '</div>'
          '</div><div class="colstack">' + fig(svg(520, 250, s, 'Training loss of the LoRA adapter'), 'loss on the label token, 10-step averages') +
          '<div class="callout sm"><b>Trained to write one word.</b> The prompt is the same as zero-shot. The loss is taken only on the label that follows it, so the adapter learns the answer and leaves the rest of the model&#8217;s behaviour alone.</div>'
          '</div></div>'))

# ================================================================ verdict: everything on one task
NS = NLP_SE['m']
allm = [('1.5B LLM, zero-shot', PP['zero-shot, 1.5B model'], AMBER, 'labels: 0'),
        ('1.5B LLM, 24 nearest examples', PP['24 nearest examples in the prompt'], AMBER, 'labels: 3,000 in a pool'),
        ('0.5B LLM, LoRA', LR_['f1'], RED, 'labels: 3,000'),
        ('DistilBERT, LoRA', L8['f1'], RED, 'labels: 3,000'),
        ('DistilBERT, full fine-tuning', FU['f1'], GREEN, 'labels: 3,000')]
best = max(allm, key=lambda r: r[1])
add(slide('Adaptation', 'Adaptation &#183; Verdict', 'Verdict',
          f"One task, five ways. The best score here is {best[1]:.3f}, from {best[0]}.",
          '<div class="cols"><div class="colstack" style="flex:1.4">' +
          fig(hbars([(a, round(b, 3), c, d_) for a, b, c, d_ in allm], w=700, label_w=260, bar_h=28, gap=11, maxv=1.0, valfmt=lambda v: f'{v:.3f}', label='Macro-F1 for five ways to do the sentiment task'),
              'macro-F1 on the same 600 test headlines') +
          '</div><div class="colstack">' +
          table(['if', 'then'],
                [['the answer is in a document', 'RAG; no training fixes missing facts'],
                 ['few labels, task changes often', 'prompt, with chosen examples'],
                 ['thousands of labels, fixed task', 'LoRA on a small model'],
                 ['one task at very high volume', 'fine-tune a small encoder']], 'd sm free') +
          '<div class="callout sm"><b>Labels outweigh size here.</b> The comparison is a 67-million-weight encoder trained on labels against a 1.5-billion-weight model that was only prompted.</div>'
          '</div></div>'))
