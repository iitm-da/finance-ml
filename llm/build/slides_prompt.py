"""LLM session, part 3: prompt engineering, measured."""
from common import *

C = []
add = C.append
SO = PR['so']; SENS = PR['sens']; FEW = PR['few']; COT = PR['cot']; INJ = PR['inj']
PN = list(SO)
FIELDS = ['borrower', 'lender', 'principal', 'rate', 'maturity', 'law']
RU, QA = NLP_IE['m']['rules'], NLP_IE['m']['qa']
INJ_ = PR['inj']

# ================================================================ concept: the parts of a prompt (build)
parts = [('role', 'who the model is answering for', SLATE, 'You answer questions from staff of Northmere Bank.'),
         ('task', 'what to do, in one sentence', BLUE, 'Extract six fields from the loan agreement below.'),
         ('format', 'the exact shape of the reply', PURP, 'Reply with one JSON object and nothing else, with exactly these keys: ...'),
         ('definitions', 'what each word in the task means', AMBER, 'The principal is the amount lent, not the arrangement fee.'),
         ('example', 'one input with its correct output', GREEN, 'Example agreement: ...   Example reply: {"borrower": ...}'),
         ('input', 'the text to work on, fenced off', INK, 'Agreement: ...')]
s = ''
for i, (a, b, c, ex) in enumerate(parts):
    y = 6 + i * 44
    piece = Rect(10, y, 150, 36, c) + T(85, y + 23, a, 13.5, '#FFFFFF', 'middle', BODY, 700)
    piece += T(174, y + 16, b, 12, INK, 'start', BODY, 600) + T(174, y + 31, ex, 11, SOFT, 'start', MONO)
    s += G_(max(0, i - 1), piece) if i > 1 else piece
add(slide('Prompting', 'Prompting &#183; Concept', 'Concept',
          'A prompt is a specification. Each part removes one way for the model to do something other than what was meant.',
          '<div class="cols"><div class="colstack" style="flex:1.5">' + fig(svg(760, 272, s, 'The parts of a prompt'), 'the parts, in the order they are written') +
          '</div><div class="colstack">'
          '<div class="callout"><b>Write it for a new colleague.</b> Someone capable who has never seen the bank&#8217;s documents needs the same things: the task, the format, the definitions and one worked case.</div>'
          '<div class="callout" data-at="3"><b>An example outweighs a paragraph.</b> One input with its correct output shows the format, the level of detail and the edge cases at once.</div>'
          '<div class="callout warn sm" data-at="4"><b>A prompt is code.</b> Keep it in version control, and rerun the question set when it changes. The next slides measure how much one sentence moves the result.</div>'
          '</div></div>' + bld(4), 'build'))

# ================================================================ structured output: four prompts
s = ''; y = 6; lw = 210; span = 330
for n in PN:
    s += T(lw - 10, y + 20, n, 12, INK, 'end', MONO, 600)
    for v, c, lab in ((SO[n]['valid'], SLATE, 'valid JSON'), (SO[n]['all'], GREEN, 'fields right')):
        s += Rect(lw, y, max(2, span * v), 16, c) + T(lw + span * v + 6, y + 12.5, f'{pct(v)}', 11.5, INK, 'start', MONO, 700)
        y += 17
    y += 12
for n_, v_, c_ in (('rules (morning session)', mean(RU), AMBER), ('QA model (morning session)', mean(QA), BLUE)):
    s += T(lw - 10, y + 15, n_, 12, SOFT, 'end', MONO, 600) + Rect(lw, y + 2, max(2, span * v_), 16, c_) + T(lw + span * v_ + 6, y + 14.5, pct(v_), 11.5, INK, 'start', MONO, 700)
    y += 26
P1, P4 = SO[PN[0]], SO[PN[-1]]
add(slide('Prompting', 'Structured output &#183; Example', 'Example',
          f"The six loan-agreement fields from the morning session, now asked of the language model on {PR['n_loans']} agreements. Four prompts, each adding one part.",
          '<div class="cols"><div class="colstack" style="flex:1.35">' +
          fig(svg(640, y + 2, s, 'Valid JSON and field accuracy by prompt'), 'share of replies that parse, and share of fields right',
              f'<span class="tag" style="background:{SLATE}">&#160;</span>valid JSON with all six keys &#160; <span class="tag" style="background:{GREEN}">&#160;</span>fields right') +
          '</div><div class="colstack">' +
          f'<div class="callout {"warn" if P1["valid"] < 0.5 else ""}"><b>Plain request: {pct(P1["valid"], 0)} parse.</b> The model answers in prose or a bullet list. A program downstream cannot read it, whatever the content.</div>'
          f'<div class="callout"><b>Naming the keys: {pct(SO[PN[1]]["valid"], 0)} parse</b>, {pct(SO[PN[1]]["all"], 0)} of fields right. Format is solved by asking for it.</div>'
          f'<div class="callout good"><b>Example and definitions: {pct(P4["all"], 0)} of fields right.</b> The rules scored {pct(mean(RU), 0)} and the question-answering model {pct(mean(QA), 0)} on these documents.</div>'
          '</div></div>'))

# ================================================================ structured output: field by field
rows = [[f, pct(RU[f], 0), pct(QA[f], 0)] + [pct(SO[n]['field'][f], 0) for n in PN[1:]] for f in FIELDS]
rows.append(['<b>all</b>', f'<b>{pct(mean(RU), 0)}</b>', f'<b>{pct(mean(QA), 0)}</b>'] + [f"<b>{pct(SO[n]['all'], 0)}</b>" for n in PN[1:]])
gain = max(FIELDS, key=lambda f: SO[PN[3]]['field'][f] - SO[PN[1]]['field'][f])
add(slide('Prompting', 'Structured output &#183; Example', 'Example continued 1',
          f"Field by field. The worked example lifts the {gain} from {pct(SO[PN[1]]['field'][gain], 0)} to {pct(SO[PN[2]]['field'][gain], 0)}, and the definitions leave it at {pct(SO[PN[3]]['field'][gain], 0)}. The rate goes the other way: {pct(SO[PN[1]]['field']['rate'], 0)}, {pct(SO[PN[2]]['field']['rate'], 0)}, {pct(SO[PN[3]]['field']['rate'], 0)}.",
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          table(['field', 'rules', 'QA model', 'prompt 2', 'prompt 3', 'prompt 4'], rows, 'd sm free') +
          '<div class="callout sm"><b>Three tools, one set of documents.</b> Rules are exact where they match. The question-answering model copies spans and scores them. The language model follows definitions written in English.</div>'
          '</div><div class="colstack">' +
          '<div class="outlbl">what prompt 4 adds to prompt 3</div>' +
          '<div class="out sm" style="white-space:normal">The principal is the amount lent, not the arrangement fee. The rate is the ordinary interest rate, not the default rate. The maturity is the date of the last repayment, not the first. Copy each value exactly as written.</div>' +
          '<div class="outlbl">reply to prompt 1, first agreement</div>' + f'<div class="out xs" style="white-space:pre-wrap;max-height:150px;overflow:hidden">{esc(P1["reply0"][:330])}</div>' +
          '</div></div>'))

# ================================================================ wording
ks = list(SENS); vals = [SENS[k]['f1'] for k in ks]
lo, hi = min(vals), max(vals)
ax = Ax(560, 300, (-0.5, len(ks) - 0.5), (max(0, lo - 0.12), min(1, hi + 0.08)), pad=(46, 14, 14, 30))
yt = [v / 20 for v in range(int(ax.yr[0] * 20) + 1, int(ax.yr[1] * 20) + 1, 2)]
s = ax.frame(yticks=yt, xticks=[(i, f'prompt {k}') for i, k in enumerate(ks)], yfmt=lambda v: f'{v:.2f}', ylab='macro-F1')
for i, k in enumerate(ks):
    s += L(ax.X(i), ax.Y(ax.yr[0]), ax.X(i), ax.Y(SENS[k]['f1']), LINE, 2) + ax.dot(i, SENS[k]['f1'], PURP, 7)
    s += T(ax.X(i), ax.Y(SENS[k]['f1']) - 12, f"{SENS[k]['f1']:.3f}", 11.5, INK, 'middle', MONO, 700)
kb_, kw_ = max(ks, key=lambda k: SENS[k]['f1']), min(ks, key=lambda k: SENS[k]['f1'])
instr = PR['instr']
add(slide('Prompting', 'Wording &#183; Example', 'Example',
          f"Six instructions for the sentiment task that a person would call equivalent, on the same {PR['n_sub']} headlines. Macro-F1 runs from {lo:.3f} to {hi:.3f}.",
          '<div class="cols"><div class="colstack" style="flex:1.1">' + fig(svg(560, 300, s, 'Macro-F1 for six instructions'), 'zero-shot macro-F1, six wordings') +
          '</div><div class="colstack" style="flex:1.2">' +
          f'<div class="outlbl">prompt {kb_}: {SENS[kb_]["f1"]:.3f}</div><div class="out sm" style="white-space:normal">{esc(instr[kb_].replace("{t}", "<headline>"))}</div>' +
          f'<div class="outlbl">prompt {kw_}: {SENS[kw_]["f1"]:.3f}</div><div class="out sm" style="white-space:normal">{esc(instr[kw_].replace("{t}", "<headline>"))}</div>' +
          f'<div class="callout warn sm"><b>A spread of {hi - lo:.3f} from wording alone.</b> A prompt compared with another on five examples by eye says nothing. Compare prompts on the question set, as models are compared.</div>'
          '</div></div>'))

# ================================================================ few-shot
kk = sorted(FEW['random'], key=int)
z = SENS['A']['f1']
allv = [z] + [FEW[m][k] for m in FEW for k in kk]
ax = Ax(600, 310, (-1, 25), (max(0, min(allv) - 0.06), min(1, max(allv) + 0.06)), pad=(46, 14, 14, 34))
yt = [v / 20 for v in range(int(ax.yr[0] * 20) + 1, int(ax.yr[1] * 20) + 1)]
s = ax.frame(yticks=yt, xticks=[(0, '0')] + [(int(k), k) for k in kk], yfmt=lambda v: f'{v:.2f}', ylab='macro-F1', xlab='labelled examples in the prompt')
for m, c in (('random', AMBER), ('nearest', GREEN)):
    xs = [0] + [int(k) for k in kk]; ys = [z] + [FEW[m][k] for k in kk]
    s += ax.line(xs, ys, c, 2.6)
    for x_, y_ in zip(xs, ys):
        s += ax.dot(x_, y_, c, 4)
    s += T(ax.X(24) - 6, ax.Y(ys[-1]) + (-10 if m == 'nearest' else 18), f'{m}: {ys[-1]:.3f}', 11.5, c, 'end', MONO, 700)
kmax = kk[-1]
add(slide('Prompting', 'Examples &#183; Example', 'Example',
          f"Labelled headlines added to the prompt. {kmax} random examples move macro-F1 from {z:.3f} to {FEW['random'][kmax]:.3f}; the {kmax} nearest examples to each headline reach {FEW['nearest'][kmax]:.3f}.",
          '<div class="cols"><div class="colstack" style="flex:1.3">' + fig(svg(600, 310, s, 'Macro-F1 against number of examples'), f'macro-F1 on {PR["n_sub"]} headlines') +
          '</div><div class="colstack">'
          '<div class="callout"><b>Random.</b> The same examples for every headline, drawn once from the training rows.</div>'
          '<div class="callout good"><b>Nearest.</b> For each headline, embed it and take the most similar labelled rows from a pool of 3,000. Retrieval, applied to examples.</div>'
          '<div class="callout sm"><b>The cost is prompt length.</b> 24 examples add about 700 tokens to every call. A fine-tuned model carries the same knowledge in its weights and reads a short prompt.</div>'
          '</div></div>'))

# ================================================================ chain of thought
c0, c1 = COT['answer at once'], COT['step by step']
calc = PR['calc'][1]
items = [('answer at once', round(c0['acc'], 3), AMBER, ''), ('step by step', round(c1['acc'], 3), GREEN, '')]
add(slide('Prompting', 'Working &#183; Example', 'Example',
          f"{PR['n_calc']} fee calculations with one right answer each. Asked for the amount only, the model is right on {pct(c0['acc'], 0)}; asked to work step by step first, on {pct(c1['acc'], 0)}.",
          '<div class="cols"><div class="colstack">' +
          fig(hbars(items, w=540, label_w=150, bar_h=34, gap=14, maxv=1.0, valfmt=lambda v: pct(v, 0), label='Accuracy with and without working'), 'share of calculations right') +
          f'<div class="outlbl">one of the calculations (right answer {calc[1]:g})</div><div class="out sm" style="white-space:normal">{esc(calc[0])}</div>' +
          '<div class="callout sm"><b>What working does.</b> The model writes one token at a time, and each intermediate result becomes input for the next token. A mistake is carried forward the same way: below, the model multiplies correctly and then adds the minimum fee.</div>'
          '</div><div class="colstack">' + '<div class="outlbl">the model&#8217;s working</div>' +
          f'<div class="out xs" style="white-space:pre-wrap;max-height:300px;overflow:hidden">{esc(c1["ex"][:620])}</div>' +
          '<div class="callout warn sm"><b>For arithmetic that matters, call a function.</b> Let the model pick the rule and the inputs, and let code do the multiplication.</div>'
          '</div></div>'))

# ================================================================ injection
i0, i1 = INJ['strict prompt'], INJ['+ fenced context and a warning']
items = [('strict prompt', round(i0['obeyed'], 3), RED, f"still correct {pct(i0['correct'], 0)}"),
         ('+ fenced context, warning', round(i1['obeyed'], 3), AMBER, f"still correct {pct(i1['correct'], 0)}"),
         ('+ check on the output', round(i1['obeyed_after_check'], 3), GREEN, '')]
add(slide('Prompting', 'Hidden instructions &#183; Example', 'Example',
          f"One retrieved section has a sentence added that tells the model what to say. Over {INJ['n']} questions the model obeyed it {pct(i0['obeyed'], 0)} of the time.",
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          '<div class="outlbl">the sentence planted in a document</div>' + f'<div class="out sm" style="white-space:normal">{esc(PR["inj_text"])}</div>' +
          fig(hbars(items, w=620, label_w=230, bar_h=30, gap=12, maxv=1.0, valfmt=lambda v: pct(v, 0), label='How often the model obeyed the planted instruction'), 'share of answers that obeyed the planted sentence') +
          '</div><div class="colstack">'
          '<div class="callout warn"><b>The model reads one stream of text.</b> It has no channel that separates the bank&#8217;s instruction from a sentence inside a document. Whoever can edit a document that gets retrieved can address the model.</div>'
          '<div class="callout"><b>Fences and a warning.</b> Marking the context as reference material took the rate from ' + pct(INJ_['strict prompt']['obeyed'], 0) + ' to ' + pct(INJ_['+ fenced context and a warning']['obeyed'], 0) + ' on these questions, and correct answers from ' + pct(INJ_['strict prompt']['correct'], 0) + ' to ' + pct(INJ_['+ fenced context and a warning']['correct'], 0) + '. A zero on ' + str(INJ_['n']) + ' questions is no guarantee for the next document.</div>'
          '<div class="callout sm"><b>Controls outside the model.</b> Check the output against the context. Keep the model&#8217;s permissions narrow: a model that can only read cannot be talked into sending a payment.</div>'
          '</div></div>'))

# ================================================================ verdict
add(slide('Prompting', 'Prompting &#183; Verdict', 'Verdict',
          'Four measurements, four habits.',
          '<div class="cols4">'
          f'<div class="card"><h3>State the format</h3><p>Valid JSON went from {pct(SO[PN[0]]["valid"], 0)} to {pct(SO[PN[1]]["valid"], 0)} when the keys were named. Parse the reply in code and retry on failure.</p></div>'
          f'<div class="card"><h3>Define the terms</h3><p>Fields right went from {pct(SO[PN[1]]["all"], 0)} to {pct(SO[PN[3]]["all"], 0)} with one example and three definitions.</p></div>'
          f'<div class="card g"><h3>Choose the examples</h3><p>{kmax} nearest examples scored {FEW["nearest"][kmax]:.3f} against {FEW["random"][kmax]:.3f} for {kmax} random ones.</p></div>'
          f'<div class="card o"><h3>Measure the wording</h3><p>Six equivalent instructions spread over {hi - lo:.3f} of macro-F1. Keep a question set and rerun it on every edit.</p></div>'
          '</div>'
          '<div class="cols nat"><div class="callout"><b>A prompt changes behaviour, not knowledge.</b> It can make the model use the page, follow a format and show its working. It cannot supply a fact that is absent from both the weights and the context.</div>'
          f'<div class="callout warn"><b>Treat retrieved text as untrusted input.</b> The planted sentence was obeyed {pct(i0["obeyed"], 0)} of the time under a careful prompt.</div></div>'))
