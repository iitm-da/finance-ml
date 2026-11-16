"""LLM session, part 2: evaluating answers, detecting and mitigating hallucination."""
from common import *

B = []
add = B.append
POOL = EV['pool']; MET = EV['metric']; DET = EV['det']; LAD = EV['ladder']; STEPS = EV['steps']
NP = sum(POOL.values())
DCOL = {'number check': GREEN, 'entailment model': BLUE, 'self-consistency': PURP, 'model as judge': AMBER}

# ================================================================ concept: what to measure
add(slide('Evaluation', 'Evaluation &#183; Concept', 'Concept',
          'An LLM system is evaluated like any other model: a fixed set of inputs with known answers, run again after every change.',
          '<div class="cols3">'
          '<div class="card big"><h3>1. A question set</h3><p>Real questions from the people who will use the system, each with the accepted answer and the document that holds it. Include questions that have no answer. A few hundred is enough to see a change of a few points.</p></div>'
          '<div class="card big"><h3>2. A grade per answer</h3><p>Three outcomes: correct, abstained, wrong. They have different costs. An abstention sends the member of staff to the document. A wrong answer sends them away with a false number.</p></div>'
          '<div class="card big g"><h3>3. A score per stage</h3><p>Recall@k for retrieval, the three grades for the answer, and the abstention rate on unanswerable questions. One blended score hides which stage moved.</p></div>'
          '</div>'
          '<div class="cols nat"><div class="callout"><b>Here the grade is exact.</b> Every accepted answer is a number or a short phrase, so a string match decides. Free-text answers need a person, or a model checked against a person.</div>'
          f'<div class="callout warn"><b>The test bed for this block.</b> Dense retrieval with a plain prompt, the weakest pipeline so far: {POOL["correct"]} correct, {POOL["abstained"]} abstained, {POOL["wrong"]} wrong out of {NP}. It supplies enough wrong answers to test a detector on.</div></div>'))

# ================================================================ overlap metrics
items = [(k, round(MET[k]['auc'], 3), SLATE, f"mean {MET[k]['mean_ok']:.2f} when correct, {MET[k]['mean_bad']:.2f} when not") for k in MET]
tr = EV['tricky'][0]
add(slide('Evaluation', 'Evaluation &#183; Example', 'Example',
          'Three scores that compare an answer with a reference sentence. AUC is the chance that a correct answer scores above an incorrect one; 0.5 is a coin toss.',
          '<div class="cols"><div class="colstack" style="flex:1.25">' +
          fig(hbars(items, w=680, label_w=190, bar_h=30, gap=14, maxv=1.45, valfmt=lambda v: f'{v:.3f}', label='AUC of three overlap metrics'), 'AUC for separating correct from incorrect answers') +
          '<div class="outlbl">an answer graded wrong</div>' +
          f'<div class="out sm" style="white-space:normal"><b>reference:</b> {esc(tr[0])}<br/><b>answer:</b> {esc(tr[1][:230])}<br/><b>ROUGE-L {tr[2]:.2f}, embedding similarity {tr[3]:.2f}</b></div>' +
          '</div><div class="colstack">'
          '<div class="callout"><b>What the scores count.</b> Token F1 and ROUGE-L count shared words. Embedding similarity compares meaning as a sentence-embedding model sees it.</div>'
          '<div class="callout warn"><b>One digit is the whole answer.</b> A sentence that copies the reference and changes 450 to 300 shares nearly every word with it. In finance the number is the fact.</div>'
          '<div class="callout sm">Use overlap scores to track drift in wording. Grade correctness on the values.</div>'
          '</div></div>'))

# ================================================================ hallucination: concept
una_w = LAD['RAG, dense, top 3']['una']['wrong']; ans_w = LAD['RAG, dense, top 3']['ans']['wrong']
add(slide('Hallucination', 'Hallucination &#183; Concept', 'Concept',
          f"A hallucination is an answer stated as fact that the documents do not support. In the test bed, {POOL['wrong']} of {NP} answers are wrong.",
          '<div class="cols3">'
          '<div class="card big r"><h3>No support retrieved</h3><p>The answering section was not among the top three, or does not exist. The model fills the gap with a plausible value, often a number from a neighbouring product.</p></div>'
          '<div class="card big o"><h3>Support misread</h3><p>The right section is in the prompt. The model takes the wrong number from it, mixes two sections, or answers a slightly different question.</p></div>'
          '<div class="card big"><h3>Support out of date</h3><p>A superseded section is retrieved and quoted faithfully. The answer is grounded in a document and still wrong.</p></div>'
          '</div>'
          '<div class="cols nat">'
          f'<div class="callout"><b>Unanswerable questions are the hard case.</b> With the plain prompt the model gave a wrong answer to {pct(una_w, 0)} of them, against {pct(ans_w, 0)} of the answerable ones.</div>'
          '<div class="callout"><b>Fluency carries no signal.</b> The model writes a wrong answer in the same confident sentence as a right one. Detection has to come from outside the sentence.</div></div>'))

# ================================================================ detectors: mechanism
add(slide('Hallucination', 'Hallucination &#183; Mechanism', 'Mechanism',
          'Four detectors that need no reference answer. Each reads the question, the retrieved context and the answer, and raises a flag.',
          '<div class="cols4">'
          '<div class="card g"><h3>1. Number check</h3><p>Extract every number from the answer. Flag the answer if any of them is absent from the retrieved context. Ten lines of code and no model.</p></div>'
          '<div class="card"><h3>2. Entailment model</h3><p>A small encoder trained on natural language inference reads context and answer and scores whether the first supports the second.</p></div>'
          '<div class="card"><h3>3. Self-consistency</h3><p>Ask the same question five more times at temperature 0.8. Flag the answer if most samples give different numbers. A model that knows repeats itself.</p></div>'
          '<div class="card o"><h3>4. Model as judge</h3><p>Show the model the context, the question and the answer, and ask: is every fact in the answer stated in the context? Flag on NO.</p></div>'
          '</div>'
          '<div class="cols nat"><div class="callout"><b>Cost.</b> The number check is free. The entailment model adds one small forward pass. The judge adds one generation. Self-consistency adds five.</div>'
          '<div class="callout sm"><b>Scoring a detector.</b> Recall: the share of wrong answers it flags. Precision: the share of its flags that are wrong answers.</div></div>'))

# ================================================================ detectors: example
rows = [[f'<span class="tag" style="background:{DCOL[k]}">&#160;</span>{k}', d['flagged'], f"{d['caught']} of {d['wrong']}", f"{d['precision']:.2f}", f"<b>{d['recall']:.2f}</b>", f"{d['auc']:.3f}"] for k, d in DET.items()]
best_r = max(DET, key=lambda k: DET[k]['recall']); best_p = max(DET, key=lambda k: DET[k]['precision']); worst = min(DET, key=lambda k: DET[k]['auc'])
s = ''; x0 = 60; W_ = 520; H_ = 300
ax = Ax(W_, H_, (0, 1), (0, 1), pad=(46, 16, 14, 34))
s = ax.frame(yticks=[0, .25, .5, .75, 1], xticks=[(v, f'{v:.2f}') for v in (0, .25, .5, .75, 1)], yfmt=lambda v: f'{v:.2f}', ylab='precision', xlab='recall')
for k, d in DET.items():
    s += ax.dot(d['recall'], d['precision'], DCOL[k], 8) + T(ax.X(d['recall']) + (12 if d['recall'] < .7 else -12), ax.Y(d['precision']) + 4, k, 11.5, DCOL[k], 'start' if d['recall'] < .7 else 'end', MONO, 700)
add(slide('Hallucination', 'Hallucination &#183; Example', 'Example',
          f"The four detectors on the {NP} test-bed answers, {POOL['wrong']} of them wrong. {best_r.capitalize()} flags the most wrong answers ({DET[best_r]['recall']:.2f}); {best_p} has the cleanest flags ({DET[best_p]['precision']:.2f}).",
          '<div class="cols"><div class="colstack" style="flex:1.3">' +
          table(['detector', 'flagged', 'wrong answers caught', 'precision', 'recall', 'AUC'], rows, 'd sm free') +
          f'<div class="callout {"warn" if DET["model as judge"]["auc"] < 0.7 else ""}"><b>The judge is the model that wrote the answer.</b> It said YES to {pct(EV["judge_yes"], 0)} of all answers. '
          + ('A model that produced an error is poorly placed to notice it.' if DET['model as judge']['recall'] < 0.6 else 'Asked a narrow yes or no question, it catches more than it writes.') + '</div>'
          f'<div class="callout sm">Self-consistency cost {EV["t_sc"]:.0f} seconds of generation for five extra samples a question.</div>'
          '</div><div class="colstack">' + fig(svg(W_, H_, s, 'Precision and recall of four detectors'), 'each detector as one point') +
          '</div></div>'))

# ================================================================ self-consistency example
ex = EV['sc_ex'][0]; ex2 = EV['sc_ex'][-1]
def scblock(e):
    samples = ''.join(f'<div style="font-family:var(--mono);font-size:11.5px;padding:3px 0;border-top:1px solid var(--hair)">{i + 1}. {esc(x[:140])}</div>' for i, x in enumerate(e[2]))
    col = {'correct': 'g', 'wrong': 'r', 'abstained': 'o'}[e[3]]
    return (f'<div class="card sm {col}"><h3>{esc(e[0])}</h3><p style="margin-bottom:6px"><b>first answer ({e[3]}):</b> {esc(e[1][:170])}</p>{samples}</div>')
add(slide('Hallucination', 'Hallucination &#183; Example', 'Example continued 1',
          'Self-consistency on two questions: the first answer at temperature 0, then five samples at temperature 0.8.',
          '<div class="cols"><div class="colstack">' + scblock(ex) + '</div><div class="colstack">' + scblock(ex2) + '</div></div>'
          '<div class="cols nat"><div class="callout"><b>Disagreement is the signal.</b> When the number is on the page in front of the model, sampling returns it again. When the model is guessing, each sample guesses differently.</div>'
          '<div class="callout warn sm"><b>Agreement is weaker evidence.</b> A model can be consistently wrong, for instance when it copies the same superseded section five times.</div></div>'))

# ================================================================ mitigation ladder
short = {'closed book': 'closed book', 'RAG, dense, top 3': '+ dense retrieval', 'RAG, hybrid + reranker, top 3': '+ hybrid and reranker',
         '+ strict prompt': '+ strict prompt', '+ number check on the output': '+ number check on the output'}
first, last = LAD[STEPS[0]], LAD[STEPS[-1]]
add(slide('Hallucination', 'Hallucination &#183; Example', 'Example continued 2',
          f"Mitigation, one step at a time, on all {NP} questions. Wrong answers fall from {pct(first['wrong'], 0)} to {pct(last['wrong'], 0)}.",
          '<div class="cols"><div class="colstack" style="flex:1.5">' +
          fig(stacked([(short[s_], LAD[s_]) for s_ in STEPS], w=720, label_w=240, bar_h=34, gap=14, label='Grades by mitigation step'), f'all {NP} questions, answerable and unanswerable', GLEG) +
          '</div><div class="colstack">' +
          table(['step', 'wrong', 'correct'], [[short[s_], f"<b>{pct(LAD[s_]['wrong'])}</b>", pct(LAD[s_]['correct'])] for s_ in STEPS], 'd sm free') +
          f'<div class="callout sm"><b>Each step removes a different error.</b> Retrieval supplies the fact. The strict prompt stops answers to unanswerable questions. The number check stops figures that are absent from the context.</div>'
          '</div></div>'))

# ================================================================ mitigation: what it costs
a_first = LAD['+ strict prompt']['ans']; a_last = LAD['+ number check on the output']['ans']
rows = [[short[s_], pct(LAD[s_]['ans']['correct']), pct(LAD[s_]['ans']['abstained']), pct(LAD[s_]['ans']['wrong']), pct(LAD[s_]['una']['abstained'])] for s_ in STEPS]
add(slide('Hallucination', 'Hallucination &#183; Drawbacks', 'Drawbacks',
          f"Each guard also blocks some right answers. The number check moves correct answers on answerable questions from {pct(a_first['correct'])} to {pct(a_last['correct'])}.",
          '<div class="cols"><div class="colstack" style="flex:1.4">' +
          table(['step', 'answerable: correct', 'abstained', 'wrong', 'unanswerable: abstained'], rows, 'd sm free') +
          '</div><div class="colstack">'
          '<div class="callout warn"><b>Abstention has a price.</b> A system that refuses whenever it is unsure is never wrong and soon unused. The share of answerable questions it refuses belongs on the same report as the error rate.</div>'
          '<div class="callout"><b>A number check passes a wrong number that is on the page.</b> It tests whether the figure came from the context, and has no view on whether it is the right figure.</div>'
          '<div class="callout sm"><b>No step reaches zero.</b> For answers that move money or go to a customer, the last guard is a person who can see the cited section.</div>'
          '</div></div>'))
