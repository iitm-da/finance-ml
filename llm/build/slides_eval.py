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
          f'<div class="callout warn"><b>The test bed for this block.</b> Dense retrieval with a plain prompt, the retrieval pipeline with the most wrong answers: {POOL["correct"]} correct, {POOL["abstained"]} abstained, {POOL["wrong"]} wrong out of {NP}. It supplies enough wrong answers to test a detector on.</div></div>'))

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
          f"The four detectors on the {NP} test-bed answers, {POOL['wrong']} of them wrong. The entailment model catches all {DET['entailment model']['caught']} by flagging {DET['entailment model']['flagged']} of the {NP}. The judge catches {DET['model as judge']['caught']} with {DET['model as judge']['flagged']} flags; {best_p} has the cleanest flags (precision {DET[best_p]['precision']:.2f}).",
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
          '<div class="cols nat"><div class="callout"><b>Disagreement is the signal, and it misfires.</b> The first answer on the left is correct. Two samples drift to the GBP 2 statement fee and three give no figure, so the number comparison counts all five as changed.</div>'
          '<div class="callout warn sm"><b>Agreement is weaker evidence.</b> The wrong answer on the right is repeated by all five samples. The model copies the same superseded section each time.</div></div>'))

# ================================================================ finer checks: claim-level faithfulness
XM = LM['eval_more']; CL, LGP, JD, TRI = XM['claim'], XM['logp'], XM['judge'], XM['triad']
CD = CL['det']; CN = list(CD)
cshort = {CN[0]: 'whole context', CN[1]: 'sentence by sentence', CN[2]: 'claims, by sentence'}
items = [(cshort[k], round(CD[k]['auc'], 3), GREEN if k == CN[2] else BLUE, f"flags {CD[k]['flagged']}, catches {CD[k]['caught']} of {CD[k]['wrong']}") for k in CN]
ex = CL['ex']
exrows = ''.join(f'<div style="font-family:var(--mono);font-size:11.5px;padding:4px 0;border-top:1px solid var(--hair)"><b style="color:{GREEN if u >= .5 else RED}">{u:.2f}</b> &#160;{esc(c)}</div>' for c, u in ex[2])
add(slide('Hallucination', 'Hallucination &#183; Claim-level faithfulness &#183; Example', 'Example',
          f"The entailment model again, on smaller pieces. Whole context against whole answer: AUC {CD[CN[0]]['auc']:.3f}. Each claim against each context sentence: {CD[CN[2]]['auc']:.3f}.",
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          fig(hbars(items, w=680, label_w=190, bar_h=30, gap=14, maxv=1.5, valfmt=lambda v: f'{v:.3f}', label='AUC of the entailment check at three grain sizes'),
              f'AUC for flagging the {CD[CN[0]]["wrong"]} wrong answers among {NP}') +
          table(['check', 'flagged', 'caught', 'precision', 'recall'],
                [[cshort[k], CD[k]['flagged'], f"{CD[k]['caught']} of {CD[k]['wrong']}", f"{CD[k]['precision']:.2f}", f"<b>{CD[k]['recall']:.2f}</b>"] for k in CN], 'd sm free') +
          '</div><div class="colstack">' +
          f'<div class="card sm r"><h3>{esc(ex[0])}</h3><p style="margin-bottom:6px"><b>answer (wrong):</b> {esc(ex[1])}</p>'
          f'<div class="outlbl" style="margin-top:4px">claims written by the model, with the best support among the context sentences</div>{exrows}</div>'
          f'<div class="callout sm"><b>Two steps.</b> The model splits each answer into claims, {CL["mean_claims"]:.1f} on average. A claim is supported if one context sentence entails it. The {CL["n_abst"]} answers worded as an abstention make no claim.</div>'
          f'<div class="callout warn sm"><b>Precision stays low.</b> {CD[CN[2]]["flagged"]} answers flagged to catch {CD[CN[2]]["caught"]}. A split claim such as the second one above has no sentence to match, whatever the answer.</div>'
          '</div></div>'))

# ================================================================ finer checks: token probabilities
LD = LGP['det']; LN = list(LD)
def strip(e, w=620):
    s = ''; x = 6
    for t, pr in zip(e['tok'], e['p']):
        t = t.strip() or '·'; bw = max(34, len(t) * 8.6 + 14)
        col = GREEN if pr >= .9 else (AMBER if pr >= .5 else RED)
        s += Rect(x, 4, bw, 30, '#FFFFFF', col, 1.6) + Rect(x, 30, bw * pr, 4, col) + T(x + bw / 2, 23, t, 12.5, INK, 'middle', MONO, 600) + T(x + bw / 2, 50, f'{pr:.2f}', 10.5, col, 'middle', MONO, 700)
        x += bw + 4
    return svg(w, 56, s, 'Probability the model gave each token of its answer')
e_ok, e_bad = LGP['ex']
SC_ = DET['self-consistency']; MJ = DET['model as judge']
add(slide('Hallucination', 'Hallucination &#183; Token probability &#183; Example', 'Example',
          f"The model gave every token of its answer a probability as it wrote it. The mean log-probability separates wrong answers with AUC {LD[LN[0]]['auc']:.3f}, read back in {LGP['secs']:.0f} seconds for {NP} answers.",
          '<div class="cols"><div class="colstack" style="flex:1.25">' +
          fig(strip(e_ok), f'graded correct &#183; {esc(e_ok["q"])}') + fig(strip(e_bad), f'graded wrong &#183; {esc(e_bad["q"])}') +
          table(['score', 'AUC', 'flagged', 'caught', 'precision', 'recall'],
                [[k, f"<b>{d['auc']:.3f}</b>", d['flagged'], f"{d['caught']} of {d['wrong']}", f"{d['precision']:.2f}", f"{d['recall']:.2f}"] for k, d in LD.items()], 'd sm free') +
          '</div><div class="colstack">'
          f'<div class="callout good sm"><b>One forward pass.</b> AUC {LD[LN[0]]["auc"]:.3f} against {SC_["auc"]:.3f} for self-consistency, which took {EV["t_sc"]:.0f} seconds of extra generation, and {MJ["auc"]:.3f} for the judge.</div>'
          f'<div class="callout sm"><b>The flag rule matters.</b> Flagging when the probability is below 0.5: the average over the tokens flags {LD[LN[0]]["flagged"]} answers at precision {LD[LN[0]]["precision"]:.2f}; the lowest token flags {LD[LN[1]]["flagged"]} at {LD[LN[1]]["precision"]:.2f}.</div>'
          f'<div class="callout warn sm"><b>The digits are written with confidence.</b> The lowest probability on a number token has AUC {LD[LN[2]]["auc"]:.3f}. In the wrong answer, the lower strip, the doubt sits on the first digit, 3, and the two zeros follow at 1.00.</div>'
          '<div class="callout sm">An API that returns no token probabilities rules this check out.</div>'
          '</div></div>'))

# ================================================================ finer checks: validating the judge
J1, J2_ = JD['yes or no'], JD['copy the sentence, then verdict']
def confusion(j, w=420):
    s = T(210, 14, 'exact grade', 11, SOFT, 'middle', MONO) + T(150, 34, 'wrong', 12, INK, 'middle', MONO, 700) + T(310, 34, 'not wrong', 12, INK, 'middle', MONO, 700)
    for r, (lab, a, b) in enumerate((('judge: NO', j['tp'], j['fp']), ('judge: YES', j['fn'], j['tn']))):
        y = 44 + r * 74
        s += T(64, y + 40, lab, 12, INK, 'end', MONO, 700)
        for c, (v, good) in enumerate(((a, r == 0), (b, r == 1))):
            s += Rect(72 + c * 160, y, 154, 68, '#EAF6F0' if good else '#FBECEE', GREEN if good else RED, 1.4) + T(149 + c * 160, y + 44, v, 26, GREEN if good else RED, 'middle', MONO, 700)
    return svg(w, 198, s, 'Confusion matrix of the judge against the exact grade')
add(slide('Hallucination', 'Hallucination &#183; Judge validation &#183; Test', 'Test',
          f"A judge is a classifier and is scored like one. Against the exact grade, the yes-or-no judge agrees on {pct(J1['agree'], 0)} of {JD['n']} answers, with Cohen&#8217;s kappa {J1['kappa']:.2f}.",
          '<div class="cols"><div class="colstack">' +
          fig(confusion(J1), 'the yes-or-no judge &#183; NO means &#8220;some fact in the answer is absent from the context&#8221;') +
          table(['judge', 'agreement', 'kappa', 'flagged'],
                [['yes or no', pct(J1['agree'], 0), f"<b>{J1['kappa']:.2f}</b>", J1['tp'] + J1['fp']],
                 ['copy the sentence, then verdict', pct(J2_['agree'], 0), f"<b>{J2_['kappa']:.2f}</b>", J2_['tp'] + J2_['fp']]], 'd sm free') +
          '</div><div class="colstack">'
          '<div class="callout sm"><b>Kappa</b> is agreement beyond what chance would give: 0 is chance, 1 is perfect. Most answers are right, so a judge that says YES to everything already agrees on ' + pct((J1['fp'] + J1['tn']) / JD['n'], 0) + '.</div>'
          f'<div class="callout warn sm"><b>A longer rubric made it worse.</b> Asked to copy the supporting sentence before the verdict, this model flags {J2_["tp"] + J2_["fp"]} of {JD["n"]} answers. Kappa {J2_["kappa"]:.2f}.</div>'
          f'<div class="callout warn sm"><b>Asked again, it changes its mind.</b> Sampled three more times at temperature 0.8, the yes-or-no judge repeats its first verdict all three times on {pct(JD["stable"], 0)} of answers.</div>'
          '<div class="callout good sm"><b>Before a judge scores anything,</b> grade a few hundred answers by hand or by rule, report kappa against them, and run the judge at temperature 0.</div>'
          '</div></div>'))

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
          f'<div class="callout sm"><b>What each step did.</b> Retrieval supplies the fact. The strict prompt stops answers to unanswerable questions. The number check removed no wrong answer on this run: wrong stays at {pct(LAD[STEPS[-1]]['wrong'])} and correct falls from {pct(LAD[STEPS[-2]]['correct'])} to {pct(LAD[STEPS[-1]]['correct'])}.</div>'
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

# ================================================================ RAG triad: concept (build)
Pq, Pc, Pa = (280, 40), (90, 250), (470, 250)
def node(p, lab, sub, col):
    return Rect(p[0] - 80, p[1] - 26, 160, 52, '#FFFFFF', HAIR, 1) + Rect(p[0] - 80, p[1] - 26, 160, 4, col) + T(p[0], p[1] + 2, lab, 14, INK, 'middle', BODY, 700) + T(p[0], p[1] + 18, sub, 10, SOFT, 'middle', MONO)
s = node(Pq, 'question', 'from the member of staff', INK) + node(Pc, 'context', 'the retrieved sections', BLUE) + node(Pa, 'answer', 'written by the model', PURP)
s += G_(1, L(Pq[0] - 50, Pq[1] + 28, Pc[0] + 20, Pc[1] - 28, BLUE, 2.4) + T(118, 140, 'context relevance', 12.5, BLUE, 'middle', BODY, 700) + T(118, 156, 'retrieval', 10, SOFT, 'middle', MONO))
s += G_(2, L(Pc[0] + 82, Pc[1], Pa[0] - 82, Pa[1], GREEN, 2.4) + T(280, 240, 'faithfulness', 12.5, GREEN, 'middle', BODY, 700) + T(280, 268, 'generation', 10, SOFT, 'middle', MONO))
s += G_(3, L(Pq[0] + 50, Pq[1] + 28, Pa[0] - 20, Pa[1] - 28, AMBER, 2.4) + T(444, 140, 'answer relevance', 12.5, AMBER, 'middle', BODY, 700) + T(444, 156, 'the whole pipeline', 10, SOFT, 'middle', MONO))
add(slide('Evaluation', 'Evaluation &#183; RAG triad &#183; Concept', 'Concept',
          'Three scores, one for each link between question, context and answer. None needs a reference answer, so they can run on live traffic.',
          '<div class="cols"><div class="colstack">' + fig(svg(560, 286, s, 'The RAG triad: question, context, answer'), 'which stage each score watches') +
          '</div><div class="colstack">'
          '<div class="card sm" data-at="1"><h3>Context relevance</h3><p>Does the retrieved context bear on the question? Here: the reranker&#8217;s score for the best of the three sections.</p></div>'
          '<div class="card sm g" data-at="2"><h3>Faithfulness</h3><p>Is the answer supported by the context? Here: the share of the answer&#8217;s claims that a context sentence entails.</p></div>'
          '<div class="card sm o" data-at="3"><h3>Answer relevance</h3><p>Does the answer address the question? Here: the model writes the question the answer replies to, and it is compared with the real one by embedding similarity.</p></div>'
          '<div class="callout sm" data-at="3">Each score is a model&#8217;s opinion. On the test bed the truth is known, so each can be checked.</div>'
          '</div></div>' + bld(3), 'build'))

# ================================================================ RAG triad: example
TA = TRI['auc']; TG = TRI['groups']
items = [('context relevance', round(TA['cr'], 3), BLUE, 'against: right section in the top 3'),
         ('faithfulness', round(TA['fa'], 3), GREEN, 'against: answer not graded wrong'),
         ('answer relevance', round(TA['ar'], 3), AMBER, 'against: answer graded correct')]
gm = TG['wrong, right section missing']; gr = TG['wrong, right section retrieved']; gc = TG['correct']
tx = TRI['ex']
add(slide('Evaluation', 'Evaluation &#183; RAG triad &#183; Example', 'Example',
          f"The triad on the {NP} test-bed answers. Context relevance and faithfulness track the truth (AUC {TA['cr']:.2f} and {TA['fa']:.2f}). Answer relevance does not ({TA['ar']:.2f}).",
          '<div class="cols"><div class="colstack" style="flex:1.15">' +
          fig(hbars(items, w=660, label_w=170, bar_h=30, gap=14, maxv=1.75, valfmt=lambda v: f'{v:.3f}', label='AUC of each triad score against the truth it stands in for'), 'AUC of each score against the truth it stands in for &#183; 0.5 is a coin toss') +
          table(['answers', 'n', 'context relevance', 'faithfulness', 'answer relevance'],
                [[g, d['n'], f"{d['cr']:.2f}", f"{d['fa']:.2f}", f"{d['ar']:.2f}"] for g, d in TG.items()], 'd sm free') +
          '</div><div class="colstack">'
          f'<div class="callout good sm"><b>The pair locates the fault.</b> Wrong with the right section missing: context relevance {gm["cr"]:.2f}, faithfulness {gm["fa"]:.2f}. Wrong with the section present: {gr["cr"]:.2f} and {gr["fa"]:.2f}. Correct: {gc["cr"]:.2f} and {gc["fa"]:.2f}.</div>'
          f'<div class="callout warn sm"><b>Answer relevance is flat.</b> {gc["ar"]:.2f} for correct answers and {gm["ar"]:.2f} for wrong ones. A wrong daily limit is still an answer about the daily limit.</div>'
          f'<div class="callout sm"><b>Its lowest score:</b> the answer &#8220;{esc(tx[1])}&#8221; led the model to write back &#8220;{esc(tx[2])}&#8221;.</div>'
          f'<div class="callout sm"><b>Faithfulness is strict.</b> Correct answers average {gc["fa"]:.2f}, since split claims fail to find a sentence. Use it to watch for changes over time.</div>'
          '</div></div>'))
