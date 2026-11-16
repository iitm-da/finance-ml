"""LLM session, part 1: cover, plan, the model, retrieval-augmented generation."""
from common import *

A = []
add = A.append
GEN = RG['gen']; RET = RG['ret']; RK = ['BM25', 'dense', 'hybrid', 'hybrid + reranker']
RCOL = {'BM25': AMBER, 'dense': BLUE, 'hybrid': PURP, 'hybrid + reranker': GREEN}
CB = GEN['closed book']; ST = GEN['+ strict prompt']; LAD = EV['ladder']
B15 = f"{RG['n_llm'] / 1e9:.1f}"
NQ = RG['n_ans'] + RG['n_una']
FS = PR['so']['4 + field definitions']

# ================================================================ cover
add(f'''
<section class="slide cover" data-sec="Cover">
  <div class="coverpat">
    <div><span class="h-dim">closed book      correct </span><span class="h-bad">{pct(CB['ans']['correct'], 0)}</span></div>
    <div><span class="h-dim">with retrieval   correct </span><span class="h-val">{pct(ST['ans']['correct'], 0)}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">wrong answers, plain RAG   </span><span class="h-bad">{pct(LAD['RAG, dense, top 3']['wrong'], 0)}</span></div>
    <div><span class="h-dim">wrong answers, guarded     </span><span class="h-val">{pct(LAD['+ number check on the output']['wrong'], 0)}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">sentiment, zero-shot   macro-F1 </span><span class="h-bad">{PF['prompt']['zero-shot, 1.5B model']:.3f}</span></div>
    <div><span class="h-dim">sentiment, LoRA        macro-F1 </span><span class="h-val">{PF['llm']['LoRA, rank 8, 3,000 labels']['f1']:.3f}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">LoRA trains </span><span class="h-val">{PF['llm']['LoRA, rank 8, 3,000 labels']['trainable'] / PF['llm']['LoRA, rank 8, 3,000 labels']['total']:.2%}</span><span class="h-dim"> of the weights</span></div>
  </div>
  <div class="kicker">Machine Learning for Finance</div>
  <h1>LLMs in Depth</h1>
  <div class="pt">Retrieval, evaluation, prompting, fine-tuning</div>
  <div class="rule"></div>
  <p class="sub">A {B15}-billion-weight open model answers questions on a bank&#8217;s internal knowledge base. Every claim about
  retrieval, hallucination, prompts and LoRA is a number measured on it.</p>
  <div class="by"><span>Instructor</span>Dr. Arun B Ayyar</div>
  <table class="meta">
    <tr><td>Duration</td><td>90 minutes</td></tr>
    <tr><td>Model</td><td>Qwen2.5-1.5B-Instruct &#183; runs on a free Colab GPU</td></tr>
    <tr><td>Data</td><td>fictional bank &#183; {RG['n_docs']} documents &#183; {NQ} questions with known answers</td></tr>
    <tr><td>Code</td><td>llms_in_depth.ipynb</td></tr>
  </table>
</section>
''')

# ================================================================ plan
blocks = [('RAG', 30, BLUE, ['closed book', 'retrieval, four ways', 'chunks and prompts']),
          ('Evaluation and hallucination', 20, RED, ['grading answers', 'four detectors', 'mitigation, step by step']),
          ('Prompt engineering', 20, PURP, ['structured output', 'wording and examples', 'hidden instructions']),
          ('LoRA and fine-tuning', 20, GREEN, ['what LoRA trains', 'prompt, LoRA or full'])]
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
          'Four blocks on one small model and one fictional bank. Every number comes from the notebook, which needs no API key.',
          fig(svg(1120, 172, s, 'Session plan'), 'minutes from the start') +
          f'<div class="cols nat"><div class="callout"><b>Why a small model.</b> {B15} billion weights fit on a laptop GPU, so every experiment can be rerun in the room. A larger model moves the levels. The way each number is measured stays the same.</div>'
          f'<div class="callout"><b>Why a fictional bank.</b> The model cannot have seen "Northmere Bank" in training. Any correct answer about it has to come from the documents handed to the model.</div></div>'
          '<div class="callout warn sm">The order differs from the printed syllabus: LoRA comes last, because it is compared with the prompting results that precede it.</div>'))

# ================================================================ the model: one token at a time (build)
NX = RG['next']
def tokbars(rows, col, w=520, title=''):
    s = ''; mx = max(p for _, p in rows)
    for i, (t, p) in enumerate(rows[:7]):
        y = 6 + i * 27
        s += T(70, y + 16, repr(t.strip()) if t.strip() else '" "', 12.5, INK, 'end', MONO, 600)
        s += Rect(80, y, max(2, (w - 170) * p / max(mx, .01)), 20, col) + T(86 + (w - 170) * p / max(mx, .01), y + 15, f'{p:.3f}', 11.5, INK, 'start', MONO, 700)
    return svg(w, 6 + 7 * 27, s, title)
bx = [('prompt', 'tokens so far', INK), ('model', f'{B15} billion weights', BLUE), ('probabilities', 'one per vocabulary token', PURP), ('pick one', 'most likely, or sampled', AMBER)]
s = ''; x = 30; bw = 200; gp = 60
for i, (a, b, c) in enumerate(bx):
    piece = box(x, 20, bw, 58, a, b, c) + (arrow(x - gp + 4, 49, x - 4, 49) if i else '')
    s += G_(i, piece) if i else piece
    x += bw + gp
s += G_(3, f'<path d="M{30 + 3 * (bw + gp) + bw / 2:.0f},78 L{30 + 3 * (bw + gp) + bw / 2:.0f},104 L130,104 L130,84" fill="none" stroke="{SOFT}" stroke-width="1.6" stroke-dasharray="5 4"/>' +
        T(560, 99, 'append the token and run again', 11.5, INK, 'middle', MONO, 700))
add(slide('Model', 'Model &#183; Concept', 'Concept',
          'A language model does one thing: given the tokens so far, it gives a probability to every possible next token. Text is that step in a loop.',
          fig(svg(1140, 112, s, 'The generation loop'), 'one step of generation') +
          '<div class="cols3">'
          '<div class="card sm" data-at="1"><h3>A decoder</h3><p>The attention of this morning, with each token allowed to see only the tokens before it. Trained on one task: predict the next token of text from the web.</p></div>'
          '<div class="card sm" data-at="2"><h3>Probabilities</h3><p>The output is a list of about 150,000 probabilities that add to 1. The model has no separate store of facts and no notion of true.</p></div>'
          '<div class="card sm o" data-at="3"><h3>Temperature</h3><p>At temperature 0 the most likely token is taken every time and the answer is repeatable. Above 0 the next token is drawn at random by probability.</p></div>'
          '</div>' + bld(3), 'build'))

# ================================================================ data: the knowledge base
ks = RG['kb_sample']
qrows = [[esc(q['question']), q['kind'], esc(', '.join(q['answers'][:1])) or '&#8212;'] for q in RG['q_sample']]
add(slide('RAG', 'RAG &#183; Data', 'A fictional bank&#8217;s knowledge base',
          f"{RG['n_docs']} internal documents in {RG['n_kb']} sections, {comma(RG['kb_words'])} words. {RG['n_ans']} questions have an answer in one section; {RG['n_una']} have no answer anywhere.",
          '<div class="cols"><div class="colstack">' +
          f'<div class="outlbl">{ks["id"]} &#183; {esc(ks["doc"])} &#183; {esc(ks["section"])}</div><div class="out sm" style="white-space:normal">{esc(ks["text"])}</div>' +
          '<div class="kv k3">' + tile('sections', RG['n_kb']) + tile('answerable', RG['n_ans'], 'ok') + tile('unanswerable', RG['n_una'], 'bad') + '</div>' +
          '</div><div class="colstack" style="flex:1.25">' + table(['question', 'kind', 'accepted'], qrows, 'd sm free') +
          f'<div class="callout sm"><b>Three kinds of answerable question.</b> {RG["kinds"]["direct"]} share words with the section, {RG["kinds"]["paraphrase"]} say the same thing in other words, {RG["kinds"]["code"]} name the document by its code.</div>'
          '<div class="callout sm"><b>Grading.</b> Correct: the answer holds the accepted value. Abstained: it says the knowledge base has no answer. Wrong: anything else.</div>'
          '</div></div>'))

# ================================================================ closed book
rows = [[esc(q), esc(a[:120]) + ('&#8230;' if len(a) > 120 else ''), esc(', '.join(k[:1])) or 'no answer exists'] for q, a, k in RG['closed_ex'][:4]]
add(slide('RAG', 'RAG &#183; Example', 'Example',
          f"The model alone, with no documents. It answers {pct(CB['ans']['correct'], 0)} of the answerable questions correctly and gives a wrong answer to {pct(CB['ans']['wrong'], 0)} of them.",
          '<div class="cols"><div class="colstack" style="flex:1.7">' + table(['question', 'the model said', 'accepted'], rows, 'd sm free') +
          '</div><div class="colstack">' + fig(tokbars(NX['closed'], RED, 440, 'Next-token probabilities without the document'),
              'next token after "... can withdraw up to GBP"', 'No document in the prompt. The true value is 450.') +
          f'<div class="callout warn sm"><b>It does not know that it does not know.</b> On the {RG["n_una"]} questions with no answer it abstained on {pct(CB["una"]["abstained"], 0)} and made something up for the rest.</div>'
          '</div></div>'))

# ================================================================ RAG pipeline (build)
s = T(10, 16, 'once, offline', 11, SOFT, 'start', MONO, 700) + T(10, 118, 'for every question', 11, SOFT, 'start', MONO, 700)
off = [('documents', f'{RG["n_docs"]} policies and guides', INK), ('chunks', f'{RG["n_kb"]} sections', SLATE), ('index', 'words and vectors', BLUE)]
x = 10
for i, (a, b, c) in enumerate(off):
    piece = box(x, 26, 190, 56, a, b, c) + (arrow(x - 36, 54, x - 4, 54) if i else '')
    s += G_(1, piece); x += 230
on = [('question', 'from a member of staff', INK), ('retrieve', 'top k sections', BLUE), ('prompt', 'sections + question', PURP), ('model', 'writes the answer', AMBER), ('answer', 'with its sources', GREEN)]
x = 10
for i, (a, b, c) in enumerate(on):
    k = [0, 2, 3, 4, 4][i]
    piece = box(x, 128, 190, 56, a, b, c) + (arrow(x - 36, 156, x - 4, 156) if i else '')
    s += G_(k, piece) if k else piece
    x += 230
s += G_(2, arrow(10 + 2 * 230 + 95, 82, 10 + 230 + 95, 126, BLUE))
add(slide('RAG', 'RAG &#183; Concept', 'Concept',
          'Retrieval-augmented generation: find the sections that bear on the question, paste them into the prompt, and let the model answer from them.',
          fig(svg(1140, 194, s, 'The retrieval-augmented generation pipeline'), 'two paths: indexing and answering') +
          '<div class="cols4">'
          '<div class="card sm" data-at="1"><h3>Index</h3><p>Split the documents into chunks and store each one so that it can be searched. Rerun when a document changes.</p></div>'
          '<div class="card sm" data-at="2"><h3>Retrieve</h3><p>Score every chunk against the question and keep the best k. Nothing else reaches the model.</p></div>'
          '<div class="card sm" data-at="3"><h3>Prompt</h3><p>The chunks become part of the prompt. The model weights are untouched; a policy change needs a re-index, no training.</p></div>'
          '<div class="card sm g" data-at="4"><h3>Answer</h3><p>The model reads the chunks and writes one sentence. The chunk ids travel with the answer so a person can check it.</p></div>'
          '</div>' + bld(4), 'build'))

# ================================================================ retrieval: four ways
add(slide('RAG', 'RAG &#183; Mechanism', 'Mechanism',
          'Four ways to rank the sections for a question. Each fixes a failure of the one before.',
          '<div class="cols4">'
          '<div class="card o"><h3>BM25</h3><p>Scores a section by the words it shares with the question, with rare words counting more. Exact on codes and product names. Blind to synonyms: "cash machine" does not match "ATM".</p></div>'
          '<div class="card"><h3>Dense</h3><p>An embedding model turns question and section into vectors; rank by the angle between them. Matches meaning across different words. Can drift to a section on the right topic for the wrong product.</p></div>'
          '<div class="card"><h3>Hybrid</h3><p>Run both and merge the two rankings. A section ranked high by either method ends near the top.</p></div>'
          '<div class="card g"><h3>Reranker</h3><p>A second, slower model reads the question and one section together and scores the pair. Run on the top 20 only.</p></div>'
          '</div>'
          '<div class="cols"><div class="colstack">' +
          out("""score(section) = 1 / (60 + rank in BM25)
               + 1 / (60 + rank in dense)""", '', 'hybrid: reciprocal rank fusion') +
          '</div><div class="colstack">'
          '<div class="callout sm"><b>Why ranks and not scores.</b> A BM25 score and a cosine similarity are on different scales. Their ranks can be added.</div>'
          '</div></div>'))

# ================================================================ retrieval: recall by question kind
KINDS = ['direct', 'paraphrase', 'code']
s = ''; gw = 330; x0 = 60; H = 270; ph = H - 70
for v in (0, .25, .5, .75, 1):
    y = 20 + (1 - v) * ph
    s += L(x0, y, x0 + gw * 3, y, HAIR, 1) + T(x0 - 6, y + 4, f'{v:.2f}', 10, SOFT, 'end', MONO)
for i, kd in enumerate(KINDS):
    gx = x0 + i * gw + 30
    for j, r in enumerate(RK):
        v = RET[r][kd]['r3']; h = v * ph
        s += Rect(gx + j * 66, 20 + ph - h, 58, h, RCOL[r]) + T(gx + j * 66 + 29, 20 + ph - h - 5, f'{v:.2f}', 11, INK, 'middle', MONO, 700)
    s += T(gx + 132, H - 30, f'{kd} questions', 12.5, INK, 'middle', MONO, 700) + T(gx + 132, H - 14, f'{RG["kinds"][kd]} questions', 10, SOFT, 'middle', MONO)
leg = ' &#160; '.join(f'<span class="tag" style="background:{RCOL[r]}">&#160;</span>{r}' for r in RK)
worst_b = min(KINDS, key=lambda k: RET['BM25'][k]['r3']); worst_d = min(KINDS, key=lambda k: RET['dense'][k]['r3'])
add(slide('RAG', 'RAG &#183; Example', 'Example continued 1',
          f"Recall@3: the share of questions whose answering section is among the top three. BM25 is weakest on {worst_b} questions ({RET['BM25'][worst_b]['r3']:.2f}), dense on {worst_d} questions ({RET['dense'][worst_d]['r3']:.2f}).",
          fig(svg(1120, H, s, 'Recall at 3 by retriever and question kind'), 'recall@3', leg) +
          '<div class="kv">' + ''.join(tile(f'{r}, all questions', f"{RET[r]['all']['r3']:.3f}", 'ok' if r == max(RK, key=lambda q: RET[q]['all']['r3']) else '') for r in RK) + '</div>'
          f'<div class="callout sm"><b>Recall@1 for the best retriever is {max(RET[r]["all"]["r1"] for r in RK):.3f}.</b> The right section is first most of the time, and passing three sections covers most of the rest.</div>'))

# ================================================================ widget: retrieval explorer
btns = ''.join(f'<button data-v="{r}"{" class=on" if r == "BM25" else ""}>{r}</button>' for r in RK)
add(slide('RAG', 'RAG &#183; Example', 'Retrieval, question by question',
          f"{len(RG['explore'])} of the questions. For each: the top four sections under each retriever, the closed-book answer and the answer with retrieval.",
          f'<div class="wctl"><button class="wbtn" id="rx-prev">&#8249; previous</button><button class="wbtn" id="rx-next">next &#8250;</button>'
          f'<div class="seg" id="rx-rank">{btns}</div>{JSB}</div>'
          '<div class="cols"><div class="colstack">'
          '<div class="out" id="rx-q" style="min-height:74px;white-space:normal"></div>'
          '<div class="outlbl">top four sections</div><div class="out sm" id="rx-top" style="min-height:108px"></div>'
          '</div><div class="colstack">'
          '<div class="outlbl">closed book</div><div class="out sm" id="rx-closed" style="min-height:66px;white-space:normal"></div>'
          '<div class="outlbl">with retrieval, strict prompt</div><div class="out sm" id="rx-rag" style="min-height:66px;white-space:normal"></div>'
          '</div></div>' + jdata('rx-data', {'q': RG['explore'], 'rk': RK})))

# ================================================================ chunk size
CH = RG['chunk']
fixed = [c for c in CH if c['words'] != 'by section']; sec_ = next(c for c in CH if c['words'] == 'by section')
rows = [[f"{c['words']} words" + (f", {c['overlap']} overlap" if c['overlap'] else ''), c['chunks'], c['context words'], f"<b>{pct(c['served'])}</b>"] for c in fixed]
rows.append(['<b>one chunk per section</b>', sec_['chunks'], sec_['context words'], f"<b>{pct(sec_['served'])}</b>"])
bestf = max(fixed, key=lambda c: c['served']); small_ = fixed[0]
items = [((f"{c['words']}w" + (f"+{c['overlap']}" if c['overlap'] else '')), round(c['served'], 3), BLUE if c['overlap'] == 0 else PURP, f"{c['context words']} words to the model") for c in fixed]
items.append(('by section', round(sec_['served'], 3), GREEN, f"{sec_['context words']} words to the model"))
add(slide('RAG', 'RAG &#183; Example', 'Example continued 2',
          f"Chunk size. Each document is cut into windows of a fixed number of words; a question is served if one of the top three chunks holds the whole answering sentence.",
          '<div class="cols"><div class="colstack" style="flex:1.3">' +
          fig(hbars(items, w=640, label_w=110, bar_h=24, gap=9, maxv=1.12, valfmt=lambda v: f'{v:.3f}', label='Questions served by chunk size'),
              'share of answerable questions served (dense retrieval, top 3)') +
          '</div><div class="colstack">' +
          f'<div class="callout warn"><b>Small chunks cut sentences.</b> At {small_["words"]} words a chunk, {pct(1 - small_["served"], 0)} of questions are not served: the answering sentence is split across two chunks, or the chunk lost the product name that sat in the sentence before.</div>'
          f'<div class="callout"><b>Large chunks cost context.</b> Three chunks of {fixed[-1]["words"]} words send {fixed[-1]["context words"]} words to the model for a one-line answer, and the model has to find the line.</div>'
          f'<div class="callout good sm"><b>Cut where the author cut.</b> Splitting at section headings serves {pct(sec_["served"])} with {sec_["context words"]} words of context.</div>'
          '</div></div>'))

# ================================================================ the prompt
add(slide('RAG', 'RAG &#183; Mechanism', 'Mechanism continued 1',
          'The prompt has two parts: an instruction that stays the same for every question, and the retrieved sections with the question.',
          '<div class="cols"><div class="colstack">' +
          '<div class="outlbl">system prompt</div>' + f'<div class="out sm" style="white-space:normal">{esc(RG["sys_strict"])}</div>' +
          '<div class="outlbl">user message</div>' + f'<div class="out xs" style="white-space:pre-wrap">{esc(RG["prompt_ex"])}</div>' +
          '</div><div class="colstack">' + fig(tokbars(NX['rag'], GREEN, 440, 'Next-token probabilities with the document'),
              'next token after "... can withdraw up to GBP", section in the prompt', 'Same question as before. The probability has moved to the number on the page.') +
          '<div class="callout sm"><b>Three instructions do the work.</b> Use only the context. Quote the figure as written. Say a fixed phrase when the context has no answer, so that code can detect it.</div>'
          '</div></div>'))

# ================================================================ answers by pipeline
order = ['closed book', 'RAG, dense, top 3', 'RAG, hybrid + reranker, top 3', '+ strict prompt', 'strict, top 1', 'strict, top 5', 'strict, the right section given']
short = {'closed book': 'closed book', 'RAG, dense, top 3': 'dense, top 3, plain prompt', 'RAG, hybrid + reranker, top 3': 'reranked, top 3, plain prompt',
         '+ strict prompt': 'reranked, top 3, strict prompt', 'strict, top 1': 'reranked, top 1, strict', 'strict, top 5': 'reranked, top 5, strict',
         'strict, the right section given': 'right section handed over'}
ORA = GEN['strict, the right section given']
add(slide('RAG', 'RAG &#183; Example', 'Example continued 3',
          f"Seven pipelines on the same questions. With reranked retrieval and the strict prompt the model is correct on {pct(ST['ans']['correct'], 0)} of answerable questions and abstains on {pct(ST['una']['abstained'], 0)} of unanswerable ones.",
          '<div class="cols nat"><div class="colstack">' + fig(stacked([(short[k], GEN[k]['ans']) for k in order], w=560, label_w=240, bar_h=24, gap=9, label='Answerable questions by pipeline'),
              f'{RG["n_ans"]} answerable questions', GLEG) +
          '</div><div class="colstack">' + fig(stacked([(short[k], GEN[k]['una']) for k in order], w=560, label_w=240, bar_h=24, gap=9, label='Unanswerable questions by pipeline'),
              f'{RG["n_una"]} unanswerable questions: abstaining is the right answer') +
          '</div></div>'
          '<div class="cols nat">'
          f'<div class="callout"><b>The prompt decides the unanswerable questions.</b> With a plain prompt the model abstains on {pct(GEN["RAG, hybrid + reranker, top 3"]["una"]["abstained"], 0)} of them; with the strict prompt, {pct(ST["una"]["abstained"], 0)}.</div>'
          f'<div class="callout"><b>The ceiling is the reader.</b> Handed exactly the right section, this model is correct on {pct(ORA["ans"]["correct"], 0)}. Retrieval explains the gap from {pct(ST["ans"]["correct"], 0)} to there.</div></div>'))

# ================================================================ test: where the errors come from
E = RG['err']   # rows correct/abstained/wrong, cols retrieved False/True
tot = sum(sum(r) for r in E)
rows = [[g, E[i][1], E[i][0]] for i, g in enumerate(('correct', 'abstained', 'wrong'))]
bad = [[esc(q), esc(a[:110]), esc(', '.join(k[:1]))] for q, a, k in RG['bad'][:3]]
add(slide('RAG', 'RAG &#183; Test', 'Test',
          f"The {RG['n_ans']} answerable questions under the strict prompt, split by whether the answering section was among the three retrieved.",
          '<div class="cols"><div class="colstack">' +
          table(['graded', 'right section retrieved', 'right section missing'], rows, 'd sm free') +
          f'<div class="callout"><b>Two different failures.</b> {E[2][0] + E[1][0]} questions failed because retrieval missed the section; of those the model abstained on {E[1][0]} and answered wrongly on {E[2][0]}. '
          f'{E[2][1]} were answered wrongly with the right section in front of the model.</div>'
          f'<div class="callout warn sm"><b>Archived sections.</b> {RG["stale"][0]} questions had a superseded section among the top three, and {RG["stale"][1]} of those were answered wrongly.</div>'
          '</div><div class="colstack" style="flex:1.3">' + table(['question', 'answer graded wrong', 'accepted'], bad, 'd sm free') +
          '<div class="callout sm"><b>Fix the stage that failed.</b> A retrieval miss needs a better index or query. A misread section needs a better prompt or a stronger model. The grade alone does not say which.</div>'
          '</div></div>'))

# ================================================================ drawbacks
add(slide('RAG', 'RAG &#183; Drawbacks', 'Drawbacks',
          'What this knowledge base leaves out, and what a production one has to handle.',
          '<div class="cols3">'
          '<div class="card big r"><h3>Stale and conflicting documents</h3><p>The index returns whatever is in it. Superseded versions need a status field and a filter at query time, or the model will quote the old limit with full confidence.</p></div>'
          '<div class="card big o"><h3>Permissions</h3><p>Retrieval must apply the access rights of the person asking. A chunk the user may not read must never enter the prompt, because the model will repeat it.</p></div>'
          '<div class="card big"><h3>Tables and scans</h3><p>Fee tables and scanned forms lose their meaning when flattened to text. They need their own extraction step before indexing.</p></div>'
          '</div>'
          '<div class="cols nat"><div class="callout"><b>Questions that need several sections.</b> "Which account has the highest cash machine limit?" has no single answering section. Top-3 retrieval cannot see all five product guides.</div>'
          f'<div class="callout"><b>Cost and delay.</b> Every question now runs an embedding model, a reranker and a prompt of several hundred tokens. On this GPU the {NQ} questions took {ST["secs"]:.0f} seconds with retrieval and {CB["secs"]:.0f} without.</div></div>'))
