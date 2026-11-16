"""NLP session, part 4: document classification and information extraction, summary, references."""
from common import *

D = []
add = D.append
ACC = DC['acc']; NOISE = DC['noise']; TYPES = DC['types']
FIELDS = IE['fields']; RU, QA = IE['m']['rules'], IE['m']['qa']
FCOL = {'borrower': BLUE, 'lender': PURP, 'principal': AMBER, 'rate': RED, 'maturity': SLATE, 'law': GREEN}
FNAME = {'borrower': 'borrower', 'lender': 'lender', 'principal': 'principal', 'rate': 'interest rate', 'maturity': 'final repayment date', 'law': 'governing law'}
nice = lambda t: t.replace('_', ' ')
mean = lambda d: sum(d.values()) / len(d)

# ================================================================ concept: the pipeline (build)
bx = [('scan', 'image of a page', INK), ('text', 'OCR, with errors', SLATE), ('classify', 'which document type', BLUE),
      ('extract', 'fields and their scores', PURP), ('validate', 'rules on the values', AMBER), ('accept or review', 'by confidence', GREEN)]
s = ''; x = 10; bw = 160; gp = 28
for i, (a, b, c) in enumerate(bx):
    piece = box(x, 30, bw, 62, a, b, c)
    if i:
        piece += arrow(x - gp + 3, 61, x - 3, 61)
    k = max(0, i - 1)
    s += G_(k, piece) if k else piece
    x += bw + gp
add(slide('Documents', 'Documents &#183; Concept', 'Concept',
          'A document pipeline is a chain of small decisions. Each press adds a stage, and each stage can be measured on its own.',
          fig(svg(1140, 104, s, 'Stages of a document pipeline'), 'from a scanned page to a record') +
          '<div class="cols4">'
          '<div class="card sm" data-at="1"><h3>Classify</h3><p>Loan agreement, guarantee, statement or complaint? The type decides which fields to look for and which team owns the page.</p></div>'
          '<div class="card sm" data-at="2"><h3>Extract</h3><p>Pull the borrower, the amount, the rate and the dates out of the text, each with a score.</p></div>'
          '<div class="card sm o" data-at="3"><h3>Validate</h3><p>Checks that need no model: the date parses, the amount is positive, the borrower is a customer on file.</p></div>'
          '<div class="card sm g" data-at="4"><h3>Accept or review</h3><p>Confident, valid values go straight through. The rest go to a person, and their corrections become training data.</p></div>'
          '</div>' + bld(4), 'build'))

# ================================================================ classification: data
smp = DC['sample']
def pagecard(t):
    return f'<div class="card sm"><h3>{nice(t)}</h3><p style="font-size:12.5px;line-height:1.5">{esc(smp[t][:330])}{"&#8230;" if len(smp[t]) > 330 else ""}</p></div>'
add(slide('Classification', 'Classification &#183; Data', 'Pages of eight document types',
          f"{comma(DC['n_train'])} training pages and {DC['n_test']} test pages, simulated. A page is a few paragraphs, and the three legal types share boilerplate.",
          '<div class="cols"><div class="colstack">' + pagecard('loan_agreement') + pagecard('guarantee') +
          '</div><div class="colstack">' + pagecard('account_statement') +
          '<div class="outlbl">the eight types</div><div class="out sm">' + '\n'.join('   '.join(f'{nice(t):18s}' for t in TYPES[i:i + 2]) for i in range(0, 8, 2)) + '</div>' +
          '<div class="callout sm"><b>Pages, not documents.</b> Scans arrive as loose pages. A page from the middle of a loan agreement may hold only a notices clause and a governing-law clause, which a guarantee has too.</div>'
          '</div></div>'))

# ================================================================ classification: accuracy against scan damage
ax = Ax(640, 320, (-0.01, NOISE[-1] + 0.012), (min(0.3, min(min(v.values()) for v in ACC.values()) - 0.05), 1.02), pad=(46, 14, 14, 34))
yt = [v / 10 for v in range(int(ax.yr[0] * 10) + 1, 11)]
s = ax.frame(yticks=yt, xticks=[(p, f'{p:.0%}') for p in NOISE], yfmt=lambda v: f'{v:.1f}', ylab='accuracy', xlab='share of characters damaged')
MC = {'word TF-IDF': BLUE, 'character n-gram TF-IDF': GREEN, 'embeddings': PURP}
order = sorted(ACC, key=lambda k: -ACC[k][str(NOISE[-1])])
for k in ACC:
    ys = [ACC[k][str(p)] for p in NOISE]
    s += ax.line(NOISE, ys, MC[k], 2.6)
    for p, v in zip(NOISE, ys):
        s += ax.dot(p, v, MC[k], 4)
for r_, k in enumerate(order):
    s += T(ax.l + 12, ax.t + ax.ph - 14 - (len(order) - 1 - r_) * 17, f'{k}  {ACC[k]["0"]:.3f} → {ACC[k][str(NOISE[-1])]:.3f}', 11.5, MC[k], 'start', MONO, 700)
clean_best = max(ACC, key=lambda k: ACC[k]['0']); noisy_best = order[0]; p10 = '0.1'
noisy = DC['noisy']
add(slide('Classification', 'Classification &#183; Example', 'Example',
          f"Three classifiers trained on clean text, tested on pages with scan damage. On clean pages all three score between {min(ACC[k]['0'] for k in ACC):.3f} and {max(ACC[k]['0'] for k in ACC):.3f}; "
          f"at {NOISE[-1]:.0%} damage they range from {min(ACC[k][str(NOISE[-1])] for k in ACC):.3f} to {max(ACC[k][str(NOISE[-1])] for k in ACC):.3f}.",
          '<div class="cols"><div class="colstack" style="flex:1.35">' + fig(svg(640, 320, s, 'Accuracy against scan damage'), f'accuracy on the {DC["n_test"]} test pages') +
          '</div><div class="colstack">' +
          '<div class="outlbl">a page with 10% of characters damaged</div>' + f'<div class="out sm" style="white-space:normal">{esc(noisy["0.1"])}</div>' +
          f'<div class="callout"><b>{noisy_best.capitalize()} holds up best.</b> ' +
          ('A damaged word still shares most of its character runs with the clean word, so the features survive.' if noisy_best.startswith('character') else
           'The model has seen misspelled text during pretraining and maps a damaged word near the clean one.' if noisy_best == 'embeddings' else
           'Enough whole words survive on a page for the word counts to carry the class.') + '</div>'
          '<div class="callout warn sm"><b>Test on what production sends.</b> A score on clean digital text says little about pages that have been through a scanner and OCR.</div>'
          '</div></div>'))

# ================================================================ classification: which pages are wrong
rows = [[nice(a), nice(b), esc(t[:150]) + ('&#8230;' if len(t) > 150 else '')] for a, b, t in DC['wrong']][:5]
cm = DC['cm']
pairs = sorted(((cm[i][j], TYPES[i], TYPES[j]) for i in range(8) for j in range(8) if i != j and cm[i][j]), reverse=True)[:4]
add(slide('Classification', 'Classification &#183; Test', 'Test',
          f"Word TF-IDF misclassifies {DC['n_wrong']} of {DC['n_test']} clean test pages. The errors sit where two types share paragraphs.",
          '<div class="cols"><div class="colstack" style="flex:1.8">' + table(['label', 'predicted', 'page'], rows, 'd sm free') +
          '</div><div class="colstack">' + '<div class="outlbl">largest confusions on clean pages</div><div class="out sm">' +
          '\n'.join(f'{n:3d}  {nice(a)} read as {nice(b)}' for n, a, b in pairs) + '</div>' +
          '<div class="callout"><b>Some pages cannot be classified alone.</b> A page of boilerplate belongs to whatever document it came from. The fix is in the pipeline: classify the page together with its neighbours.</div>'
          '<div class="callout sm"><b>Look at the wrong pages before changing the model.</b> Five rows of errors showed a data problem that no larger model would fix.</div>'
          '</div></div>'))

# ================================================================ extraction: data
L0 = IE['loan0']; txt = esc(L0['text'])
for f in FIELDS:
    v = esc(L0[f])
    txt = txt.replace(v, f'<span style="background:{FCOL[f]}22;border-bottom:2px solid {FCOL[f]};padding:0 2px">{v}<small style="font-family:var(--mono);font-size:9px;font-weight:700;color:{FCOL[f]};margin-left:3px">{f}</small></span>', 1)
import re as _re
txt = _re.sub(r'(arrangement fee of GBP [\d,]+|default interest of [\d.]+%|Repayments begin on \d+ \w+ \d{4}|first instalment falls due on \d+ \w+ \d{4}|guaranteed by [^(]+)',
              r'<span style="text-decoration:underline wavy #B02F3B">\1</span>', txt)
add(slide('Extraction', 'Extraction &#183; Data', 'Fields to extract from a loan agreement',
          f"{IE['n']} simulated loan agreements, about {IE['words']} words each. Six values are wanted from each; the underlined values look right and are wrong.",
          '<div class="cols"><div class="colstack" style="flex:2.1">' + f'<div class="card" style="font-size:13px;line-height:1.75">{txt}</div>' +
          '</div><div class="colstack">' +
          table(['field', 'value'], [[f'<span class="tag" style="background:{FCOL[f]}">{f}</span>', esc(L0[f])] for f in FIELDS], 'd sm free') +
          '<div class="callout warn sm"><b>Distractors.</b> An arrangement fee is an amount, a default rate is a rate, a first repayment date is a date, a guarantor is a company. "The first amount in the text" is not a rule that works.</div>'
          '<div class="callout sm">Clauses come in a random order and each has three phrasings, one of them much the commonest.</div>'
          '</div></div>'))

# ================================================================ extraction: rules
def fbars(d1, d2=None, w=560, n1='rules', n2='QA model'):
    s = ''; y = 6; bh = 17 if d2 else 26; lw = 166; span = w - lw - 70
    for f in FIELDS:
        s += T(lw - 10, y + (bh * (2 if d2 else 1)) / 2 + 4, FNAME[f], 12, INK, 'end', MONO, 600)
        for d, c in ((d1, AMBER), (d2, GREEN)) if d2 else ((d1, AMBER),):
            s += Rect(lw, y, max(2, span * d[f]), bh - 2, c) + T(lw + span * d[f] + 6, y + bh / 2 + 3, pct(d[f]), 11.5, INK, 'start', MONO, 700)
            y += bh
        y += 10
    return svg(w, y, s, 'Share of agreements with the field extracted correctly')
BV = IE['byv']
r0 = sum(r[3] * r[2] for r in BV if r[1] == 0) / sum(r[2] for r in BV if r[1] == 0)
rx = IE['rx']
add(slide('Extraction', 'Extraction &#183; Example', 'Example',
          f"One regular expression per field, written against the usual wording. They are right on {pct(mean(RU))} of fields: {pct(r0, 0)} on the usual wording and 0% on the other two.",
          '<div class="cols"><div class="colstack" style="flex:1.2">' +
          code("RX = {\n" + '\n'.join(f" '{f}': r'{rx[f]}'," for f in FIELDS).replace('\\d{1,2} [A-Z][a-z]+ \\d{4}', '{DATE}') + "\n}", 'xs') +
          '<div class="callout sm"><b>A rule never returns a wrong value here.</b> It returns the right one or nothing, so every failure is visible and can be sent to a person.</div>'
          '</div><div class="colstack">' + fig(fbars(RU), 'share of agreements with the field right') +
          '<div class="callout warn sm"><b>Coverage is the limit.</b> Each new phrasing needs a new pattern, and the list of phrasings in real contracts has no end.</div>'
          '</div></div>'))

# ================================================================ extraction: question answering
qs = IE['questions']
add(slide('Extraction', 'Extraction &#183; Example', 'Example continued 1',
          f"A question-answering model reads the agreement and one question per field, and returns a span of the text. It has seen no loan agreement, and is right on {pct(mean(QA))} of fields.",
          '<div class="cols"><div class="colstack">' +
          '<div class="outlbl">one question per field</div><div class="out sm">' + '\n'.join(f'{f:10s} {qs[f]}' for f in FIELDS) + '</div>' +
          code("""qa = pipeline('question-answering',
              model='deepset/roberta-base-squad2')
qa(question='Who is the borrower?', context=text)
# {'answer': ..., 'score': ..., 'start': ..., 'end': ...}""", 'xs') +
          f'<div class="callout sm"><b>{IE["t_qa"] / IE["n"] * 1000:.0f} ms an agreement</b> for all six questions on a {"GPU" if IE["dev"] == "cuda" else "CPU"}. The answer is always copied from the text, so it cannot invent a value that is absent from the page.</div>'
          '</div><div class="colstack">' + fig(fbars(RU, QA), 'share of agreements with the field right',
              f'<span class="tag" style="background:{AMBER}">&#160;</span>rules &#160; <span class="tag" style="background:{GREEN}">&#160;</span>question-answering model') +
          '</div></div>'))

# ================================================================ extraction: where each fails
VK = {'borrower': 'parties', 'lender': 'parties', 'principal': 'principal', 'rate': 'rate', 'maturity': 'maturity', 'law': 'law'}
rows = []
for f in FIELDS:
    rr = [r for r in BV if r[0] == f]
    rows.append([f'<span class="tag" style="background:{FCOL[f]}">{f}</span>'] + [f'{pct(r[4], 0)}' for r in rr] + [f'<b>{pct(QA[f], 0)}</b>'])
worst = min(FIELDS, key=lambda f: QA[f])
wr = [[f'<span class="tag" style="background:{FCOL[f]}">{f}</span>', esc(str(t)), esc(str(a)[:44]), f'{sc:.2f}'] for f, t, a, sc in IE['qa_wrong'] if f in (worst, 'rate', 'maturity', 'principal')][:6]
add(slide('Extraction', 'Extraction &#183; Example', 'Example continued 2',
          f"The question-answering model by phrasing, and six of its {IE['n_qa_wrong']} wrong answers. Its weakest field is the {FNAME[worst]} at {pct(QA[worst])}.",
          '<div class="cols"><div class="colstack">' +
          table(['field', 'usual wording', 'second', 'third', 'all'], rows, 'd sm free') +
          '<div class="callout sm"><b>The model does not need the usual wording.</b> Rules score 0% on the second and third phrasings. The model reads them, less well where the clause also holds a distractor.</div>'
          '</div><div class="colstack" style="flex:1.25">' + table(['field', 'truth', 'model answered', 'score'], wr, 'd sm free') +
          '<div class="callout warn sm"><b>A wrong answer looks like a right one.</b> It is a real date or a real amount from the page. Only the score, or a second check, separates them.</div>'
          '</div></div>'))

# ================================================================ widget: confidence threshold
rt = IE['route']
add(slide('Extraction', 'Extraction &#183; Example', 'Accept or review, by score',
          f"Every answer carries a score. Above the threshold the value is accepted unchecked; below it, a person looks. {len(IE['sc']):,} answers, green right and red wrong.",
          '<div class="cols"><div class="colstack" style="flex:1.5"><div class="fig"><div class="cap">answers by score</div>'
          '<svg id="ct-svg" viewBox="0 0 700 300" width="100%" role="img" aria-label="Answers by score with a threshold"></svg></div>'
          f'<div class="wctl"><label class="rg">threshold: <b id="ct-tl">0.50</b><input type="range" id="ct-thr" min="0" max="0.98" step="0.02" value="0.5"/></label>{JSB}</div>'
          '</div><div class="colstack"><div class="out" id="ct-out" style="min-height:112px"></div>'
          '<div class="callout"><b>The threshold is a staffing decision.</b> It trades wrong values in the system against pages on a reviewer&#8217;s desk. The model gives the curve; operations picks the point.</div>'
          '<div class="callout sm"><b>Check the score before trusting it.</b> A threshold works only if wrong answers have lower scores. Plot this histogram on the bank&#8217;s own documents first.</div>'
          '</div></div>' + jdata('ct-data', {'sc': IE['sc']})))

# ================================================================ verdict
add(slide('Documents', 'Documents &#183; Verdict', 'Verdict',
          f"Rules were right on {pct(mean(RU))} of fields and never wrong when they answered. The question-answering model was right on {pct(mean(QA))} and wrong with confidence on some of the rest.",
          '<div class="cols3">'
          '<div class="card big o"><h3>Rules first</h3><p>Where a rule matches, take its value. It is exact, fast and explainable. Count how often no rule matches; that number is the case for a model.</p></div>'
          '<div class="card big"><h3>A model for the rest</h3><p>Send what the rules miss to a model, and keep its score. Fine-tune it on reviewed documents when the score histogram shows right and wrong answers mixed.</p></div>'
          '<div class="card big g"><h3>A person for the uncertain</h3><p>Route by score and by validation checks. Store every correction with the page; those rows are the labelled data for the next model.</p></div>'
          '</div>'
          '<div class="cols nat"><div class="callout warn"><b>Not covered by text models.</b> Tables, stamps, signatures and anything whose meaning is in the layout. Those need a layout-aware model that reads positions on the page as well as words.</div>'
          '<div class="callout"><b>Next session.</b> The same six fields are extracted with a language model and a prompt, and scored against these two results.</div></div>'))
