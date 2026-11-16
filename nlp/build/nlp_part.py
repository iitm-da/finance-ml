# %%
import os, sys, urllib.request

if 'google.colab' in sys.modules:            # Colab ships torch, transformers and scikit-learn
    get_ipython().system('pip -q install sentence-transformers')

REPO = 'https://raw.githubusercontent.com/iitm-da/finance-ml/master/nlp/data/'
os.makedirs('data', exist_ok=True)
for f in ('ner_train.jsonl', 'ner_test.jsonl', 'pages_train.csv', 'pages_test.csv', 'loans.jsonl'):
    if not os.path.exists('data/' + f):      # not run from the repo folder: fetch the simulated files
        urllib.request.urlretrieve(REPO + f, 'data/' + f)

# %%
import json, re, time, warnings, logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
import transformers; transformers.logging.set_verbosity_error(); transformers.logging.disable_progress_bar()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = 'data/'
RES = {}
print('device:', DEV)

# %% [markdown]
# # 1. Attention
#
# DistilBERT (6 layers, 12 heads a layer, 66 million weights) is small enough to open up.
# The first cell shows what the model receives: word pieces, not words.

# %%
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
enc = AutoModel.from_pretrained('distilbert-base-uncased', attn_implementation='eager').eval()

WORDS = ['overdraft', 'remortgage', 'chargeback', 'collateralised', 'EBITDA', 'SONIA', 'bank', 'interest']
pieces = {w: tok.tokenize(w) for w in WORDS}
for w, p in pieces.items():
    print(f'{w:16s}', p)
print('vocabulary size:', tok.vocab_size)

# %% [markdown]
# ## One head, computed by hand
#
# Each token's vector is turned into a query and a key. The score between two tokens is the dot product of one
# token's query with the other's key, divided by the square root of the head size. A softmax over the row turns
# scores into weights that add to 1.

# %%
SENT = 'The bank raised its base rate after inflation rose.'
x = tok(SENT, return_tensors='pt')
T_ = tok.convert_ids_to_tokens(x['input_ids'][0])
with torch.no_grad():
    o = enc(**x, output_attentions=True, output_hidden_states=True)
ATT = torch.stack(o.attentions)[:, 0]          # (6 layers, 12 heads, tokens, tokens)
i_its, i_bank = T_.index('its'), T_.index('bank')

# the head in which "its" looks hardest at "bank"
flat = ATT[:, :, i_its, i_bank]
LYR, HEAD = [int(v) for v in np.unravel_index(int(flat.argmax()), flat.shape)]
print('tokens:', T_)
print(f'layer {LYR + 1}, head {HEAD + 1}: weight from "its" to "bank" = {flat.max():.3f}')

# %%
att = enc.transformer.layer[LYR].attention
h = o.hidden_states[LYR][0]                    # what layer LYR receives: (tokens, 768)
nh, dh = 12, 768 // 12
with torch.no_grad():
    Q = att.q_lin(h).view(-1, nh, dh)[:, HEAD] # (tokens, 64) queries of this head
    K = att.k_lin(h).view(-1, nh, dh)[:, HEAD] # (tokens, 64) keys
scores = (Q @ K.T) / dh ** 0.5                 # scaled dot product
weights = torch.softmax(scores, dim=-1)
assert torch.allclose(weights, ATT[LYR, HEAD], atol=1e-4)      # same numbers the model reports
hand = pd.DataFrame({'token': T_, 'score q.k/8': scores[i_its].numpy().round(2), 'weight': weights[i_its].numpy().round(3)})
hand

# %% [markdown]
# ## The same word in two contexts
#
# After six layers of attention, the vector for "bank" depends on the sentence around it.

# %%
BANKS = [('money', 'The bank raised its base rate after inflation rose.'),
         ('money', 'She moved her savings to another bank last month.'),
         ('money', 'The bank approved the loan within two days.'),
         ('river', 'They walked along the bank of the river at dusk.'),
         ('river', 'The boat was tied to the muddy bank.'),
         ('river', 'Reeds grew thick on the far bank of the stream.')]

def word_vec(sentence, word, layer=-1):
    e = tok(sentence, return_tensors='pt')
    with torch.no_grad():
        hs = enc(**e, output_hidden_states=True).hidden_states
    return hs[layer][0, tok.convert_ids_to_tokens(e['input_ids'][0]).index(word)]

def cos_matrix(layer):
    V = torch.stack([word_vec(s, 'bank', layer) for _, s in BANKS])
    V = V / V.norm(dim=1, keepdim=True)
    return (V @ V.T).numpy()

C0, C6 = cos_matrix(0), cos_matrix(-1)
same = [(i, j) for i in range(6) for j in range(i + 1, 6) if BANKS[i][0] == BANKS[j][0]]
diff = [(i, j) for i in range(6) for j in range(i + 1, 6) if BANKS[i][0] != BANKS[j][0]]
mean = lambda C, prs: float(np.mean([C[i, j] for i, j in prs]))
print('cosine similarity between two "bank" vectors')
print(f'  input embeddings   same sense {mean(C0, same):.3f}   different sense {mean(C0, diff):.3f}')
print(f'  after 6 layers     same sense {mean(C6, same):.3f}   different sense {mean(C6, diff):.3f}')

# %% [markdown]
# ## What attention costs
#
# Every token attends to every token, so the work grows with the square of the length.

# %%
encd = enc.to(DEV)
cost = {}
for n in (32, 64, 128, 256, 512):
    xb = {'input_ids': torch.randint(1000, 5000, (16, n), device=DEV), 'attention_mask': torch.ones(16, n, dtype=torch.long, device=DEV)}
    with torch.no_grad():
        for _ in range(5): encd(**xb)                  # warm up
        if DEV == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): encd(**xb)
        if DEV == 'cuda': torch.cuda.synchronize()
    cost[n] = (time.time() - t0) / 20 * 1000
    print(f'{n:4d} tokens  {cost[n]:7.1f} ms for a batch of 16   attention cells per head {n * n:,}')
enc = enc.to('cpu')

# %% [build]
SENTS = ['The bank raised its base rate after inflation rose.',
         'Shares in the lender fell after it cut its dividend.',
         'The borrower shall repay the loan on the final repayment date.']
attn = []
for s in SENTS:
    e = tok(s, return_tensors='pt')
    with torch.no_grad():
        A = torch.stack(enc(**e, output_attentions=True).attentions)[:, 0]
    attn.append({'tokens': tok.convert_ids_to_tokens(e['input_ids'][0]), 'a': np.round(A.numpy(), 2).tolist()})
RES['attn'] = {'pieces': pieces, 'vocab': tok.vocab_size, 'sent': SENT, 'tokens': T_, 'layer': LYR, 'head': HEAD,
               'i_its': i_its, 'i_bank': i_bank, 'scores': scores[i_its].tolist(), 'weights': weights[i_its].tolist(),
               'banks': BANKS, 'C0': C0.tolist(), 'C6': C6.tolist(),
               'cos': {'in_same': mean(C0, same), 'in_diff': mean(C0, diff), 'out_same': mean(C6, same), 'out_diff': mean(C6, diff)},
               'cost': cost, 'dev': DEV, 'widget': attn, 'n_params': sum(p.numel() for p in enc.parameters())}

# %% [markdown]
# # 2. Financial sentiment
#
# 11,931 finance tweets and headlines labelled bearish, bullish or neutral (the `zeroshot/twitter-financial-news-sentiment`
# dataset, MIT licence). The published validation file is the test set here. Two thirds of the rows are neutral, so
# accuracy alone flatters a model that says "neutral" every time; macro-F1 averages the three classes equally.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

HF = 'https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/'
for f in ('sent_train.csv', 'sent_valid.csv'):
    if not os.path.exists(DATA + f):
        urllib.request.urlretrieve(HF + f, DATA + f)
clean = lambda s: re.sub(r'https?://\S+', '', s).strip()
tr = pd.read_csv(DATA + 'sent_train.csv'); te = pd.read_csv(DATA + 'sent_valid.csv')
tr['text'] = tr.text.map(clean); te['text'] = te.text.map(clean)
LAB = ['bearish', 'bullish', 'neutral']
ytr, yte = tr.label.values, te.label.values
print(len(tr), 'train,', len(te), 'test')
print('share of each class in test:', {LAB[k]: round(float((yte == k).mean()), 3) for k in range(3)})
tr.groupby('label').head(3).sort_values('label')

# %%
SENTI = {}
def report(name, pred):
    SENTI[name] = {'acc': float(accuracy_score(yte, pred)), 'f1': float(f1_score(yte, pred, average='macro')),
                   'f1c': f1_score(yte, pred, average=None).tolist(), 'cm': confusion_matrix(yte, pred).tolist()}
    print(f"{name:34s} accuracy {SENTI[name]['acc']:.3f}   macro-F1 {SENTI[name]['f1']:.3f}")
PRED = {}

PRED['always neutral'] = np.full(len(te), 2)
report('always neutral', PRED['always neutral'])

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr, Xte = tfidf.fit_transform(tr.text), tfidf.transform(te.text)
lr = LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
PRED['TF-IDF + logistic regression'] = lr.predict(Xte)
report('TF-IDF + logistic regression', PRED['TF-IDF + logistic regression'])

# %% [markdown]
# ## Two models used as they come
#
# A sentiment model trained on general tweets, and FinBERT, trained on financial news sentences.
# Neither has seen a row of this training set. Their "negative" is read as bearish and "positive" as bullish.

# %%
def run_pipe(model_name, texts, bs=64):
    p = pipeline('text-classification', model=model_name, device=0 if DEV == 'cuda' else -1, truncation=True, max_length=128)
    out = [r['label'].lower() for r in p(list(texts), batch_size=bs)]
    del p
    return out

MAP = {'negative': 0, 'positive': 1, 'neutral': 2}
for name, mid in (('general tweet model, as is', 'cardiffnlp/twitter-roberta-base-sentiment-latest'), ('FinBERT, as is', 'ProsusAI/finbert')):
    t0 = time.time()
    PRED[name] = np.array([MAP[l] for l in run_pipe(mid, te.text)])
    report(name, PRED[name])

# %% [markdown]
# ## Frozen embeddings and a linear model
#
# A sentence-embedding model turns each text into 384 numbers once. A logistic regression is then trained on those
# numbers. No transformer weights change, so this trains in seconds.

# %%
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer('BAAI/bge-small-en-v1.5', device=DEV)
t0 = time.time()
Etr = emb.encode(tr.text.tolist(), batch_size=128, normalize_embeddings=True)
Ete = emb.encode(te.text.tolist(), batch_size=128, normalize_embeddings=True)
t_embed = time.time() - t0
lr_e = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(Etr, ytr)
PRED['embeddings + logistic regression'] = lr_e.predict(Ete)
report('embeddings + logistic regression', PRED['embeddings + logistic regression'])
print(f'embedding {len(tr) + len(te):,} texts took {t_embed:.0f} s on {DEV}')

# %% [markdown]
# ## Fine-tuning
#
# All 66 million DistilBERT weights are updated on the labelled rows for three passes.

# %%
def batches(n, bs, shuffle, seed=0):
    idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]

def finetune(texts, labels, test_texts, base='distilbert-base-uncased', epochs=3, lr=3e-5, bs=32, seed=0, keep=False):
    torch.manual_seed(seed)
    tk = AutoTokenizer.from_pretrained(base)
    m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(texts) / bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * steps), max(0.0, (steps - s) / (0.9 * steps))))
    t0 = time.time()
    m.train()
    for ep in range(epochs):
        for b in batches(len(texts), bs, True, seed + ep):
            x = tk([texts[i] for i in b], padding=True, truncation=True, max_length=64, return_tensors='pt').to(DEV)
            loss = m(**x, labels=torch.tensor(labels[b]).to(DEV)).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
    secs = time.time() - t0
    m.eval(); out = []
    with torch.no_grad():
        for b in batches(len(test_texts), 128, False):
            x = tk([test_texts[i] for i in b], padding=True, truncation=True, max_length=64, return_tensors='pt').to(DEV)
            out.append(m(**x).logits.softmax(-1).cpu().numpy())
    P = np.concatenate(out)
    if keep:
        return P, secs, m, tk
    del m
    return P, secs

TR, TE = tr.text.tolist(), te.text.tolist()
P_ft, t_ft = finetune(TR, ytr, TE)
PRED['DistilBERT, fine-tuned'] = P_ft.argmax(1)
report('DistilBERT, fine-tuned', PRED['DistilBERT, fine-tuned'])
print(f'fine-tuning took {t_ft:.0f} s on {DEV}')

# %% [markdown]
# ## How many labels?
#
# The same three methods trained on fewer labelled rows. Each small size is repeated on three random samples.

# %%
SIZES = [100, 300, 1000, 3000, len(tr)]
CURVE = {'TF-IDF + logistic regression': {}, 'embeddings + logistic regression': {}, 'DistilBERT, fine-tuned': {}}
for n in SIZES:
    f = {k: [] for k in CURVE}
    for seed in range(1 if n == len(tr) else 3):
        idx = np.arange(len(tr)) if n == len(tr) else train_test_split(np.arange(len(tr)), train_size=n, stratify=ytr, random_state=seed)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1 if n < 1000 else 2, sublinear_tf=True)
        p1 = LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(v.fit_transform(tr.text[idx]), ytr[idx]).predict(v.transform(te.text))
        p2 = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(Etr[idx], ytr[idx]).predict(Ete)
        if n == len(tr):
            p3 = PRED['DistilBERT, fine-tuned']
        else:
            p3 = finetune([TR[i] for i in idx], ytr[idx], TE, epochs=max(3, min(20, 3000 // n)), seed=seed)[0].argmax(1)
        for k, p in zip(CURVE, (p1, p2, p3)):
            f[k].append(float(f1_score(yte, p, average='macro')))
    for k in CURVE:
        CURVE[k][n] = float(np.mean(f[k]))
    print(f'{n:5d} labels   ' + '   '.join(f'{k.split(",")[0].split(" +")[0]:11s} {CURVE[k][n]:.3f}' for k in CURVE))

# %% [markdown]
# ## Rows the general model gets wrong and the fine-tuned model gets right

# %%
g, f_ = PRED['general tweet model, as is'], PRED['DistilBERT, fine-tuned']
fix = te[(g != yte) & (f_ == yte)].assign(general=[LAB[k] for k in g[(g != yte) & (f_ == yte)]])
fix['truth'] = fix.label.map(lambda k: LAB[k])
pd.set_option('display.max_colwidth', 110)
fix[fix.text.str.len() < 90][['text', 'truth', 'general']].head(12)

# %% [build]
rng = np.random.default_rng(1)
both_wrong = te[(f_ != yte)]
ex_fix = fix[fix.text.str.len() < 95][['text', 'truth', 'general']].head(40).values.tolist()
ex_wrong = [[r.text, LAB[r.label], LAB[f_[i]]] for i, r in both_wrong.iterrows() if len(r.text) < 95][:40]
samp = rng.choice(len(te), 400, replace=False)
RES['senti'] = {'n_train': len(tr), 'n_test': len(te), 'share': [float((yte == k).mean()) for k in range(3)], 'lab': LAB,
                'm': SENTI, 'curve': CURVE, 'sizes': SIZES, 't_ft': t_ft, 't_embed': t_embed, 'dev': DEV,
                'head': tr.groupby('label').head(4).sort_values('label')[['text', 'label']].values.tolist(),
                'ex_fix': ex_fix, 'ex_wrong': ex_wrong,
                'browse': [{'t': te.text[i][:120], 'y': int(yte[i]), 'p': {k: int(PRED[k][i]) for k in PRED if k != 'always neutral'}} for i in samp]}

# %% [markdown]
# # 3. Named entity recognition
#
# 3,200 simulated sentences of four kinds: payment narratives in capitals, case notes, news lines and complaint text.
# Six entity types are tagged. The test sentences use sentence patterns, people, companies and places that do not
# appear in the training sentences.

# %%
load = lambda f: [json.loads(l) for l in open(DATA + f, encoding='utf-8')]
ner_tr, ner_te = load('ner_train.jsonl'), load('ner_test.jsonl')
TYPES = ['PER', 'ORG', 'LOC', 'MONEY', 'DATE', 'ACCT']
for r in ner_te[:4]:
    print(f"[{r['genre']}] " + ' '.join(f'{t}/{g}' if g != 'O' else t for t, g in zip(r['tokens'], r['tags'])))

# %% [markdown]
# ## Scoring
#
# An entity counts as found only if the predicted span has the same first token, the same last token and the same type.
# Half of a company name earns nothing.

# %%
def spans(tags):
    """BIO tags -> set of (type, first token, last token)"""
    out, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ['O']):
        if start is not None and not (t.startswith('I-') and t[2:] == typ):
            out.add((typ, start, i - 1)); start = None
        if t.startswith('B-') or (t.startswith('I-') and start is None):
            start, typ = i, t[2:]
    return out

def prf(gold, pred, types=TYPES, rows=None):
    res = {}
    rows = range(len(gold)) if rows is None else rows
    for ty in list(types) + ['all']:
        keep = (lambda s: s[0] in types) if ty == 'all' else (lambda s: s[0] == ty)
        tp = fp = fn = 0
        for i in rows:
            g, p = {s for s in gold[i] if keep(s)}, {s for s in pred[i] if keep(s)}
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res[ty] = {'p': pr, 'r': rc, 'f1': 2 * pr * rc / max(pr + rc, 1e-9), 'n': tp + fn}
    return res

GOLD = [spans(r['tags']) for r in ner_te]
GENRES = ['narrative', 'note', 'news', 'complaint']
by_genre = lambda g: [i for i, r in enumerate(ner_te) if r['genre'] == g]
print('test entities:', {t: sum(s[0] == t for g in GOLD for s in g) for t in TYPES})

# %% [markdown]
# ## Rules
#
# Amounts, dates and account numbers have a shape, so a regular expression can describe them.

# %%
MON = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
RULES = {
 'MONEY': r'(?:£|(?:GBP|EUR)\s)\d[\d,]*(?:\.\d+)?(?:\s?(?:m|million))?\b|\b\d[\d,]*\.\d\d\sGBP\b|\b\d+(?:\.\d+)?\smillion\spounds\b',
 'DATE': rf'\b\d{{1,2}}\s{MON}\s\d{{4}}\b|\b\d{{2}}/\d{{2}}/\d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b{MON}\s\d{{1,2}},\s\d{{4}}\b',
 'ACCT': r'\bGB\d{2}\s[A-Z]{4}(?:\s\d{4}){3}\s\d{2}\b|\b\d{2}-\d{2}-\d{2}\s\d{8}\b',
}

def char_to_tokens(tokens):
    text, offs, pos = ' '.join(tokens), [], 0
    for t in tokens:
        offs.append((pos, pos + len(t))); pos += len(t) + 1
    return text, offs

def to_span(offs, a, b, typ):
    idx = [i for i, (s, e) in enumerate(offs) if s < b and e > a]
    return (typ, idx[0], idx[-1]) if idx else None

def rule_ner(tokens):
    text, offs = char_to_tokens(tokens)
    out = set()
    for typ, pat in RULES.items():
        for m in re.finditer(pat, text, flags=re.I):
            out.add(to_span(offs, m.start(), m.end(), typ))
    return out - {None}

P_rule = [rule_ner(r['tokens']) for r in ner_te]
NER = {'rules': prf(GOLD, P_rule, ['MONEY', 'DATE', 'ACCT'])}
pd.DataFrame(NER['rules']).T.round(3)

# %% [markdown]
# ## A general NER model, as is
#
# `dslim/bert-base-NER` was trained on news text to find people, organisations and places.

# %%
gen = pipeline('token-classification', model='dslim/bert-base-NER', aggregation_strategy='simple', device=0 if DEV == 'cuda' else -1)
def general_ner(tokens):
    text, offs = char_to_tokens(tokens)
    return {to_span(offs, e['start'], e['end'], e['entity_group']) for e in gen(text) if e['entity_group'] in ('PER', 'ORG', 'LOC')} - {None}
P_gen = [general_ner(r['tokens']) for r in ner_te]
NER['general'] = prf(GOLD, P_gen, ['PER', 'ORG', 'LOC'])
NER['general_genre'] = {g: prf(GOLD, P_gen, ['PER', 'ORG', 'LOC'], by_genre(g))['all'] for g in GENRES}
del gen
print(pd.DataFrame(NER['general']).T.round(3))
print('\nF1 on people, organisations and places, by kind of text')
for g in GENRES:
    print(f"  {g:10s} {NER['general_genre'][g]['f1']:.3f}")

# %% [markdown]
# ## Fine-tuning a tagger
#
# The model sees word pieces, the labels are per word. Each word's label goes on its first piece and the other
# pieces are ignored in the loss.

# %%
TAGS = ['O'] + [f'{b}-{t}' for t in TYPES for b in 'BI']
T2I = {t: i for i, t in enumerate(TAGS)}
ntok = AutoTokenizer.from_pretrained('distilbert-base-cased')

def encode(rows):
    e = ntok([r['tokens'] for r in rows], is_split_into_words=True, padding=True, truncation=True, max_length=96, return_tensors='pt')
    lab = torch.full(e['input_ids'].shape, -100)
    for i, r in enumerate(rows):
        prev = None
        for j, w in enumerate(e.word_ids(i)):
            if w is not None and w != prev:
                lab[i, j] = T2I[r['tags'][w]]
            prev = w
    return e, lab

e1, l1 = encode(ner_te[1:2])
align = [(p, TAGS[l] if l >= 0 else 'ignored') for p, l in zip(ntok.convert_ids_to_tokens(e1['input_ids'][0]), l1[0].tolist())][1:-1]
print(ner_te[1]['tokens'])
for p, l in align:
    print(f'  {p:12s} {l}')

# %%
def train_tagger(rows, epochs=4, lr=5e-5, bs=32, seed=0):
    torch.manual_seed(seed)
    m = AutoModelForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(TAGS)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time(); m.train()
    for ep in range(epochs):
        for b in batches(len(rows), bs, True, seed + ep):
            e, lab = encode([rows[i] for i in b])
            m(**e.to(DEV), labels=lab.to(DEV)).loss.backward(); opt.step(); opt.zero_grad()
    return m.eval(), time.time() - t0

def predict_tagger(m, rows):
    out = []
    with torch.no_grad():
        for b in batches(len(rows), 64, False):
            e, lab = encode([rows[i] for i in b])
            pr = m(**e.to(DEV)).logits.argmax(-1).cpu()
            for k, i in enumerate(b):
                out.append(spans([TAGS[p] for p, l in zip(pr[k].tolist(), lab[k].tolist()) if l != -100]))
    return out

NER['curve'] = {}
for n in (100, 300, 1000, len(ner_tr)):
    m, secs = train_tagger(ner_tr[:n], epochs=4 if n >= 1000 else 10)
    P_ft_ner = predict_tagger(m, ner_te)
    NER['curve'][n] = prf(GOLD, P_ft_ner)['all']['f1']
    print(f'{n:5d} training sentences   F1 {NER["curve"][n]:.3f}   ({secs:.0f} s)')
    if n < len(ner_tr): del m
NER['ft'] = prf(GOLD, P_ft_ner)
NER['ft_genre'] = {g: prf(GOLD, P_ft_ner, TYPES, by_genre(g))['all'] for g in GENRES}
NER['t_ft'] = secs
pd.DataFrame(NER['ft']).T.round(3)

# %% [markdown]
# ## What the fine-tuned tagger still misses

# %%
miss = []
for i, r in enumerate(ner_te):
    for (ty, a, b) in sorted(GOLD[i] - P_ft_ner[i]):
        got = [s for s in P_ft_ner[i] if s[1] <= b and s[2] >= a]
        miss.append({'genre': r['genre'], 'type': ty, 'entity': ' '.join(r['tokens'][a:b + 1]),
                     'predicted': '; '.join(f"{s[0]}: {' '.join(r['tokens'][s[1]:s[2] + 1])}" for s in got) or 'nothing'})
miss = pd.DataFrame(miss)
print(len(miss), 'missed entities of', sum(len(g) for g in GOLD))
print(miss.groupby(['genre', 'type']).size().sort_values(ascending=False).head(6))
miss.head(12)

# %% [build]
def show(i, pred):
    r = ner_te[i]
    return {'tokens': r['tokens'], 'genre': r['genre'], 'gold': sorted(GOLD[i]), 'pred': sorted(pred[i])}
pickg = {g: by_genre(g)[:6] for g in GENRES}
RES['ner'] = {'n_train': len(ner_tr), 'n_test': len(ner_te), 'types': TYPES, 'm': NER, 'align': align,
              'align_tokens': ner_te[1]['tokens'], 'align_tags': ner_te[1]['tags'],
              'n_ent': {t: sum(s[0] == t for g in GOLD for s in g) for t in TYPES},
              'genre_n': {g: len(by_genre(g)) for g in GENRES},
              'examples': {g: [{'gen': show(i, P_gen), 'ft': show(i, P_ft_ner), 'rule': show(i, P_rule)} for i in pickg[g]] for g in GENRES},
              'miss': miss.head(40).values.tolist(), 'n_miss': len(miss),
              'miss_top': [[a, b, int(c)] for (a, b), c in miss.groupby(['genre', 'type']).size().sort_values(ascending=False).head(6).items()]}
del m

# %% [markdown]
# # 4. Documents: classification
#
# 2,000 simulated pages of eight document types. A page holds one to three paragraphs of its own type and, for most
# types, one or two paragraphs that several types share (governing law, notices, statement footers).
# The models train on clean text. The test pages are then degraded the way a poor scan degrades text.

# %%
ptr, pte = pd.read_csv(DATA + 'pages_train.csv'), pd.read_csv(DATA + 'pages_test.csv')
DT = sorted(ptr.doc_type.unique())
print(len(ptr), 'train pages,', len(pte), 'test pages')
for t in ('loan_agreement', 'guarantee'):
    print(f'\n[{t}] ' + ptr[ptr.doc_type == t].text.iloc[0])

# %%
SWAP = {'l': '1', 'I': 'l', 'O': '0', 'o': 'c', 'e': 'c', 'S': '5', 'B': '8', 'm': 'rn', 'h': 'b', 'a': 'o', 'u': 'v', 'i': 'l', 't': 'f', 'n': 'r'}
def ocr_noise(text, p, seed=0):
    """each character is, with probability p, swapped for a look-alike, dropped, or has its space removed"""
    r = np.random.default_rng(seed)
    out = []
    for ch in text:
        if r.random() < p:
            ch = SWAP.get(ch, '' if ch == ' ' or r.random() < .5 else ch)
        out.append(ch)
    return ''.join(out)

NOISE = [0, 0.02, 0.05, 0.10, 0.20]
print(ocr_noise(pte.text[0][:150], 0.10))

# %%
vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, min_df=2)
mw = LogisticRegression(C=10, max_iter=2000).fit(vw.fit_transform(ptr.text), ptr.doc_type)
mc = LogisticRegression(C=10, max_iter=2000).fit(vc.fit_transform(ptr.text), ptr.doc_type)
Eptr = emb.encode(ptr.text.tolist(), batch_size=64, normalize_embeddings=True)
me = LogisticRegression(C=10, max_iter=3000).fit(Eptr, ptr.doc_type)

DOC = {'word TF-IDF': {}, 'character n-gram TF-IDF': {}, 'embeddings': {}}
PRED_DOC = {}
for p in NOISE:
    noisy = [ocr_noise(t, p, seed=i) for i, t in enumerate(pte.text)]
    preds = (mw.predict(vw.transform(noisy)), mc.predict(vc.transform(noisy)),
             me.predict(emb.encode(noisy, batch_size=64, normalize_embeddings=True)))
    for k, pr in zip(DOC, preds):
        DOC[k][p] = float((pr == pte.doc_type).mean())
        PRED_DOC[(k, p)] = pr
    print(f'{p:5.0%} of characters damaged   ' + '   '.join(f'{k} {DOC[k][p]:.3f}' for k in DOC))

# %%
cm_doc = pd.crosstab(pte.doc_type, PRED_DOC[('word TF-IDF', 0)], rownames=['truth'], colnames=['predicted']).reindex(index=DT, columns=DT, fill_value=0)
wrong = pte[PRED_DOC[('word TF-IDF', 0)] != pte.doc_type].assign(predicted=PRED_DOC[('word TF-IDF', 0)][PRED_DOC[('word TF-IDF', 0)] != pte.doc_type])
print(len(wrong), 'of', len(pte), 'clean test pages misclassified by word TF-IDF')
wrong[['doc_type', 'predicted', 'text']].head(5)

# %% [markdown]
# # 5. Documents: information extraction
#
# 400 simulated loan agreements. Six fields are wanted from each: borrower, lender, principal, interest rate,
# final repayment date and governing law. Each agreement also holds values that look right and are wrong:
# an arrangement fee, a default rate, a first repayment date, a guarantor.

# %%
loans = load('loans.jsonl')
FIELDS = ['borrower', 'lender', 'principal', 'rate', 'maturity', 'law']
print(loans[0]['text'], '\n')
print({f: loans[0][f] for f in FIELDS})

# %% [markdown]
# ## Rules written from the usual wording
#
# One pattern per field, written against the phrasing that most agreements use.

# %%
DATE_ = r'\d{1,2} [A-Z][a-z]+ \d{4}'
RX = {
 'borrower': r'made between (.+?) \(the "Borrower"\)',
 'lender': r'\(the "Borrower"\) and (.+?) \(the "Lender"\)',
 'principal': r'principal amount of (GBP [\d,]+)',
 'rate': r'accrues on the loan at (.+?) per annum',
 'maturity': rf'repaid in full on ({DATE_})',
 'law': r'governed by the law of (.+?)\.',
}
def rule_extract(text):
    out = {}
    for f, pat in RX.items():
        m = re.search(pat, text)
        out[f] = m.group(1) if m else None
    return out

norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'^(the|law of) ', '', (s or '').strip().strip('."\',;:()').lower())).replace(' per annum', '').replace(' a year', '')
ok = lambda pred, gold: norm(pred) == norm(gold)
EX_rule = [rule_extract(r['text']) for r in loans]
IE = {'rules': {f: float(np.mean([ok(p[f], r[f]) for p, r in zip(EX_rule, loans)])) for f in FIELDS}}
IE['rules_none'] = {f: float(np.mean([p[f] is None for p in EX_rule])) for f in FIELDS}
print('share of agreements with the field extracted correctly')
print({f: round(v, 3) for f, v in IE['rules'].items()})

# %% [markdown]
# ## A question-answering model
#
# `deepset/roberta-base-squad2` reads a passage and a question and returns the span of the passage that answers it,
# with a score. The span is chosen from the model's start and end probabilities. It was trained on Wikipedia questions and has seen no loan agreement.

# %%
QUESTIONS = {'borrower': 'Who is the borrower?', 'lender': 'Who is the lender?',
             'principal': 'What is the principal amount of the loan?', 'rate': 'What is the rate of interest on the loan?',
             'maturity': 'On what date must the loan be repaid in full?', 'law': 'Which law governs the agreement?'}
from transformers import AutoModelForQuestionAnswering
qtok = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
qam = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2').to(DEV).eval()

def answer(question, context, max_len=30):
    """the span of the context with the highest start x end probability, and that probability"""
    e = qtok(question, context, return_tensors='pt', truncation='only_second', max_length=512, return_offsets_mapping=True)
    off = e.pop('offset_mapping')[0]
    in_ctx = torch.tensor([s_ == 1 for s_ in e.sequence_ids(0)])
    with torch.no_grad():
        o = qam(**e.to(DEV))
    ps = o.start_logits[0].softmax(-1).cpu() * in_ctx; pe = o.end_logits[0].softmax(-1).cpu() * in_ctx
    sc = torch.triu(ps[:, None] * pe[None, :]) - torch.triu(ps[:, None] * pe[None, :], diagonal=max_len)   # start <= end < start + 30
    i, j = divmod(int(sc.argmax()), sc.shape[1])
    return context[off[i][0]:off[j][1]], float(sc[i, j])

t0 = time.time()
EX_qa = [{f: answer(QUESTIONS[f], r['text']) for f in FIELDS} for r in loans]
t_qa = time.time() - t0
del qam
IE['qa'] = {f: float(np.mean([ok(p[f][0], r[f]) for p, r in zip(EX_qa, loans)])) for f in FIELDS}
print({f: round(v, 3) for f, v in IE['qa'].items()})
print(f'{t_qa / len(loans) * 1000:.0f} ms an agreement on {DEV}')

# %% [markdown]
# ## Where each method fails
#
# The simulator records which of three phrasings each clause used. Phrasing 0 is the usual one.

# %%
VKEY = {'borrower': 'parties', 'lender': 'parties', 'principal': 'principal', 'rate': 'rate', 'maturity': 'maturity', 'law': 'law'}
byv = []
for f in FIELDS:
    for v in range(3):
        rows = [i for i, r in enumerate(loans) if r['variant'][VKEY[f]] == v]
        byv.append({'field': f, 'phrasing': v, 'agreements': len(rows),
                    'rules': np.mean([ok(EX_rule[i][f], loans[i][f]) for i in rows]),
                    'qa model': np.mean([ok(EX_qa[i][f][0], loans[i][f]) for i in rows])})
byv = pd.DataFrame(byv)
byv.round(3)

# %%
qa_wrong = [{'field': f, 'truth': r[f], 'qa answer': p[f][0], 'score': round(p[f][1], 2)}
            for p, r in zip(EX_qa, loans) for f in FIELDS if not ok(p[f][0], r[f])]
print(len(qa_wrong), 'wrong answers of', len(loans) * len(FIELDS))
pd.DataFrame(qa_wrong).groupby('field').head(3).sort_values('field')

# %% [markdown]
# ## Using the score
#
# The model's score lets the pipeline accept confident answers and send the rest to a person.

# %%
sc = np.array([p[f][1] for p in EX_qa for f in FIELDS]); good = np.array([ok(p[f][0], r[f]) for p, r in zip(EX_qa, loans) for f in FIELDS])
THR = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
route = [{'threshold': t, 'accepted': float((sc >= t).mean()), 'wrong among accepted': float(1 - good[sc >= t].mean())} for t in THR]
pd.DataFrame(route).round(3)

# %% [build]
RES['doc'] = {'n_train': len(ptr), 'n_test': len(pte), 'types': DT, 'acc': {k: {str(p): v for p, v in d.items()} for k, d in DOC.items()},
              'noise': NOISE, 'cm': cm_doc.values.tolist(), 'n_wrong': len(wrong),
              'wrong': wrong[['doc_type', 'predicted', 'text']].head(8).values.tolist(),
              'sample': {t: ptr[ptr.doc_type == t].text.iloc[0] for t in DT},
              'noisy': {str(p): ocr_noise(pte.text[0][:230], p) for p in NOISE}}
RES['ie'] = {'n': len(loans), 'fields': FIELDS, 'm': IE, 'byv': byv.values.tolist(), 'loan0': {k: loans[0][k] for k in ['text'] + FIELDS},
             'rx': RX, 'questions': QUESTIONS, 't_qa': t_qa, 'dev': DEV, 'qa_wrong': pd.DataFrame(qa_wrong).groupby('field').head(6).values.tolist(),
             'n_qa_wrong': len(qa_wrong), 'route': route,
             'sc': [[round(float(s), 3), int(g)] for s, g in zip(sc, good)],
             'words': int(np.median([len(r['text'].split()) for r in loans])),
             'vshare': {k: [float(np.mean([r['variant'][k] == v for r in loans])) for v in range(3)] for k in ['parties', 'principal', 'rate', 'maturity', 'law']}}
json.dump(RES, open('build/m_nlp.json', 'w'))
print('wrote build/m_nlp.json', os.path.getsize('build/m_nlp.json'), 'bytes')
