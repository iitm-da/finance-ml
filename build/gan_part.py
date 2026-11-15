# %% [markdown]
# # Generative models, applied
#
# 100,000 simulated card transactions over 180 days, 0.9% of them fraud.
# Part 1 trains a GAN on them and tests the synthetic table three ways: fidelity, utility, privacy.
# Part 2 builds the fraud detection pipeline on the same table.

# %%
import json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

DATA = 'data/'
tx = pd.read_csv(DATA + 'transactions.csv')
tx.head()

# %%
NUM = ['hour', 'amount', 'customer_age', 'tenure_months', 'distance_km', 'txns_last_24h']
CAT = ['category', 'channel']
CATS = {c: sorted(tx[c].unique()) for c in CAT}

train = tx[tx.day < 120].reset_index(drop=True)        # first 120 days
test = tx[tx.day >= 120].reset_index(drop=True)        # last 60 days
hold = train.sample(frac=0.2, random_state=0)          # real rows the GAN does not see
fit = train.drop(hold.index).reset_index(drop=True)    # rows the GAN is trained on
hold = hold.reset_index(drop=True)
print('GAN training rows', len(fit), '| held out', len(hold), '| test', len(test))
print('fraud rate  fit %.2f%%  test %.2f%%' % (fit.is_fraud.mean() * 100, test.is_fraud.mean() * 100))

# %% [markdown]
# ## Encoding
#
# Numeric columns go through a quantile transform to a normal shape. Category and channel are one-hot.
# The fraud flag is the condition: the generator is told which kind of row to write.

# %%
qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0).fit(fit[NUM])

def encode(df):
    parts = [qt.transform(df[NUM])]
    for c in CAT:
        parts.append(np.eye(len(CATS[c]))[pd.Categorical(df[c], categories=CATS[c]).codes])
    return np.hstack(parts).astype(np.float32)

def decode(a, cond):
    out = pd.DataFrame(qt.inverse_transform(a[:, :len(NUM)]), columns=NUM)
    k = len(NUM)
    for c in CAT:
        m = len(CATS[c])
        out[c] = np.array(CATS[c])[a[:, k:k + m].argmax(1)]
        k += m
    for c in ['hour', 'customer_age', 'tenure_months', 'txns_last_24h']:
        out[c] = out[c].round().astype(int)
    out['amount'] = out.amount.round(2)
    out['distance_km'] = out.distance_km.round(1)
    out['is_fraud'] = cond.astype(int)
    return out[NUM + CAT + ['is_fraud']]

Xfit = torch.tensor(encode(fit))
Cfit = torch.tensor(fit.is_fraud.values[:, None].astype(np.float32))
D_X, D_Z = Xfit.shape[1], 64
BLOCKS = [len(CATS[c]) for c in CAT]
print('encoded width', D_X)

# %% [markdown]
# ## The two networks

# %%
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_Z + 1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, D_X))

    def forward(self, z, c, tau=0.2):
        h = self.net(torch.cat([z, c], 1))
        out, k = [h[:, :len(NUM)]], len(NUM)
        for m in BLOCKS:                           # one soft one-hot per categorical column
            out.append(nn.functional.gumbel_softmax(h[:, k:k + m], tau=tau))
            k += m
        return torch.cat(out, 1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_X + 1, 256), nn.LeakyReLU(0.2), nn.Linear(256, 256),
                                 nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1))

def sample(G, n, fraud_rate=None, cond=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    if cond is None:
        cond = (torch.rand(n, 1, generator=g) < fraud_rate).float()
    z = torch.randn(n, D_Z, generator=g)
    G.eval()
    with torch.no_grad():
        a = G(z, cond).numpy()
    G.train()
    return decode(a, cond.numpy()[:, 0])

# %% [markdown]
# ## Training
#
# Wasserstein loss with a gradient penalty. The critic takes five steps for each generator step.
# Fraud rows are drawn into every batch at 20% so the generator sees enough of them.

# %%
torch.manual_seed(0); np.random.seed(0)
G, C = Generator(), Critic()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
optC = torch.optim.Adam(C.parameters(), lr=2e-4, betas=(0.5, 0.9))
BATCH, N_CRITIC, EPOCHS = 512, 5, 150
STEPS = len(fit) // BATCH
idx_f = np.where(fit.is_fraud.values == 1)[0]
idx_l = np.where(fit.is_fraud.values == 0)[0]
RATE = float(fit.is_fraud.mean())
SNAP_AT = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 110, 150]
snaps, wdist = [], []

def snapshot(ep):
    s = sample(G, 4000, fraud_rate=RATE, seed=1)
    ks = {c: float(ks_2samp(s[c], fit[c]).statistic) for c in NUM}
    snaps.append({'epoch': ep, 'ks': ks, 'amount': s.amount.values[:600].tolist(), 'hour': s.hour.values[:600].tolist()})
    print('epoch %3d  KS amount %.3f  hour %.3f  mean %.3f' % (ep, ks['amount'], ks['hour'], np.mean(list(ks.values()))))

t0 = time.time()
snapshot(0)
for ep in range(1, EPOCHS + 1):
    w = 0.0
    for step in range(STEPS):
        nf = BATCH // 5
        b = np.concatenate([np.random.choice(idx_f, nf), np.random.choice(idx_l, BATCH - nf)])
        x, c = Xfit[b], Cfit[b]
        fake = G(torch.randn(BATCH, D_Z), c).detach()
        eps = torch.rand(BATCH, 1)
        mix = (eps * x + (1 - eps) * fake).requires_grad_(True)
        grad = torch.autograd.grad(C(mix, c).sum(), mix, create_graph=True)[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        gap = C(x, c).mean() - C(fake, c).mean()          # the critic's estimate of the distance
        lossC = -gap + 10 * gp
        optC.zero_grad(); lossC.backward(); optC.step()
        w += gap.item()
        if step % N_CRITIC == 0:
            lossG = -C(G(torch.randn(BATCH, D_Z), c), c).mean()
            optG.zero_grad(); lossG.backward(); optG.step()
    wdist.append(w / STEPS)
    if ep in SNAP_AT:
        snapshot(ep)
GAN_TIME = time.time() - t0
print('trained in %.0f s' % GAN_TIME)

# %%
synth = sample(G, len(fit), fraud_rate=RATE, seed=2)
synth.head()

# %% [markdown]
# ## Test 1: fidelity
#
# Column by column: the KS statistic for numeric columns, total variation distance for categorical ones.
# The held-out real rows give the value two real samples produce.

# %%
def tvd(a, b):
    p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)
    i = p.index.union(q.index)
    return float((p.reindex(i, fill_value=0) - q.reindex(i, fill_value=0)).abs().sum() / 2)

fid = pd.DataFrame({
    'synthetic vs real': {**{c: ks_2samp(synth[c], fit[c]).statistic for c in NUM}, **{c: tvd(synth[c], fit[c]) for c in CAT}},
    'held-out real vs real': {**{c: ks_2samp(hold[c], fit[c]).statistic for c in NUM}, **{c: tvd(hold[c], fit[c]) for c in CAT}},
}).round(3)
fid

# %%
# pairwise structure: Spearman correlations of the numeric columns and the fraud flag
cols = NUM + ['is_fraud']
cr, cs = fit[cols].corr('spearman').values, synth[cols].corr('spearman').values
iu = np.triu_indices(len(cols), 1)
corr_diff = np.abs(cr - cs)[iu]
print('correlation gap: mean %.3f, largest %.3f' % (corr_diff.mean(), corr_diff.max()))

# %%
# rules that hold in every real row
def rules(df):
    return pd.Series({
        'online_retail not on the online channel': ((df.category == 'online_retail') & (df.channel != 'online')).mean(),
        'cash_withdrawal not at an ATM': ((df.category == 'cash_withdrawal') & (df.channel != 'atm')).mean(),
        'online with a distance above 0': ((df.channel == 'online') & (df.distance_km > 0)).mean(),
    })
rule_tab = (pd.DataFrame({'real': rules(fit), 'synthetic': rules(synth)}) * 100).round(2)
rule_tab

# %% [markdown]
# ## Test 2: utility
#
# Train a fraud classifier on the synthetic table, test it on the real last 60 days, and compare with the same
# classifier trained on the real rows.

# %%
def features(df):
    X = df[NUM].copy()
    for c in CAT:
        X[c] = pd.Categorical(df[c], categories=CATS[c]).codes
    return X

def fraud_model(df, seed=0, **kw):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=seed,
                                          categorical_features=[len(NUM), len(NUM) + 1], **kw).fit(features(df), df.is_fraud)

def evaluate(model):
    p = model.predict_proba(features(test))[:, 1]
    return roc_auc_score(test.is_fraud, p), average_precision_score(test.is_fraud, p)

trtr = evaluate(fraud_model(fit))
tstr = evaluate(fraud_model(synth))
print('fraud rate  real %.2f%%  synthetic %.2f%%' % (fit.is_fraud.mean() * 100, synth.is_fraud.mean() * 100))
print('train on real,      test on real: ROC-AUC %.3f  PR-AUC %.3f' % trtr)
print('train on synthetic, test on real: ROC-AUC %.3f  PR-AUC %.3f' % tstr)

# %% [markdown]
# ## Test 3: privacy
#
# Distance to the closest record. For each synthetic row, find the nearest training row.
# Held-out real rows give the distances a table with no copying produces.
# A copy of the training rows with a little noise added is the generator this test has to catch.

# %%
def dcr(a, ref):
    nn_ = NearestNeighbors(n_neighbors=1).fit(ref)
    return nn_.kneighbors(a)[0][:, 0]

E_fit, E_hold, E_syn = encode(fit), encode(hold), encode(synth.iloc[:len(hold)])
rng = np.random.default_rng(0)
pick = rng.choice(len(E_fit), len(hold), replace=False)
E_copy = E_fit[pick].copy()
E_copy[:, :len(NUM)] += rng.normal(0, 0.05, (len(E_copy), len(NUM)))

d_hold, d_syn, d_copy = dcr(E_hold, E_fit), dcr(E_syn, E_fit), dcr(E_copy, E_fit)
p5 = np.percentile(d_hold, 5)
for name, d in [('held-out real', d_hold), ('GAN synthetic', d_syn), ('noisy copy', d_copy)]:
    print('%-14s median distance %.3f | closer than the 5th percentile of held-out: %.1f%%'
          % (name, np.median(d), (d < p5).mean() * 100))

# %%
# which is nearer: a training row, or a real row the generator did not see?
E_fit_sub = E_fit[pick]                       # as many training rows as held-out rows
def nearer_train(a):
    return float((dcr(a, E_fit_sub) < dcr(a, E_hold)).mean())
share_syn, share_copy = nearer_train(E_syn), nearer_train(E_copy)
print('nearer to a training row than to a held-out row: GAN %.1f%%, noisy copy %.1f%% (50%% means no memorising)'
      % (share_syn * 100, share_copy * 100))

exact = len(synth.merge(fit[NUM + CAT].drop_duplicates(), on=NUM + CAT))
print('synthetic rows identical to a training row:', exact, 'of', len(synth))

# %%
# the rare class: 600 synthetic fraud rows against the training fraud rows
syn_f = sample(G, 600, cond=torch.ones(600, 1), seed=3)
Ef_fit, Ef_hold, Ef_syn = encode(fit[fit.is_fraud == 1]), encode(hold[hold.is_fraud == 1]), encode(syn_f)
df_hold, df_syn = dcr(Ef_hold, Ef_fit), dcr(Ef_syn, Ef_fit)
print('fraud rows only: median distance to the nearest training fraud row: held-out %.3f, synthetic %.3f'
      % (np.median(df_hold), np.median(df_syn)))
print('training fraud rows', len(Ef_fit), '| held-out fraud rows', len(Ef_hold))

# %% [markdown]
# # Fraud detection pipeline
#
# Train on the first 120 days, score the last 60.

# %%
n_test, n_fraud = len(test), int(test.is_fraud.sum())
print('test rows', n_test, '| fraud', n_fraud, '| accuracy of flagging nothing %.2f%%' % ((1 - n_fraud / n_test) * 100))

# %%
# without labels: isolation forest scores how easy a row is to isolate
iso = IsolationForest(n_estimators=300, random_state=0).fit(pd.get_dummies(features(train), columns=CAT))
s_iso = -iso.score_samples(pd.get_dummies(features(test), columns=CAT))
# with labels: gradient boosting
gb = fraud_model(train)
s_gb = gb.predict_proba(features(test))[:, 1]
for name, s in [('isolation forest', s_iso), ('gradient boosting', s_gb)]:
    print('%-18s ROC-AUC %.3f  PR-AUC %.3f' % (name, roc_auc_score(test.is_fraud, s), average_precision_score(test.is_fraud, s)))

# %%
# alerts per day -> what the analysts catch
def at_budget(score, per_day, days=60):
    k = per_day * days
    top = np.argsort(-score)[:k]
    tp = int(test.is_fraud.values[top].sum())
    return {'alerts': k, 'caught': tp, 'precision': tp / k, 'recall': tp / n_fraud,
            'gbp': float(test.amount.values[top][test.is_fraud.values[top] == 1].sum())}
for per_day in [5, 10, 20, 50]:
    a, b = at_budget(s_iso, per_day), at_budget(s_gb, per_day)
    print('%2d alerts a day: isolation forest catches %3d of %d (precision %.0f%%) | gradient boosting %3d (precision %.0f%%)'
          % (per_day, a['caught'], n_fraud, a['precision'] * 100, b['caught'], b['precision'] * 100))

# %% [markdown]
# ## Does synthetic fraud help the classifier?
#
# Add k GAN fraud rows to the training set and compare with repeating real fraud rows and with class weights.

# %%
def pr(df, seed=0, **kw):
    return evaluate(fraud_model(df, seed, **kw))[1]

aug = {}
real_f = train[train.is_fraud == 1]
for k in [0, 500, 2000, 5000]:
    g_, o_ = [], []
    for seed in range(5):
        if k == 0:
            g_.append(pr(train, seed)); o_.append(g_[-1]); continue
        sf = sample(G, k, cond=torch.ones(k, 1), seed=100 + seed)
        g_.append(pr(pd.concat([train, sf], ignore_index=True), seed))
        o_.append(pr(pd.concat([train, real_f.sample(k, replace=True, random_state=seed)], ignore_index=True), seed))
    aug[k] = (float(np.mean(g_)), float(np.std(g_)), float(np.mean(o_)), float(np.std(o_)))
    print('k=%4d  + GAN fraud rows: PR-AUC %.3f (sd %.3f) | + repeated real fraud rows: %.3f (sd %.3f)' % ((k,) + aug[k]))
cw = [pr(train, seed, class_weight='balanced') for seed in range(5)]
print('class_weight=balanced: PR-AUC %.3f' % np.mean(cw))

# %%
# drift check: population stability index of the model score, first 30 test days against the last 30
def psi(a, b, bins=10):
    cuts = np.quantile(a, np.linspace(0, 1, bins + 1)[1:-1])
    pa = np.bincount(np.searchsorted(cuts, a), minlength=bins) / len(a)
    pb = np.bincount(np.searchsorted(cuts, b), minlength=bins) / len(b)
    return float(((pa - pb) * np.log((pa + 1e-6) / (pb + 1e-6))).sum())
first, last = (test.day < 150).values, (test.day >= 150).values
psi_score = psi(s_gb[first], s_gb[last])
print('PSI of the score, days 120-149 against 150-179: %.4f (above 0.2 is the usual alarm)' % psi_score)

# %% [build]
def hist(v, edges):
    return (np.histogram(v, edges)[0] / len(v)).round(4).tolist()
amt_edges = np.logspace(np.log10(0.5), np.log10(3000), 41)
dcr_edges = np.linspace(0, 2.5, 51)
order_gb, order_iso = np.argsort(-s_gb), np.argsort(-s_iso)
yt, at = test.is_fraud.values, test.amount.values
def curve(order, k=3000):
    o = order[:k]
    return {'tp': np.cumsum(yt[o]).astype(int).tolist(), 'gbp': np.cumsum(at[o] * yt[o]).round(0).astype(int).tolist()}
rngs = np.random.default_rng(5)
ri = rngs.choice(len(fit), 600, replace=False)
M = {
    'n': {'tx': len(tx), 'train': len(train), 'fit': len(fit), 'hold': len(hold), 'test': len(test),
          'fraud_all': int(tx.is_fraud.sum()), 'fraud_fit': int(fit.is_fraud.sum()), 'fraud_hold': int(hold.is_fraud.sum()),
          'fraud_train': int(train.is_fraud.sum()), 'fraud_test': n_fraud},
    'rate': {'all': float(tx.is_fraud.mean()), 'fit': RATE, 'synth': float(synth.is_fraud.mean()), 'test': float(test.is_fraud.mean())},
    'rows_real': fit.head(8)[NUM + CAT + ['is_fraud']].to_dict('records'),
    'rows_synth': synth.head(8).to_dict('records'),
    'gan': {'time': round(GAN_TIME, 1), 'epochs': EPOCHS, 'steps': STEPS, 'batch': BATCH, 'd_x': D_X, 'd_z': D_Z,
            'params_g': sum(p.numel() for p in G.parameters()), 'params_c': sum(p.numel() for p in C.parameters()),
            'wdist': [round(v, 4) for v in wdist]},
    'snaps': snaps,
    'real_pts': {'amount': fit.amount.values[ri].tolist(), 'hour': fit.hour.values[ri].tolist()},
    'fid': {'synth': fid['synthetic vs real'].to_dict(), 'hold': fid['held-out real vs real'].to_dict()},
    'corr': {'mean': float(corr_diff.mean()), 'max': float(corr_diff.max()), 'cols': cols,
             'real': np.round(cr, 3).tolist(), 'synth': np.round(cs, 3).tolist()},
    'rules': {'real': rule_tab['real'].to_dict(), 'synth': rule_tab['synthetic'].to_dict()},
    'hist': {'edges_amount': amt_edges.round(3).tolist(),
             'amount_real': hist(fit.amount, amt_edges), 'amount_synth': hist(synth.amount, amt_edges),
             'hour_real': (fit.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)).round(4).tolist(),
             'hour_synth': (synth.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)).round(4).tolist(),
             'cat': CATS['category'],
             'cat_real': fit.category.value_counts(normalize=True).reindex(CATS['category']).round(4).tolist(),
             'cat_synth': synth.category.value_counts(normalize=True).reindex(CATS['category'], fill_value=0).round(4).tolist(),
             'vel_real': (fit.txns_last_24h.value_counts(normalize=True).reindex(range(12), fill_value=0)).round(4).tolist(),
             'vel_synth': (synth.txns_last_24h.value_counts(normalize=True).reindex(range(12), fill_value=0)).round(4).tolist()},
    'util': {'trtr': trtr, 'tstr': tstr},
    'priv': {'p5': float(p5), 'median': {'hold': float(np.median(d_hold)), 'synth': float(np.median(d_syn)), 'copy': float(np.median(d_copy))},
             'below_p5': {'hold': float((d_hold < p5).mean()), 'synth': float((d_syn < p5).mean()), 'copy': float((d_copy < p5).mean())},
             'share': {'synth': share_syn, 'copy': share_copy}, 'exact': exact, 'n_synth': len(synth),
             'edges': dcr_edges.round(3).tolist(), 'h_hold': hist(d_hold, dcr_edges), 'h_synth': hist(d_syn, dcr_edges),
             'h_copy': hist(d_copy, dcr_edges),
             'fraud': {'hold': float(np.median(df_hold)), 'synth': float(np.median(df_syn)), 'n_fit': len(Ef_fit), 'n_hold': len(Ef_hold)}},
    'fraud': {'auc': {'iso': [roc_auc_score(yt, s_iso), average_precision_score(yt, s_iso)],
                      'gb': [roc_auc_score(yt, s_gb), average_precision_score(yt, s_gb)]},
              'budget': {str(k): {'iso': at_budget(s_iso, k), 'gb': at_budget(s_gb, k)} for k in [5, 10, 20, 50]},
              'curve': {'iso': curve(order_iso), 'gb': curve(order_gb)},
              'gbp_total': float(at[yt == 1].sum()), 'aug': {str(k): v for k, v in aug.items()}, 'cw': float(np.mean(cw)),
              'psi': psi_score,
              'top': test.iloc[order_gb[:8]].assign(score=np.round(s_gb[order_gb[:8]], 3))[
                  ['amount', 'hour', 'category', 'channel', 'distance_km', 'txns_last_24h', 'score', 'is_fraud']].to_dict('records')},
}
json.dump(M, open('build/m_gan.json', 'w'))
synth.to_csv(DATA + 'synthetic_transactions.csv', index=False)
print('wrote build/m_gan.json')
