# %% [markdown]
# # Time series forecasting for finance
#
# Daily net outflow of a simulated bank, six segments, 2022 to 2025. Positive values are money leaving the bank.
# Every model forecasts the next 14 days. The last 182 days are the test period.

# %%
import os, sys, urllib.request

if 'google.colab' in sys.modules:            # Colab ships torch, statsmodels and scikit-learn
    get_ipython().system('pip -q install prophet pytorch-forecasting lightning')

if not os.path.exists('data/cashflow.csv'):  # not run from the repo folder: fetch the two input files
    os.makedirs('data', exist_ok=True)
    for f in ('cashflow.csv', 'transactions.csv'):
        urllib.request.urlretrieve('https://raw.githubusercontent.com/iitm-da/finance-ml/master/data/' + f, 'data/' + f)

# %%
import json, time, logging, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

DATA = 'data/'
cf = pd.read_csv(DATA + 'cashflow.csv', parse_dates=['date'])
cf.head()

# %%
SEGS = list(cf.segment.unique())
HEAD = 'retail_current'          # the segment drawn on the slides
H, ENC = 14, 56                  # forecast 14 days from the previous 56

wide = cf.pivot(index='date', columns='segment', values='net_outflow')[SEGS]
Y = wide.values                  # (1461 days, 6 segments)
cal = cf[cf.segment == HEAD].reset_index(drop=True)
n = len(cal)

TEST0 = n - 182                  # first test day
VAL0 = TEST0 - 91                # first validation day
ORIGINS = [TEST0 + 14 * k for k in range(13)]   # first day of each 14-day forecast
print(n, 'days | train to', cal.date[VAL0 - 1].date(), '| validation to', cal.date[TEST0 - 1].date(),
      '| test', cal.date[TEST0].date(), 'to', cal.date[n - 1].date())

# %% [markdown]
# ## Scoring
#
# MAE in GBP million, and MASE: MAE divided by the MAE of the seasonal naive forecast on the training days.
# MASE below 1 beats "same weekday last week".

# %%
SCALE = np.abs(Y[7:TEST0] - Y[:TEST0 - 7]).mean(axis=0)      # per segment

def score(F):
    """F: (13 origins, 14 days, 6 segments) forecasts -> MAE per segment, mean MASE"""
    A = np.stack([Y[o:o + H] for o in ORIGINS])
    mae = np.abs(F - A).mean(axis=(0, 1))
    return mae, float((mae / SCALE).mean())

ACT = np.stack([Y[o:o + H] for o in ORIGINS])
FC, TIME, RES = {}, {}, {}

# %% [markdown]
# ## Seasonal naive
#
# Each day is forecast with the last observed value for the same weekday.

# %%
def seasonal_naive(o):
    return np.stack([Y[o + h - 7 * (h // 7 + 1)] for h in range(H)])

FC['naive'] = np.stack([seasonal_naive(o) for o in ORIGINS])
RES['naive'] = score(FC['naive'])
print('seasonal naive  MASE %.3f' % RES['naive'][1])

# %% [markdown]
# ## ARIMA
#
# One seasonal ARIMA per segment, fitted once on the days before the test period.
# At each origin the fitted model is given the history up to that day and asked for 14 days.

# %%
t0 = time.time()
F = np.zeros((len(ORIGINS), H, len(SEGS)))
arima_fit = {}
for j, seg in enumerate(SEGS):
    fit = SARIMAX(Y[:TEST0, j], order=(2, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False, maxiter=200)
    arima_fit[seg] = fit
    for i, o in enumerate(ORIGINS):
        F[i, :, j] = fit.apply(Y[:o, j]).forecast(H)
FC['arima'] = F
TIME['arima'] = time.time() - t0
RES['arima'] = score(F)
print('ARIMA(2,0,1)(0,1,1)7  MASE %.3f  %.0f s' % (RES['arima'][1], TIME['arima']))
print(arima_fit[HEAD].summary().tables[1])

# %% [markdown]
# ## ARIMA with calendar regressors
#
# The same model with the calendar passed as regressors: bank holiday, payday, the three days after it,
# direct-debit day, the first working day after a holiday, and the December ramp.

# %%
cal['sp1'] = (cal.since_payday == 1).astype(int)
cal['sp2'] = (cal.since_payday == 2).astype(int)
cal['sp3'] = (cal.since_payday == 3).astype(int)
XCOLS = ['is_holiday', 'is_payday', 'sp1', 'sp2', 'sp3', 'is_dd_day', 'post_holiday', 'december']
X = cal[XCOLS].values.astype(float)

t0 = time.time()
F = np.zeros((len(ORIGINS), H, len(SEGS)))
for j, seg in enumerate(SEGS):
    fit = SARIMAX(Y[:TEST0, j], exog=X[:TEST0], order=(2, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False, maxiter=200)
    for i, o in enumerate(ORIGINS):
        F[i, :, j] = fit.apply(Y[:o, j], exog=X[:o]).forecast(H, exog=X[o:o + H])
FC['arimax'] = F
TIME['arimax'] = time.time() - t0
RES['arimax'] = score(F)
print('ARIMA + calendar  MASE %.3f  %.0f s' % (RES['arimax'][1], TIME['arimax']))

# %% [markdown]
# ## Prophet
#
# Trend, weekly and yearly seasonality, bank holidays, and the payday and direct-debit flags as regressors.
# Prophet fits a curve against the date. It does not read yesterday's value.

# %%
from prophet import Prophet
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('prophet').setLevel(logging.ERROR)

hol = pd.DataFrame({'holiday': 'bank_holiday', 'ds': cal.date[cal.is_holiday == 1]})
PREG = ['is_payday', 'sp1', 'sp2', 'sp3', 'is_dd_day', 'post_holiday']

t0 = time.time()
F = np.zeros((len(ORIGINS), H, len(SEGS)))
prophet_parts = None
for j, seg in enumerate(SEGS):
    df = pd.DataFrame({'ds': cal.date, 'y': Y[:, j]})
    df[PREG] = cal[PREG]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, holidays=hol)
    for r in PREG:
        m.add_regressor(r)
    m.fit(df.iloc[:TEST0])
    pred = m.predict(df.drop(columns='y'))
    for i, o in enumerate(ORIGINS):
        F[i, :, j] = pred.yhat.values[o:o + H]
    if seg == HEAD:
        prophet_parts = pred
FC['prophet'] = F
TIME['prophet'] = time.time() - t0
RES['prophet'] = score(F)
print('Prophet  MASE %.3f  %.0f s' % (RES['prophet'][1], TIME['prophet']))

# %% [markdown]
# ## LSTM
#
# One network for all six segments. The encoder reads 56 days of outflow and calendar.
# The decoder reads the calendar of the next 14 days and writes one number per day.

# %%
import torch
import torch.nn as nn

dow1h = np.eye(7)[cal.dow.values]
KNOWN = np.hstack([dow1h, cal[['is_holiday', 'is_payday', 'sp1', 'sp2', 'sp3', 'is_dd_day', 'post_holiday',
                               'yr_sin', 'yr_cos', 'december']].values]).astype(np.float32)
MU, SD = Y[:VAL0].mean(axis=0), Y[:VAL0].std(axis=0)
Z = ((Y - MU) / SD).astype(np.float32)           # each segment scaled by its own training mean and sd

def windows(starts):
    xe, xd, yy, sid = [], [], [], []
    for j in range(len(SEGS)):
        for s in starts:
            xe.append(np.hstack([Z[s - ENC:s, j:j + 1], KNOWN[s - ENC:s]]))
            xd.append(KNOWN[s:s + H])
            yy.append(Z[s:s + H, j])
            sid.append(j)
    return [torch.tensor(np.array(a)) for a in (xe, xd, yy)] + [torch.tensor(sid)]

TR = windows(range(ENC, VAL0 - H + 1))
VA = windows(range(VAL0, TEST0 - H + 1))
TE = windows(ORIGINS)
print('training windows', len(TR[0]), '| validation', len(VA[0]), '| test', len(TE[0]))

# %%
class Seq2Seq(nn.Module):
    def __init__(self, n_known=KNOWN.shape[1], hidden=64, n_seg=len(SEGS)):
        super().__init__()
        self.emb = nn.Embedding(n_seg, 4)
        self.enc = nn.LSTM(1 + n_known + 4, hidden, batch_first=True)
        self.dec = nn.LSTM(n_known + 4, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, xe, xd, sid):
        e = self.emb(sid)
        _, state = self.enc(torch.cat([xe, e[:, None].expand(-1, xe.shape[1], -1)], -1))
        d, _ = self.dec(torch.cat([xd, e[:, None].expand(-1, xd.shape[1], -1)], -1), state)
        return self.out(d).squeeze(-1)

def train_lstm(seed, max_epochs=60, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Seq2Seq()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.L1Loss()
    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(TR[0]))
        tot = 0.0
        for k in range(0, len(perm), 128):
            b = perm[k:k + 128]
            opt.zero_grad()
            loss = lossf(net(TR[0][b], TR[1][b], TR[3][b]), TR[2][b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            v = lossf(net(VA[0], VA[1], VA[3]), VA[2]).item()
        hist.append((tot / len(perm), v))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, hist

def lstm_forecast(net):
    net.eval()
    with torch.no_grad():
        p = net(TE[0], TE[1], TE[3]).numpy()            # (6 segments x 13 origins, 14)
    p = p.reshape(len(SEGS), len(ORIGINS), H).transpose(1, 2, 0)
    return p * SD + MU

t0 = time.time()
lstm_runs = []
for seed in range(5):
    net, hist = train_lstm(seed)
    f = lstm_forecast(net)
    lstm_runs.append((score(f)[1], f, hist))
    print('seed', seed, 'epochs', len(hist), 'MASE %.3f' % lstm_runs[-1][0])
TIME['lstm'] = (time.time() - t0) / 5
FC['lstm'] = lstm_runs[0][1]
RES['lstm'] = score(FC['lstm'])
lstm_mase = np.array([r[0] for r in lstm_runs])
lstm_params = sum(p.numel() for p in net.parameters())
print('LSTM  MASE %.3f (seed 0) | mean %.3f sd %.3f over 5 seeds | %d weights | %.0f s per run'
      % (RES['lstm'][1], lstm_mase.mean(), lstm_mase.std(), lstm_params, TIME['lstm']))

# %% [markdown]
# ## Temporal Fusion Transformer
#
# Same windows. The TFT is told which inputs are static (the segment), which are known for the future (the calendar)
# and which are only observed in the past (the outflow). It returns the 10th, 50th and 90th percentile for each day.

# %%
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
logging.getLogger('lightning').setLevel(logging.ERROR)

long = cf.copy()
long['time_idx'] = long.groupby('segment').cumcount()
for c in ['dow', 'since_payday']:
    long[c] = long[c].astype(str)
KNOWN_REALS = ['is_holiday', 'is_payday', 'is_dd_day', 'post_holiday', 'yr_sin', 'yr_cos', 'december']
long[KNOWN_REALS] = long[KNOWN_REALS].astype(float)

training = TimeSeriesDataSet(
    long[long.time_idx < VAL0],
    time_idx='time_idx', target='net_outflow', group_ids=['segment'],
    min_encoder_length=ENC, max_encoder_length=ENC,
    min_prediction_length=H, max_prediction_length=H,
    static_categoricals=['segment'],
    time_varying_known_categoricals=['dow', 'since_payday'],
    time_varying_known_reals=KNOWN_REALS,
    time_varying_unknown_reals=['net_outflow'],
    target_normalizer=GroupNormalizer(groups=['segment']),
)
validation = TimeSeriesDataSet.from_dataset(training, long[long.time_idx < TEST0],
                                            min_prediction_idx=VAL0, stop_randomization=True)
train_dl = training.to_dataloader(train=True, batch_size=128, num_workers=0)
val_dl = validation.to_dataloader(train=False, batch_size=512, num_workers=0)

# %%
pl.seed_everything(0, verbose=False)
QS = [0.1, 0.5, 0.9]
tft = TemporalFusionTransformer.from_dataset(
    training, learning_rate=0.005, hidden_size=32, attention_head_size=4, dropout=0.1,
    hidden_continuous_size=16, loss=QuantileLoss(QS), log_interval=0,
)
ckpt = ModelCheckpoint(monitor='val_loss', dirpath='tft_ckpt', filename='best', save_top_k=1)
trainer = pl.Trainer(max_epochs=60, accelerator='cpu', gradient_clip_val=0.1, logger=False,
                     enable_progress_bar=False, enable_model_summary=False,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=8), ckpt])
t0 = time.time()
trainer.fit(tft, train_dl, val_dl)
TIME['tft'] = time.time() - t0
tft = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
tft_params = sum(p.numel() for p in tft.parameters())
print('TFT  %d epochs | %d weights | %.0f s' % (trainer.current_epoch, tft_params, TIME['tft']))

# %%
def tft_at(o, mode='quantiles'):
    """forecast the 14 days starting at index o, for every segment"""
    sub = long[(long.time_idx >= o - ENC) & (long.time_idx < o + H)]
    ds = TimeSeriesDataSet.from_dataset(training, sub, predict=True, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    out = tft.predict(dl, mode=mode, return_index=True, return_x=(mode == 'raw'),
                      trainer_kwargs=dict(logger=False, enable_progress_bar=False, accelerator='cpu'))
    return out

Q = np.zeros((len(ORIGINS), H, len(SEGS), 3))
for i, o in enumerate(ORIGINS):
    out = tft_at(o)
    order = [list(out.index.segment).index(s) for s in SEGS]
    Q[i] = out.output.numpy()[order].transpose(1, 0, 2)
FC['tft'] = Q[..., 1]
RES['tft'] = score(FC['tft'])
cover = float(((ACT >= Q[..., 0]) & (ACT <= Q[..., 2])).mean())
above = float((ACT > Q[..., 2]).mean())
below = float((ACT < Q[..., 0]).mean())
print('TFT median  MASE %.3f | actual inside P10-P90 on %.1f%% of days, above P90 %.1f%%, below P10 %.1f%%'
      % (RES['tft'][1], cover * 100, above * 100, below * 100))

# %% [markdown]
# ## What the TFT used
#
# Variable selection weights and the attention over the 56 encoder days, summed over the 13 test origins.

# %%
imp = {'static': 0, 'encoder': 0, 'decoder': 0, 'attention': 0}
for o in ORIGINS:
    raw = tft_at(o, mode='raw')
    it = tft.interpret_output(raw.output, reduction='sum')
    imp['static'] = imp['static'] + it['static_variables'].numpy()
    imp['encoder'] = imp['encoder'] + it['encoder_variables'].numpy()
    imp['decoder'] = imp['decoder'] + it['decoder_variables'].numpy()
    imp['attention'] = imp['attention'] + it['attention'].numpy()
enc_imp = dict(zip(tft.encoder_variables, (imp['encoder'] / imp['encoder'].sum() * 100).round(1)))
dec_imp = dict(zip(tft.decoder_variables, (imp['decoder'] / imp['decoder'].sum() * 100).round(1)))
attn = imp['attention'][:ENC] / imp['attention'][:ENC].sum()
print('encoder', enc_imp)
print('decoder', dec_imp)
print('attention, last 15 encoder days', attn[-15:].round(3))

# %% [markdown]
# ## All models

# %%
NAMES = {'naive': 'Seasonal naive', 'arima': 'ARIMA', 'prophet': 'Prophet', 'arimax': 'ARIMA + calendar',
         'lstm': 'LSTM', 'tft': 'TFT (median)'}
tab = pd.DataFrame({NAMES[k]: list(RES[k][0].round(2)) + [round(RES[k][1], 3)] for k in NAMES},
                   index=SEGS + ['mean MASE']).T
tab

# %% [markdown]
# ## Applications: from a forecast to a decision
#
# A treasury desk holds cash against the outflow it expects. The four cells below use the forecasts already made,
# with no new model: a backtest of the P90, the margin each model needs, the whole bank, and the second week.

# %%
# 1. backtest of the P90: it should be exceeded on 10% of days
from scipy.stats import binomtest
j = SEGS.index(HEAD)
n_days = ACT[:, :, 0].size
rows = []
for i, s in enumerate(SEGS):
    up = int((ACT[:, :, i] > Q[:, :, i, 2]).sum()); dn = int((ACT[:, :, i] < Q[:, :, i, 0]).sum())
    rows.append([s, up, round(up / n_days * 100, 1), dn, round(dn / n_days * 100, 1),
                 round(float(binomtest(up, n_days, 0.10).pvalue), 4)])
bt = pd.DataFrame(rows, columns=['segment', 'days above P90', '% above', 'days below P10', '% below', 'p-value against 10%'])
APP = {'bt': rows, 'n_days': n_days}
bt

# %%
# 2. the margin each model needs: hold forecast + margin, and choose the margin that leaves 9 of 182 days short
rows = []
ERR = {}
for k in FC:
    e = (ACT[:, :, j] - FC[k][:, :, j]).ravel()              # positive: more went out than forecast
    c = float(np.quantile(e, 0.95, method='higher'))
    rows.append([NAMES[k], round(c, 1), int((e > c).sum()), round(float(np.clip(c - e, 0, None).mean()), 1)])
    ERR[k] = e.round(1).tolist()
mg = pd.DataFrame(rows, columns=['model', 'margin, GBP m', 'days short', 'idle cash per day, GBP m'])
APP['margin'] = rows; APP['err'] = ERR; APP['mean_out'] = round(float(ACT[:, :, j].mean()), 1)
mg

# %%
# 3. the whole bank: add the six segments
tot = ACT.sum(axis=2); p50 = Q[..., 1].sum(axis=2); p90 = Q[..., 2].sum(axis=2)
seg_mae = np.abs(FC['tft'] - ACT).mean(axis=(0, 1))
e_tot = (tot - p50).ravel()
need = float(np.quantile(e_tot, 0.90, method='higher'))
APP['bank'] = {'seg_mae': seg_mae.round(1).tolist(), 'sum_seg_mae': round(float(seg_mae.sum()), 1),
               'mae_total': round(float(np.abs(FC['tft'].sum(axis=2) - tot).mean()), 1),
               'seg_margin': (Q[..., 2] - Q[..., 1]).mean(axis=(0, 1)).round(1).tolist(),
               'sum_margin': round(float((p90 - p50).mean()), 1), 'above_sum_p90': int((tot > p90).sum()),
               'need_margin': round(need, 1), 'above_need': int((e_tot > need).sum()),
               'mean_total': round(float(tot.mean()), 1)}
print('TFT median: sum of the six segment MAEs %.1f | MAE of the bank total %.1f'
      % (APP['bank']['sum_seg_mae'], APP['bank']['mae_total']))
print('sum of six P90s: %.1f above the summed medians, exceeded on %d of %d days'
      % (APP['bank']['sum_margin'], APP['bank']['above_sum_p90'], n_days))
print('margin exceeded on 10%% of days at bank level: %.1f (%d days)' % (need, APP['bank']['above_need']))

# %%
# 4. the second week against the first: each holds every weekday once
rows = []
for k in FC:
    a = np.abs(FC[k] - ACT) / SCALE
    rows.append([NAMES[k], round(float(a[:, :7].mean()), 3), round(float(a[:, 7:].mean()), 3)])
APP['week'] = rows
pd.DataFrame(rows, columns=['model', 'MASE, days 1 to 7', 'MASE, days 8 to 14'])

# %% [markdown]
# ## Applications: ATM cash
#
# The same four steps on a second problem. `data/atm.csv` holds the cash withdrawn each day at 200 ATMs over two
# years, in GBP thousand; 20 of the ATMs opened 90 days before the test period. Forecasts are issued once a week
# for the next 7 days, and the last 91 days are the test.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
atm = pd.read_csv('data/atm.csv', parse_dates=['date']).sort_values(['atm_id', 'date']).reset_index(drop=True)
A_TEST = 91
d0 = atm.date.min(); atm['di'] = (atm.date - d0).dt.days
ti = int(atm.di.max()) - A_TEST + 1                              # first test day
NEW = set(atm.groupby('atm_id').di.min().loc[lambda s: s > 0].index)
atm['kind_c'] = atm.kind.astype('category').cat.codes
atm['atm_c'] = atm.atm_id.astype('category').cat.codes
atm['origin'] = atm.di - ((atm.di - ti) % 7)                     # the weekly forecast this day belongs to
atm['h'] = atm.di - atm.origin + 1                               # 1 to 7 days ahead
g = atm.groupby('atm_id').withdrawn
atm['naive'] = g.shift(7)                                        # seasonal naive: the same weekday last week
atm['r28'] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=14).mean())
lvl = atm.loc[atm.h == 1, ['atm_id', 'origin', 'r28']].rename(columns={'r28': 'level'})
atm = atm.merge(lvl, on=['atm_id', 'origin'], how='left').dropna(subset=['level', 'naive'])
a_tr, a_te = atm[atm.di < ti], atm[atm.di >= ti].copy()
print('%d ATMs, %d of them new | training rows %d, test rows %d | training days for a new ATM: %d'
      % (atm.atm_id.nunique(), len(NEW), len(a_tr), len(a_te), a_tr[a_tr.atm_id.isin(NEW)].groupby('atm_id').size().max()))

# %%
# one shared model for all 200 ATMs, one fit per quantile; and one median model per ATM on its own history
FEAT = ['dow', 'is_holiday', 'is_payday', 'since_payday', 'december', 'h', 'level', 'kind_c', 'atm_c']
LOCAL = FEAT[:7]
QS = [50, 80, 90, 95, 99]
t0 = time.time()
for q in QS:
    mdl = HGB(loss='quantile', quantile=q / 100, max_iter=400, learning_rate=.06, random_state=0,
              categorical_features=[FEAT.index(c) for c in ('dow', 'kind_c', 'atm_c')]).fit(a_tr[FEAT], a_tr.withdrawn)
    a_te[f'q{q}'] = mdl.predict(a_te[FEAT])
t_shared = time.time() - t0
t0 = time.time()
a_te['local'] = np.nan
for a, part in a_tr.groupby('atm_id'):
    mm = HGB(loss='quantile', quantile=.5, max_iter=80, learning_rate=.1, random_state=0,
             categorical_features=[0]).fit(part[LOCAL], part.withdrawn)
    a_te.loc[a_te.atm_id == a, 'local'] = mm.predict(a_te.loc[a_te.atm_id == a, LOCAL])
t_local = time.time() - t0
a_scale = (a_tr.withdrawn - a_tr.naive).abs().groupby(a_tr.atm_id).mean()
a_te['sc'] = a_te.atm_id.map(a_scale)
is_new = a_te.atm_id.isin(NEW)
def a_mase(col, mask):
    d = a_te[mask]
    return float(((d.withdrawn - d[col]).abs() / d.sc).groupby(d.atm_id).mean().mean())
rows = [[nm, round(a_mase(c, ~is_new), 3), round(a_mase(c, is_new), 3), round(a_mase(c, is_new | ~is_new), 3)]
        for nm, c in [('Seasonal naive', 'naive'), ('One model per ATM', 'local'), ('One shared model', 'q50')]]
print('shared model: %.0f s for %d quantiles | 200 models, one per ATM: %.0f s' % (t_shared, len(QS), t_local))
pd.DataFrame(rows, columns=['MASE', '180 established ATMs', '20 new ATMs', 'all 200'])

# %%
# backtest of the P95, ATM by ATM: it should be exceeded on 5% of days
above95 = (a_te.withdrawn > a_te.q95)
per_atm = (above95.groupby(a_te.atm_id).sum()).astype(int)
print('all ATM-days: above the P95 on %.2f%% | by quantile: %s'
      % (above95.mean() * 100, ', '.join('P%d %.1f%%' % (q, (a_te.withdrawn > a_te[f"q{q}"]).mean() * 100) for q in QS)))
print('ATMs by number of days above the P95, out of 91:')
print(per_atm.value_counts().sort_index().to_string())
print('ATMs with 10 or more days above (binomial p < 0.02 for a true 5%%): %d' % int((per_atm >= 10).sum()))

# %%
# replenishment: every R days an ATM is topped up to the sum of the next R daily forecasts at one quantile
Dm = a_te.pivot(index='di', columns='atm_id', values='withdrawn').values
Fq = {q: a_te.pivot(index='di', columns='atm_id', values=f'q{q}').values for q in QS}
def replenish(q, R, keep=None):
    cash = np.zeros(Dm.shape[1]); out_days = unmet = idle = loaded = 0.0; tr_ = []
    for d in range(A_TEST):
        if d % R == 0:
            target = Fq[q][d:d + R].sum(axis=0)
            loaded += np.clip(target - cash, 0, None).sum(); cash = np.maximum(cash, target)
        start = cash.copy()
        short = np.clip(Dm[d] - cash, 0, None)
        out_days += (short > 0).sum(); unmet += short.sum()
        cash = np.clip(cash - Dm[d], 0, None); idle += cash.sum()
        if keep is not None:
            tr_.append([round(float(start[keep]), 1), round(float(cash[keep]), 1), int(short[keep] > 0)])
    n_ = Dm.size
    return {'out': round(out_days / n_ * 100, 2), 'unmet': round(unmet / Dm.sum() * 100, 2),
            'idle': round(idle / n_, 1), 'visits': round(7 / R, 2), 'trace': tr_}
ids = sorted(a_te.atm_id.unique())
KEEP = next(i for i, a in enumerate(ids) if a not in NEW and a_te.loc[a_te.atm_id == a, 'kind'].iloc[0] == 'residential')
GRID = {f'{q}_{R}': replenish(q, R, KEEP) for q in QS for R in (1, 2, 3, 7)}
pol = pd.DataFrame([[f'P{k.split("_")[0]}', int(k.split('_')[1]), v['out'], v['unmet'], v['idle'], v['visits']] for k, v in GRID.items()],
                   columns=['quantile', 'days between visits', '% ATM-days empty', '% of demand not served',
                            'cash left at day end, GBP k per ATM', 'visits per ATM per week'])
print('mean withdrawn per ATM per day in the test: %.1f' % Dm.mean())
pol

# %%
# kept for the slides
ex = {}
for kd in ['high_street', 'residential', 'transport']:
    a = next(a for a in ids if a not in NEW and atm.loc[atm.atm_id == a, 'kind'].iloc[0] == kd)
    p = atm[(atm.atm_id == a) & (atm.di >= ti - 42) & (atm.di < ti)]
    ex[kd] = {'id': a, 'y': p.withdrawn.round(1).tolist(), 'dow': p.dow.tolist(), 'pay': p.is_payday.tolist(),
              'hol': p.is_holiday.tolist(), 'dates': [str(x.date()) for x in p.date]}
APP['atm'] = {'n': len(ids), 'n_new': len(NEW), 'days': int(atm.di.max()) + 1, 'test': A_TEST, 'ex': ex,
              'kinds': a_te.groupby('kind').atm_id.nunique().to_dict(), 'mase': rows,
              't_shared': round(t_shared), 't_local': round(t_local),
              'above': {str(q): round(float((a_te.withdrawn > a_te[f'q{q}']).mean() * 100), 2) for q in QS},
              'per_atm': per_atm.tolist(), 'new_train_days': int(a_tr[a_tr.atm_id.isin(NEW)].groupby('atm_id').size().max()),
              'grid': GRID, 'keep': ids[KEEP], 'mean_day': round(float(Dm.mean()), 1)}
print('kept', list(APP['atm']))

# %% [markdown]
# ## Applications: call volumes
#
# `data/calls.csv` holds the calls offered to one contact centre in each half hour from 08:00 to 20:00, for two
# years. The forecast feeds a staffing formula (Erlang C): how many agents each half hour needs so that 80% of calls
# are answered within 20 seconds, with a mean handling time of 5 minutes. Forecasts are issued once a week and the
# last 91 days are the test.

# %%
calls = pd.read_csv('data/calls.csv', parse_dates=['date']).sort_values(['date', 'interval']).reset_index(drop=True)
C_TEST = 91
calls['di'] = (calls.date - calls.date.min()).dt.days
ci = int(calls.di.max()) - C_TEST + 1
calls['origin'] = calls.di - ((calls.di - ci) % 7)
calls['h'] = calls.di - calls.origin + 1
calls['naive'] = calls.groupby('interval').calls.shift(7)          # the same half hour a week earlier
daily = calls.groupby('di').calls.sum()
lev = (daily.shift(1).rolling(28).mean() / 24).rename('level')     # mean calls per half hour, last 28 days
calls = calls.merge(lev, left_on='origin', right_index=True, how='left').dropna(subset=['level', 'naive'])
c_tr, c_te = calls[calls.di < ci], calls[calls.di >= ci].copy()
CF = ['interval', 'dow', 'is_holiday', 'post_holiday', 'is_dd_day', 'after_dd', 'december', 'h']
CQ = [50, 80, 90, 95]
# the target is calls divided by the recent level, so the trees never have to forecast a level they have not seen
for q in CQ:
    mdl = HGB(loss='quantile', quantile=q / 100, max_iter=400, learning_rate=.06, random_state=0,
              categorical_features=[0, 1]).fit(c_tr[CF], c_tr.calls / c_tr.level)
    c_te[f'q{q}'] = np.clip(mdl.predict(c_te[CF]), 0, None) * c_te.level
# the same model on raw calls with the level as an input, for comparison
raw = {q: HGB(loss='quantile', quantile=q / 100, max_iter=400, learning_rate=.06, random_state=0,
              categorical_features=[0, 1]).fit(c_tr[CF + ['level']], c_tr.calls).predict(c_te[CF + ['level']]) for q in (50, 90)}
c_raw = {'bias': round(float(c_te.calls.sum() / raw[50].sum()), 3), 'above90': round(float((c_te.calls > raw[90]).mean() * 100), 1),
         'lev_train': round(float(c_tr.level.max()), 1), 'lev_test': round(float(c_te.level.max()), 1)}
c_scale = float((c_tr.calls - c_tr.naive).abs().mean())
ACTc = c_te.pivot(index='di', columns='interval', values='calls').values.astype(float)        # 91 days x 24 half hours
FQc = {q: c_te.pivot(index='di', columns='interval', values=f'q{q}').values for q in CQ}
NVc = c_te.pivot(index='di', columns='interval', values='naive').values
c_mase = {'naive': round(float(np.abs(ACTc - NVc).mean() / c_scale), 3), 'model': round(float(np.abs(ACTc - FQc[50]).mean() / c_scale), 3)}
print('%d half hours in training, %d in the test | mean calls per half hour in the test %.1f' % (len(c_tr), ACTc.size, ACTc.mean()))
print('MASE per half hour: seasonal naive %.3f, model %.3f' % (c_mase['naive'], c_mase['model']))
print('raw-calls model: actual / forecast %.3f, above its P90 on %.1f%% | highest level in training %.1f, in the test %.1f'
      % (c_raw['bias'], c_raw['above90'], c_raw['lev_train'], c_raw['lev_test']))

# %%
# backtest of the P90, and how the misses are spread over days
up = ACTc > FQc[90]
per_day = up.sum(axis=1)
srt = np.sort(per_day)[::-1]
top_days = int(np.searchsorted(np.cumsum(srt), up.sum() / 2) + 1)          # fewest days that hold half the misses
print('half hours above the P90: %d of %d, %.1f%%' % (up.sum(), up.size, up.mean() * 100))
print('days with no half hour above: %d | days with 8 or more above: %d | half of all misses fall on %d of %d days'
      % ((per_day == 0).sum(), (per_day >= 8).sum(), top_days, C_TEST))
ratio = ACTc.sum(axis=1) / FQc[50].sum(axis=1)
print('actual day total / forecast day total: 5th percentile %.2f, 95th percentile %.2f'
      % (np.quantile(ratio, .05), np.quantile(ratio, .95)))

# %%
# intraday update: at 12:00, scale the afternoon forecast by what the morning did
AM = 8                                                              # 08:00 to 12:00 is the first 8 half hours
r_am = ACTc[:, :AM].sum(axis=1) / FQc[50][:, :AM].sum(axis=1)
upd = FQc[50][:, AM:] * r_am[:, None]
mae_before = float(np.abs(ACTc[:, AM:] - FQc[50][:, AM:]).mean()); mae_after = float(np.abs(ACTc[:, AM:] - upd).mean())
print('afternoon MAE per half hour: %.2f with the weekly forecast, %.2f after the 12:00 update (%.0f%% lower)'
      % (mae_before, mae_after, (1 - mae_after / mae_before) * 100))
print('days on which the update helped: %d of %d' % ((np.abs(ACTc[:, AM:] - upd).mean(axis=1) < np.abs(ACTc[:, AM:] - FQc[50][:, AM:]).mean(axis=1)).sum(), C_TEST))

# %%
# staffing: Erlang C turns calls per half hour into agents, for 80% answered within 20 seconds
from math import exp
AHT, TGT_S, TGT = 300, 20, 0.80
def service_level(n_calls, agents):
    a = n_calls * AHT / 1800                                        # offered load in erlangs
    if a <= 0:
        return 1.0
    if agents <= a:
        return 0.0
    b = 1.0
    for k in range(1, agents + 1):                                  # Erlang B by recursion, then Erlang C
        b = a * b / (k + a * b)
    pw = b / (1 - (a / agents) * (1 - b))
    return 1 - pw * exp(-(agents - a) * TGT_S / AHT)
def agents_for(n_calls):
    n = max(1, int(n_calls * AHT / 1800) + 1)
    while service_level(n_calls, n) < TGT:
        n += 1
    return n
STAFF = {}; rows = []
for q in CQ:
    ag = np.vectorize(agents_for)(FQc[q]); sl = np.vectorize(service_level)(ACTc, ag)
    STAFF[q] = (ag, sl)
    rows.append([f'P{q}', round(float(ag.sum() * 0.5 / C_TEST), 1), round(float((sl >= TGT).mean() * 100), 1),
                 round(float((sl * ACTc).sum() / ACTc.sum() * 100), 1), int(((sl < TGT).sum(axis=1) >= 6).sum())])
ag0 = np.vectorize(agents_for)(ACTc)
print('agent-hours a day if the calls were known in advance: %.1f' % (ag0.sum() * 0.5 / C_TEST))
curve = [[n, round(service_level(100, n) * 100, 1)] for n in range(17, 26)]
print('100 calls in a half hour: service level by agents', curve)
pd.DataFrame(rows, columns=['staffed to', 'agent-hours a day', '% of half hours at 80/20', '% of calls answered in 20 s',
                            'days with 6 or more half hours below target'])

# %%
# kept for the slides
wk0 = c_tr[c_tr.di >= ci - 7]
busy = int(np.argmax(np.where(c_te.groupby('di').is_holiday.max().values == 1, 0, r_am)))
APP['calls'] = {'n_train': len(c_tr), 'n_test': int(ACTc.size), 'mean': round(float(ACTc.mean()), 1), 'mase': c_mase, 'raw': c_raw,
                'week': {'calls': wk0.calls.tolist(), 'dow': wk0.dow.tolist()[::24], 'dates': [str(d.date()) for d in wk0.date][::24]},
                'up': up.astype(int).tolist(), 'up_n': int(up.sum()), 'up_pct': round(float(up.mean() * 100), 1),
                'day0': int((per_day == 0).sum()), 'day8': int((per_day >= 8).sum()), 'top_days': top_days,
                'ratio': [round(float(np.quantile(ratio, .05)), 2), round(float(np.quantile(ratio, .95)), 2)],
                'upd': {'before': round(mae_before, 2), 'after': round(mae_after, 2),
                        'helped': int((np.abs(ACTc[:, AM:] - upd).mean(axis=1) < np.abs(ACTc[:, AM:] - FQc[50][:, AM:]).mean(axis=1)).sum()),
                        'day': busy, 'r': round(float(r_am[busy]), 2), 'act': ACTc[busy].astype(int).tolist(),
                        'fc': FQc[50][busy].round(1).tolist(), 'new': upd[busy].round(1).tolist()},
                'staff': rows, 'known': round(float(ag0.sum() * 0.5 / C_TEST), 1), 'curve': curve,
                'act': ACTc.astype(int).tolist(), 'dates': [str(d.date()) for d in c_te.date.drop_duplicates()],
                'dow': c_te.groupby('di').dow.first().tolist(),
                'q': {str(q): {'f': FQc[q].round(0).astype(int).tolist(), 'ag': STAFF[q][0].tolist(),
                               'sl': (STAFF[q][1] * 100).round(0).astype(int).tolist()} for q in CQ}}
print('kept', list(APP['calls']))

# %% [build]
# everything the deck needs
j = SEGS.index(HEAD)
kind = np.select([cal.is_holiday == 1, cal.post_holiday == 1, cal.is_payday == 1, cal.since_payday.between(1, 3),
                  cal.is_dd_day == 1], ['bank holiday', 'day after holiday', 'payday', '3 days after payday',
                                        'direct-debit day'], 'ordinary day')
test_idx = np.array([o + h for o in ORIGINS for h in range(H)])
by_kind = {}
for k in FC:
    err = np.abs(FC[k][:, :, j] - ACT[:, :, j]).reshape(-1)
    by_kind[k] = {t: [round(float(err[kind[test_idx] == t].mean()), 2), int((kind[test_idx] == t).sum())]
                  for t in pd.unique(kind[test_idx])}
from statsmodels.tsa.stattools import acf
M = {
    'dates': [str(d.date()) for d in cal.date], 'segs': SEGS, 'head': HEAD,
    'y': {s: [round(float(v), 1) for v in Y[:, i]] for i, s in enumerate(SEGS)},
    'cal': {c: [int(v) for v in cal[c]] for c in ['dow', 'is_holiday', 'is_payday', 'is_dd_day', 'since_payday',
                                                 'post_holiday']},
    'n': n, 'TEST0': TEST0, 'VAL0': VAL0, 'ORIGINS': ORIGINS, 'H': H, 'ENC': ENC,
    'scale': [round(float(v), 2) for v in SCALE],
    'mae': {k: [round(float(v), 2) for v in RES[k][0]] for k in RES},
    'mase': {k: round(RES[k][1], 3) for k in RES},
    'mase_seg': {k: [round(float(v), 3) for v in RES[k][0] / SCALE] for k in RES},
    'time': {k: round(v, 1) for k, v in TIME.items()},
    'fc_head': {k: np.round(FC[k][:, :, j], 1).tolist() for k in FC},
    'q_head': np.round(Q[:, :, j, :], 1).tolist(),
    'by_kind': by_kind,
    'acf': [round(float(v), 3) for v in acf(Y[:TEST0, j], nlags=28)],
    'arima_params': {k: round(float(v), 3) for k, v in zip(arima_fit[HEAD].param_names, arima_fit[HEAD].params)},
    'arima_aic': round(float(arima_fit[HEAD].aic), 1),
    'prophet_parts': {c: [round(float(v), 2) for v in prophet_parts[c].values] for c in
                      ['trend', 'weekly', 'yearly', 'holidays', 'extra_regressors_additive', 'yhat']},
    'lstm': {'mase_seeds': [round(float(v), 3) for v in lstm_mase], 'params': int(lstm_params),
             'hist': [[round(a, 4), round(b, 4)] for a, b in lstm_runs[0][2]],
             'n_train': len(TR[0]), 'n_val': len(VA[0])},
    'tft': {'params': int(tft_params), 'epochs': int(trainer.current_epoch), 'cover': round(cover, 4),
            'above': round(above, 4), 'below': round(below, 4),
            'enc_imp': {k: float(v) for k, v in enc_imp.items()}, 'dec_imp': {k: float(v) for k, v in dec_imp.items()},
            'static_imp': dict(zip(tft.static_variables, [float(v) for v in (imp['static'] / imp['static'].sum() * 100).round(1)])),
            'attn': [round(float(v), 4) for v in attn],
            'cover_head': round(float(((ACT[:, :, j] >= Q[:, :, j, 0]) & (ACT[:, :, j] <= Q[:, :, j, 2])).mean()), 4),
            'above_head': round(float((ACT[:, :, j] > Q[:, :, j, 2]).mean()), 4)},
}
# liquidity: 14-day cumulative outflow of the headline segment against two buffers
cum_act = ACT[:, :, j].sum(axis=1)
M['liq'] = {'cum_act': cum_act.round(1).tolist(), 'cum_p50': Q[:, :, j, 1].sum(axis=1).round(1).tolist(),
            'cum_p90': Q[:, :, j, 2].sum(axis=1).round(1).tolist(),
            'breach_p50': int((cum_act > Q[:, :, j, 1].sum(axis=1)).sum()),
            'breach_p90': int((cum_act > Q[:, :, j, 2].sum(axis=1)).sum()),
            'day_breach_p90': int((ACT[:, :, j] > Q[:, :, j, 2]).sum()), 'days': int(ACT[:, :, j].size)}
M['app'] = APP
json.dump(M, open('build/m_ts.json', 'w'))
np.savez('build/ts_fc.npz', ACT=ACT, Q=Q, **FC)
print('wrote build/m_ts.json')
