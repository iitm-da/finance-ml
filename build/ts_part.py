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
