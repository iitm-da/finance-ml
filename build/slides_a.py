"""First half of the deck: time series forecasting."""
from fx import *

A = []
add = A.append
JSB = '<span class="jsbadge">browser JS</span>'
HEAD = TS['head']; SEGS = TS['segs']; J = SEGS.index(HEAD)
DATES = TS['dates']; Y = TS['y'][HEAD]; CAL = TS['cal']
n, TEST0, VAL0, ORIG, H, ENC = TS['n'], TS['TEST0'], TS['VAL0'], TS['ORIGINS'], TS['H'], TS['ENC']
MASE, MAE, BYK = TS['mase'], TS['mae'], TS['by_kind']
COL = {'naive': '#7C8C95', 'arima': BLUE, 'prophet': AMBER, 'arimax': SLATE, 'lstm': PURP, 'tft': GREEN}
NAME = {'naive': 'Seasonal naive', 'arima': 'ARIMA', 'prophet': 'Prophet', 'arimax': 'ARIMA + calendar',
        'lstm': 'LSTM', 'tft': 'TFT'}
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
idx = {d: i for i, d in enumerate(DATES)}
def dm(i):
    return DATES[i][8:] + ' ' + ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(DATES[i][5:7]) - 1]

def tags(ax, a0, a1, y):
    s = ''
    for i in range(a0, a1):
        tag = 'H' if CAL['is_holiday'][i] else 'P' if CAL['is_payday'][i] else 'D' if CAL['is_dd_day'][i] else ''
        if tag:
            c = {'H': RED, 'P': GREEN, 'D': AMBER}[tag]
            s += Rect(ax.X(i) - 7, y, 14, 14, c) + T(ax.X(i), y + 10.5, tag, 9.5, '#FFFFFF', 'middle', MONO, 700)
    return s

LEGEND = ('<span class="tag r">H</span>bank holiday &#160; <span class="tag g">P</span>payday &#160; '
          '<span class="tag o">D</span>direct-debit day')

def fcfig(k, models, w=620, h=300, band=False, back=28, label='forecast'):
    o = ORIG[k]; a0, a1 = o - back, o + H
    vals = Y[a0:a1] + [v for m in models for v in TS['fc_head'][m][k]]
    if band:
        vals += [q[0] for q in TS['q_head'][k]] + [q[2] for q in TS['q_head'][k]]
    lo = math.floor((min(vals) - 8) / 20) * 20; hi = math.ceil((max(vals) + 8) / 20) * 20
    ax = Ax(w, h, (a0, a1 - 1), (lo, hi), pad=(44, 12, 12, 46))
    step = 50 if hi - lo > 220 else 20
    s = ax.frame(yticks=range(lo, hi + 1, step), xticks=[(i, DATES[i][5:]) for i in range(a0, a1) if CAL['dow'][i] == 0], ylab='GBP million')
    s += Rect(ax.X(o - 0.5), ax.t, ax.X(a1 - 1) - ax.X(o - 0.5), ax.ph, '#F3F7F9')
    s += tags(ax, a0, a1, h - 26)
    if band:
        xs = list(range(o, o + H))
        s += ax.band(xs, [q[0] for q in TS['q_head'][k]], [q[2] for q in TS['q_head'][k]], GREEN, .18)
    s += ax.line(range(a0, o), Y[a0:o], GREY, 1.8)
    s += ax.line(range(o - 1, a1), Y[o - 1:a1], INK, 2.3)
    for j, m in enumerate(models):
        s += ax.line(range(o, o + H), TS['fc_head'][m][k], COL[m], 2.4)
        s += T(ax.l + 8 + j * 150, ax.t + 28, NAME[m], 11, COL[m], 'start', MONO, 700)
    s += L(ax.X(o - 0.5), ax.t, ax.X(o - 0.5), ax.t + ax.ph, INK, 1, 'stroke-dasharray="3 3"')
    s += T(ax.l + ax.pw - 6, ax.t + 14, 'black: actual', 10.5, INK, 'end', MONO, 700)
    return svg(w, h, s, label)

def window_mae(m, k):
    o = ORIG[k]
    return sum(abs(f - a) for f, a in zip(TS['fc_head'][m][k], Y[o:o + H])) / H

def kindbars(models, w=640, label_w=170):
    kinds = ['ordinary day', 'payday', '3 days after payday', 'direct-debit day', 'bank holiday', 'day after holiday']
    bh, gap = (34, 22) if len(models) == 1 else (15, 12)
    mx = max(BYK[m][kd][0] for m in models for kd in kinds)
    span = w - label_w - 70
    s = ''; y = 6
    for kd in kinds:
        s += T(label_w - 10, y + (bh * len(models)) / 2 + 4, kd, 12, INK, 'end', MONO, 500)
        s += T(label_w - 10, y + (bh * len(models)) / 2 + 17, f'{BYK[models[0]][kd][1]} days', 9.5, SOFT, 'end', MONO)
        for m in models:
            v = BYK[m][kd][0]
            s += Rect(label_w, y, max(2, span * v / mx), bh - 2, COL[m])
            s += T(label_w + span * v / mx + 6, y + bh / 2 + 3.5, f'{v:.1f}', 11.5, INK, 'start', MONO, 700)
            y += bh
        y += gap + (8 if len(models) == 1 else 6)
    return svg(w, y + 2, s, 'MAE by kind of day')

K = 1                       # the window used for the static forecast figures: 17 to 30 July 2025

# ================================================================ cover
add(f'''
<section class="slide cover" data-sec="Cover">
  <div class="coverpat">
    <div><span class="h-dim">seasonal naive   MASE </span><span class="h-bad">{MASE['naive']:.3f}</span></div>
    <div><span class="h-dim">ARIMA            MASE </span><span class="h-val">{MASE['arima']:.3f}</span></div>
    <div><span class="h-dim">Prophet          MASE </span><span class="h-val">{MASE['prophet']:.3f}</span></div>
    <div><span class="h-dim">LSTM             MASE </span><span class="h-val">{MASE['lstm']:.3f}</span></div>
    <div><span class="h-dim">TFT              MASE </span><span class="h-val">{MASE['tft']:.3f}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">train on real       PR-AUC </span><span class="h-val">{GN['util']['trtr'][1]:.3f}</span></div>
    <div><span class="h-dim">train on synthetic  PR-AUC </span><span class="h-bad">{GN['util']['tstr'][1]:.3f}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">10 alerts a day &#8594; </span><span class="h-val">{GN['fraud']['budget']['10']['gb']['caught']}</span><span class="h-dim"> of {GN['n']['fraud_test']} fraud rows</span></div>
  </div>
  <div class="kicker">Lloyds Bank</div>
  <h1>Forecasting and<br/>Generative Models</h1>
  <div class="pt">Time series forecasting for finance<br/>Generative models, applied</div>
  <div class="rule"></div>
  <p class="sub">ARIMA, Prophet, LSTM and the Temporal Fusion Transformer on one bank&#8217;s cash flow, then a GAN
  and a fraud detection pipeline on its card transactions.</p>
  <div class="by"><span>Instructor</span>Dr. Arun B Ayyar</div>
  <table class="meta">
    <tr><td>Time series with</td><td>Dr. Nirav Bhatt</td></tr>
    <tr><td>Duration</td><td>90 minutes</td></tr>
    <tr><td>Data</td><td>simulated bank &#183; 6 segments &#183; {comma(GN['n']['tx'])} transactions</td></tr>
    <tr><td>Code</td><td>lloyds_forecasting_and_generative_models.ipynb</td></tr>
  </table>
</section>
''')

# ================================================================ plan
blocks = [('Baselines', 15, BLUE, ['seasonal naive', 'ARIMA', 'Prophet']),
          ('Deep learning forecasting', 25, PURP, ['LSTM', 'Temporal Fusion Transformer', 'quantile forecasts']),
          ('Applications', 10, SLATE, ['cash flow', 'demand and market', 'liquidity risk']),
          ('GANs for synthetic data', 25, RED, ['generator and critic', 'fidelity, utility, privacy']),
          ('Fraud pipeline', 15, GREEN, ['alert budget', 'synthetic fraud rows'])]
s = ''; x = 10; tot = 90; wid = 1100; t = 0
for name, mins, col, items in blocks:
    w = wid * mins / tot
    s += Rect(x, 30, w - 4, 54, col)
    s += T(x + 10, 53, name, 13.5, '#FFFFFF', 'start', BODY, 700)
    s += T(x + 10, 72, f'{mins} min', 11, '#FFFFFF', 'start', MONO)
    s += T(x, 20, f'{t}', 10.5, SOFT, 'start', MONO)
    for i, it in enumerate(items):
        s += T(x + 10, 108 + i * 19, it, 12, INK, 'start', MONO)
    x += w; t += mins
s += T(1106, 20, '90', 10.5, SOFT, 'end', MONO)
s += Rect(10, 176, wid * 50 / 90 - 4, 30, PALE) + T(20, 196, 'data: daily net outflow, 6 segments, 2022 to 2025', 12, INK, 'start', MONO, 500)
s += Rect(10 + wid * 50 / 90, 176, wid * 40 / 90 - 4, 30, PALE) + T(20 + wid * 50 / 90, 196, f"data: {comma(GN['n']['tx'])} card transactions, 180 days", 12, INK, 'start', MONO, 500)
add(slide('Plan', 'Plan', 'Plan for the 90 minutes',
          'Five blocks on one simulated bank. Every number on the slides comes from the notebook, and the notebook runs on a laptop CPU.',
          fig(svg(1120, 214, s, 'Session plan'), 'minutes from the start') +
          '<div class="cols nat"><div class="callout"><b>Forecasting.</b> Each model forecasts the next 14 days and is scored on the same 182 held-back days, so the numbers on different slides can be compared directly.</div>'
          '<div class="callout"><b>Generative models.</b> A GAN writes a synthetic copy of the transaction table. The copy is tested for fidelity, utility and privacy, then used in a fraud detection pipeline.</div></div>'
          '<div class="callout warn sm">All data is simulated; nothing comes from a real bank. The generator, <code class="k">make_data.py</code>, is in the folder, and its effects (weekday, payday, bank holidays, direct debits) are stated on the slides that use them.</div>'))

# ================================================================ the series
ax = Ax(1140, 318, (0, n - 1), (20, 260), pad=(46, 12, 12, 30))
yr = [(idx[f'{y}-01-01'], str(y)) for y in (2022, 2023, 2024, 2025)]
s = ax.frame(yticks=[50, 100, 150, 200, 250], xticks=yr, ylab='GBP million a day')
s += Rect(ax.X(VAL0), ax.t, ax.X(TEST0) - ax.X(VAL0), ax.ph, '#FFF6E0')
s += Rect(ax.X(TEST0), ax.t, ax.X(n - 1) - ax.X(TEST0), ax.ph, '#FDE7E9')
s += ax.line(range(n), Y, BLUE, 1)
s += T(ax.X(VAL0) + 4, ax.t + 14, 'validation', 10.5, AMBER, 'start', MONO, 700)
s += T(ax.X(TEST0) + 4, ax.t + 14, 'test: 182 days', 10.5, RED, 'start', MONO, 700)
s += T(ax.X(VAL0) - 6, ax.t + 14, f'training: {VAL0:,} days', 10.5, BLUE, 'end', MONO, 700)
add(slide('Data', 'Data &#183; The series', 'Daily net outflow, retail current accounts',
          f"{n:,} days from 1 January 2022 to 31 December 2025. Positive values are money leaving the bank. Five more segments have the same shape of table.",
          fig(svg(1140, 318, s, 'Daily net outflow 2022 to 2025'), 'net outflow, retail current accounts',
              f"Models are fitted on the days before {dm(TEST0)} 2025 and scored on the days after.") +
          '<div class="kv">' + tile('segments', '6') + tile('forecast horizon', '14 days') +
          tile('history a model reads', '56 days') + tile('test forecasts per model', f'{13 * 14 * 6:,}') + '</div>'))

# ================================================================ eight weeks
a0 = idx['2025-04-07']; a1 = a0 + 56
ax = Ax(1140, 330, (a0, a1 - 1), (20, 240), pad=(46, 12, 12, 46))
s = ax.frame(yticks=[50, 100, 150, 200], xticks=[(i, DATES[i][5:]) for i in range(a0, a1) if CAL['dow'][i] == 0], ylab='GBP million a day')
wk = ''
for i in range(a0, a1):
    if CAL['dow'][i] >= 5:
        wk += Rect(ax.X(i - 0.5), ax.t, ax.X(i + 0.5) - ax.X(i - 0.5), ax.ph, '#EEF2F4')
def vmark(i, col):
    return L(ax.X(i), ax.t, ax.X(i), ax.t + ax.ph, col, 1.2, 'stroke-dasharray="3 3"')
pay = ''.join(vmark(i, GREEN) for i in range(a0, a1) if CAL['is_payday'][i])
dd = ''.join(vmark(i, AMBER) for i in range(a0, a1) if CAL['is_dd_day'][i])
hol = ''.join(vmark(i, RED) for i in range(a0, a1) if CAL['is_holiday'][i])
def tagk(key, tag, c):
    return ''.join(Rect(ax.X(i) - 7, 304, 14, 14, c) + T(ax.X(i), 314.5, tag, 9.5, '#FFFFFF', 'middle', MONO, 700)
                   for i in range(a0, a1) if CAL[key][i])
s += G_(1, wk + T(ax.X(a0 + 5.5), ax.t + ax.ph - 8, 'weekends', 10.5, SOFT, 'middle', MONO, 700))
s += G_(2, pay + tagk('is_payday', 'P', GREEN)) + G_(3, dd + tagk('is_dd_day', 'D', AMBER)) + G_(4, hol + tagk('is_holiday', 'H', RED))
s += ax.line(range(a0, a1), Y[a0:a1], BLUE, 2.2)
for i in range(a0, a1):
    s += ax.dot(i, Y[i], BLUE, 2.4)
pd_i = [i for i in range(a0, a1) if CAL['is_payday'][i]][0]
dd_i = [i for i in range(a0, a1) if CAL['is_dd_day'][i]][0]
ho_i = [i for i in range(a0, a1) if CAL['is_holiday'][i]][0]
add(slide('Data', 'Data &#183; Eight weeks', 'Eight weeks of the series',
          f"7 April to 1 June 2025. Four calendar effects sit on top of the weekly cycle, and each press adds one.",
          fig(svg(1140, 330, s, 'Eight weeks of daily net outflow'), LEGEND) +
          '<div class="cols4">' +
          f'<div class="card sm" data-at="1"><h3>Weekly cycle</h3><p>Outflow is lowest on Saturday and Sunday and highest on Friday.</p></div>'
          f'<div class="card sm g" data-at="2"><h3>Payday</h3><p>Salaries land on the 25th, or the last working day before it: {Y[pd_i]:.0f} on {dm(pd_i)}. Spending rises for three days after.</p></div>'
          f'<div class="card sm o" data-at="3"><h3>Direct-debit day</h3><p>Rent and bills leave on the first working day of the month: {Y[dd_i]:.0f} on {dm(dd_i)}.</p></div>'
          f'<div class="card sm r" data-at="4"><h3>Bank holidays</h3><p>A holiday behaves like a Sunday ({Y[ho_i]:.0f} on {dm(ho_i)}) and the held-over payments settle the next working day.</p></div>'
          '</div>' + bld(4), 'build'))

# ================================================================ scoring: rolling origin
s = ''
X0, X1 = 20, 1100
tw = (X1 - X0) / 182
s += T(X0, 16, f'test period, {dm(TEST0)} to {dm(n - 1)} 2025', 10.5, SOFT, 'start', MONO)
for k in range(13):
    piece = Rect(X0 + k * 14 * tw + 1, 26, 14 * tw - 2, 34, RED if k % 2 == 0 else '#D98A92')
    piece += T(X0 + k * 14 * tw + 7 * tw, 48, str(k + 1), 12, '#FFFFFF', 'middle', MONO, 700)
    s += G_(0 if k == 0 else 1 if k == 1 else 2, piece)
s += G_(0, Rect(X0 - 0, 70, 0.1, 0.1, 'none') + T(X0 + 7 * tw, 78, '14 days', 10, RED, 'middle', MONO) +
        T(X0, 104, 'window 1: the model reads the 56 days before 3 July and forecasts 3 to 16 July', 11.5, INK, 'start', MONO))
s += G_(1, T(X0, 124, 'window 2: it is handed the history up to 16 July and forecasts the next 14 days. Its weights are not refitted.', 11.5, INK, 'start', MONO))
s += G_(2, T(X0, 144, '13 windows x 14 days x 6 segments = 1,092 forecast days per model', 11.5, INK, 'start', MONO, 700))
rfig = svg(1120, 154, s, 'Rolling-origin evaluation')
add(slide('Scoring', 'Scoring &#183; Concept', 'Concept',
          'Rolling-origin evaluation: forecast 14 days, move the origin forward 14 days, repeat. The model always forecasts days it has not seen.',
          fig(rfig, 'thirteen forecast windows') +
          '<div class="cols"><div class="colstack">' +
          out("""MAE  = mean |actual - forecast|            in GBP million
MASE = MAE / MAE of the seasonal naive forecast
             on the training days""", '', 'the two scores') +
          '</div><div class="colstack">'
          '<div class="callout"><b>MAE</b> is in pounds, so a treasury desk can read it. It cannot be averaged across segments of different size.</div>'
          '<div class="callout"><b>MASE</b> has no unit. Below 1, the model beats repeating last week&#8217;s value for the same weekday. Net outflow can be 0 or negative, which rules out percentage errors.</div>'
          '</div></div>' + bld(2), 'build'))

# ================================================================ seasonal naive on one window
o = ORIG[0]
fa = TS['fc_head']['naive'][0]; ac = Y[o:o + H]
errs = [abs(f - a) for f, a in zip(fa, ac)]
lo = math.floor(min(fa + ac) / 20) * 20 - 20; hi = math.ceil(max(fa + ac) / 20) * 20
ax = Ax(1140, 300, (-0.6, 13.6), (lo, hi), pad=(46, 12, 12, 62))
s = ax.frame(yticks=range(lo, hi + 1, 40), ylab='GBP million')
for h_ in range(H):
    s += L(ax.X(h_), ax.Y(ac[h_]), ax.X(h_), ax.Y(fa[h_]), RED, 2.2)
    s += ax.dot(h_, fa[h_], COL['naive'], 6) + ax.dot(h_, ac[h_], INK, 6)
    s += T(ax.X(h_) + 10, (ax.Y(ac[h_]) + ax.Y(fa[h_])) / 2 + 4, f'{errs[h_]:.1f}', 11, RED, 'start', MONO, 700)
    s += T(ax.X(h_), 258, DOW[CAL['dow'][o + h_]], 10.5, INK, 'middle', MONO, 700)
    s += T(ax.X(h_), 272, dm(o + h_), 9.5, SOFT, 'middle', MONO)
    s += T(ax.X(h_), 290, f'{errs[h_]:.1f}', 10.5, RED, 'middle', MONO)
s += T(ax.l + ax.pw - 6, ax.t + 14, 'black: actual', 10.5, INK, 'end', MONO, 700)
s += T(ax.l + ax.pw - 6, ax.t + 28, 'grey: same weekday, last observed week', 10.5, COL['naive'], 'end', MONO, 700)
add(slide('Scoring', 'Scoring &#183; Example', 'Example',
          f"The seasonal naive forecast for window 1. The 14 errors below average {sum(errs) / H:.1f}; over all 13 windows the segment&#8217;s MAE is {MAE['naive'][J]:.1f}.",
          fig(svg(1140, 300, s, 'Seasonal naive forecast, window 1'), 'retail current accounts, 3 to 16 July 2025: red bars are the absolute errors',
              f"Sum of the 14 errors {sum(errs):.1f}, divided by 14 = {sum(errs) / H:.1f}.") +
          '<div class="cols nat">' +
          code("""def seasonal_naive(o):       # o = first forecast day
    return [y[o + h - 7 * (h // 7 + 1)] for h in range(14)]""", 'sm') +
          f"<div class=\"callout sm\">Across the six segments its MASE is {MASE['naive']:.3f}, a little above 1: days 8 to 14 of each window reuse a value that is two weeks old. Every model after this slide is judged against that number.</div>"
          '</div>'))

# ================================================================ ARIMA concept
acf = TS['acf']
ax = Ax(560, 380, (0.4, 28.6), (-0.4, 0.7), pad=(40, 12, 12, 30))
s = ax.frame(yticks=[-0.2, 0, 0.2, 0.4, 0.6], xticks=[(v, str(v)) for v in (1, 7, 14, 21, 28)], ylab='autocorrelation', xlab='lag in days')
for lag in range(1, 29):
    s += ax.vbar(lag - 0.32, lag + 0.32, acf[lag], RED if lag % 7 == 0 else '#9FB0B9', base=0)
s += L(ax.l, ax.Y(0), ax.l + ax.pw, ax.Y(0), INK, 1)
PA = TS['arima_params']
add(slide('Baselines', 'ARIMA &#183; Concept', 'Concept',
          f"ARIMA forecasts a series from its own past values and its own past errors. The outflow on a given day correlates {acf[7]:.2f} with the day one week earlier.",
          '<div class="cols">' + fig(svg(560, 380, s, 'Autocorrelation of the series'), 'autocorrelation of daily net outflow, training days',
                                    f"Lags 7, 14, 21 and 28 stand out: {acf[7]:.2f}, {acf[14]:.2f}, {acf[21]:.2f}, {acf[28]:.2f}.") +
          '<div class="colstack fill">' +
          card('AR: autoregression', 'Today is a weighted sum of the last p days. Here p = 2.', 'b') +
          card('I: integration', 'Difference the series until it stops drifting. Here one seasonal difference: each day minus the same weekday a week earlier.', 'g') +
          card('MA: moving average', 'Today also depends on the last q forecast errors. Here q = 1, plus one seasonal error term at lag 7.', 'o') +
          '<div class="callout sm">Written ARIMA(2,0,1)(0,1,1)<sub>7</sub>: five numbers to estimate for each segment.</div>'
          '</div></div>'))

# ================================================================ ARIMA example
ptxt = '\n'.join(f'{k:10s} {v:8.3f}' for k, v in PA.items())
add(slide('Baselines', 'ARIMA &#183; Example', 'Example',
          f"One model per segment, fitted once in {TS['time']['arima']:.0f} seconds for all six. MASE {MASE['arima']:.3f}, against {MASE['naive']:.3f} for the seasonal naive forecast.",
          '<div class="cols w4060"><div class="colstack">' +
          code("""fit = SARIMAX(y[:TEST0], order=(2, 0, 1),
              seasonal_order=(0, 1, 1, 7)).fit()

# at each origin: new history, same weights
fit.apply(y[:o]).forecast(14)""", 'sm') +
          out(ptxt, 'xs', 'fitted parameters, retail current accounts') +
          '</div>' +
          fig(fcfig(K, ['arima'], w=660, h=400), f'window 2, {dm(ORIG[K])} to {dm(ORIG[K] + 13)} 2025 &#183; {LEGEND}',
              f"MAE in this window {window_mae('arima', K):.1f}. The forecast repeats the weekly shape and has no way to know that {dm([i for i in range(ORIG[K], ORIG[K] + H) if CAL['is_payday'][i]][0])} is payday.") +
          '</div>'))

# ================================================================ ARIMA drawbacks
bk = BYK['arima']
add(slide('Baselines', 'ARIMA &#183; Drawbacks', 'Drawbacks',
          f"ARIMA&#8217;s error on an ordinary day is {bk['ordinary day'][0]:.1f}. On payday it is {bk['payday'][0]:.1f} and on direct-debit day {bk['direct-debit day'][0]:.1f}.",
          '<div class="cols w6040">' + fig(kindbars(['arima']), 'MAE by kind of day, retail current accounts, 182 test days') +
          '<div class="colstack">'
          '<ul class="b lg"><li>The model sees one column: the series itself. A payday that moves between the 23rd and the 25th has no fixed lag, so no AR or MA term can catch it.</li>'
          '<li>It is linear. An effect that grows with the size of the book is outside what it can represent.</li>'
          '<li>Each segment gets its own model. What the corporate segment shows about bank holidays teaches the SME model nothing.</li></ul>'
          f"<div class=\"callout good sm\">The first drawback has a classical fix: pass the calendar as regressors. That model, <b>ARIMA + calendar</b>, scores {MASE['arimax']:.3f} and returns on the verdict slide.</div>"
          '</div></div>'))

# ================================================================ Prophet concept
P = TS['prophet_parts']
def pmini(key, title, col, yr_):
    ax = Ax(560, 92, (a0, a1 - 1), yr_, pad=(46, 6, 18, 4))
    s = Rect(ax.l, ax.t, ax.pw, ax.ph, '#FFFFFF', HAIR, 1)
    s += L(ax.l, ax.Y(0), ax.l + ax.pw, ax.Y(0), HAIR, 1) if yr_[0] < 0 < yr_[1] else ''
    s += ax.line(range(a0, a1), P[key][a0:a1], col, 2)
    s += T(ax.l, 12, title, 10.5, col, 'start', MONO, 700)
    s += T(ax.l - 6, ax.t + 10, f'{yr_[1]}', 9.5, SOFT, 'end', MONO) + T(ax.l - 6, ax.t + ax.ph - 2, f'{yr_[0]}', 9.5, SOFT, 'end', MONO)
    return svg(560, 92, s, title)
def rng_(key):
    v = P[key][a0:a1]
    return (math.floor(min(v) / 10) * 10 - 10, math.ceil(max(v) / 10) * 10 + 10)
pm = (pmini('trend', 'trend', BLUE, rng_('trend')) + pmini('weekly', 'weekly seasonality', GREEN, rng_('weekly')) +
      pmini('yearly', 'yearly seasonality', PURP, rng_('yearly')) +
      pmini('holidays', 'bank holidays', RED, rng_('holidays')) +
      pmini('extra_regressors_additive', 'payday and direct-debit regressors', AMBER, rng_('extra_regressors_additive')))
add(slide('Baselines', 'Prophet &#183; Concept', 'Concept',
          'Prophet writes the series as a sum of curves against the date: a trend, a weekly and a yearly cycle, and a jump for each named event.',
          '<div class="cols">' + fig(pm, 'the fitted pieces, 7 April to 1 June 2025', 'Add the five curves and the result is the forecast.') +
          '<div class="colstack">' +
          code("""m = Prophet(weekly_seasonality=True,
            yearly_seasonality=True,
            holidays=bank_holidays)
for r in ['is_payday', 'sp1', 'sp2', 'sp3',
          'is_dd_day', 'post_holiday']:
    m.add_regressor(r)
m.fit(df[:TEST0])""", 'sm') +
          '<ul class="b"><li>Every piece can be plotted and explained to someone outside the team.</li>'
          '<li>Events are inputs. An analyst who knows about payday adds one line.</li>'
          '<li>The forecast for a date depends on the date alone. Yesterday&#8217;s outflow is not an input.</li></ul>'
          '</div></div>'))

# ================================================================ Prophet example + drawback
act_t = [Y[o_ + h_] for o_ in ORIG for h_ in range(H)]
def lag1(m):
    r = [a - f for a, f in zip(act_t, [v for row in TS['fc_head'][m] for v in row])]
    mu = sum(r) / len(r)
    num = sum((r[i] - mu) * (r[i - 1] - mu) for i in range(1, len(r)))
    return num / sum((v - mu) ** 2 for v in r)
r1p = lag1('prophet')
add(slide('Baselines', 'Prophet &#183; Example', 'Example',
          f"Fitted in {TS['time']['prophet']:.0f} seconds for six segments. MASE {MASE['prophet']:.3f}; its payday error is {BYK['prophet']['payday'][0]:.1f} against ARIMA&#8217;s {BYK['arima']['payday'][0]:.1f}.",
          '<div class="cols w6040">' +
          fig(fcfig(K, ['arima', 'prophet'], w=680, h=400), f'window 2, {dm(ORIG[K])} to {dm(ORIG[K] + 13)} 2025 &#183; {LEGEND}',
              f"MAE in this window: ARIMA {window_mae('arima', K):.1f}, Prophet {window_mae('prophet', K):.1f}.") +
          '<div class="colstack fill">' +
          '<div class="kv" style="grid-template-columns:1fr 1fr">' + tile('Prophet MASE', f"{MASE['prophet']:.3f}", 'ok') + tile('ARIMA MASE', f"{MASE['arima']:.3f}", 'b') + '</div>' +
          f"<div class=\"callout warn sm\"><b>Drawback.</b> Prophet&#8217;s error on one day correlates {r1p:.2f} with its error the day before. When outflow ran high yesterday it will run high today, and the model has no term that uses this.</div>"
          '<div class="callout sm"><b>Drawback.</b> Effects add. A payday effect that is larger on a Friday, or larger in a bigger book, has to be entered by hand as another regressor.</div>'
          '</div></div>'))

# ================================================================ baseline verdict
items = [(NAME[m], MASE[m], COL[m], '') for m in ('naive', 'arima', 'prophet', 'arimax')]
vfig = hbars(items, w=700, label_w=180, bar_h=40, gap=18, maxv=1.1, label='MASE of the baselines', valfmt=lambda v: f'{v:.3f}')
add(slide('Baselines', 'Baselines &#183; Verdict', 'Verdict',
          f"Handing the calendar to ARIMA takes its MASE from {MASE['arima']:.3f} to {MASE['arimax']:.3f}. That is the number the deep models have to beat.",
          '<div class="cols w6040">' + fig(vfig, 'MASE over 13 windows and 6 segments, lower is better',
                                          'Same test days, same 14-day horizon, weights fitted once.') +
          '<div class="colstack fill">' +
          card('What the calendar bought', f"ARIMA with regressors for bank holidays, payday and direct debits cuts the error by {(1 - MASE['arimax'] / MASE['arima']) * 100:.0f}%. Knowing the business calendar was worth more than the choice of model.", 'g') +
          card('What it cost', 'Somebody had to know that payday moves to the previous working day and that spending follows it for three days, and write each effect as a column.', 'o') +
          card('The question for deep learning', 'Can a network learn those effects, and their interactions, from windows of data and a plain calendar?', 'p') +
          '</div></div>'))

# ================================================================ LSTM concept: windows
s = ''
c = 11
def strip(y, start, label, k):
    p = ''
    for i in range(96):
        col = '#DCE4E8'
        if start <= i < start + 56:
            col = BLUE
        elif start + 56 <= i < start + 70:
            col = RED
        p += Rect(170 + i * c, y, c - 2, 22, col)
    p += T(160, y + 16, label, 11.5, INK, 'end', MONO, 700)
    return G_(k, p)
s += strip(30, 0, 'window 1', 0) + strip(62, 1, 'window 2', 1) + strip(94, 2, 'window 3', 1) + strip(126, 20, 'window 21', 2)
s += G_(0, T(170 + 28 * c, 20, 'encoder: 56 days of outflow + calendar', 10.5, BLUE, 'middle', MONO, 700) +
        T(170 + 63 * c, 20, 'target: next 14 days', 10.5, RED, 'middle', MONO, 700))
s += G_(2, T(170, 176, f"{TS['lstm']['n_train'] // 6:,} start days x 6 segments = {TS['lstm']['n_train']:,} training windows", 12, INK, 'start', MONO, 700))
add(slide('LSTM', 'LSTM &#183; Concept', 'Concept',
          'Forecasting becomes supervised learning: slide a 70-day window along each series, give the network the first 56 days and ask for the last 14.',
          fig(svg(1240, 186, s, 'Sliding windows over a series'), 'one square is one day') +
          '<div class="cols3">' +
          card('One model, six segments', 'Windows from every segment go into the same network, with the segment as an input. Each series is scaled by its own mean and standard deviation first.', 'b') +
          card('The calendar is an input', 'Weekday, bank holiday, payday, days since payday and direct-debit day, for the 56 past days and the 14 future ones. The future calendar is known today.', 'g') +
          card('No effect is written by hand', 'The network is not told that payday lowers outflow or that spending follows it. It has to find that in the windows.', 'p') +
          '</div>' + bld(2), 'build'))

# ================================================================ LSTM mechanism: the cell
def circ(x, y, sym, col=INK):
    return (f'<circle cx="{x}" cy="{y}" r="14" fill="#FFFFFF" stroke="{col}" stroke-width="1.8"/>' +
            T(x, y + 5, sym, 15, col, 'middle', MONO, 700))
def gate(x, y, lab, col):
    return Rect(x - 24, y - 15, 48, 30, '#FFFFFF', col, 1.8) + T(x, y + 4.5, lab, 12, col, 'middle', MONO, 700)
s = Rect(150, 30, 560, 250, '#FFFFFF', LINE, 1.2)
# step 0: the cell state
s += G_(0, L(40, 70, 800, 70, INK, 2.4) + f'<path d="M800,70 L789,65 L789,75 z" fill="{INK}"/>' +
        T(40, 58, 'c(t-1)', 12, INK, 'start', MONO, 700) + T(800, 58, 'c(t)', 12, INK, 'end', MONO, 700) +
        T(430, 22, 'cell state: the memory carried along the sequence', 11, INK, 'middle', MONO, 700) +
        L(40, 250, 180, 250, SOFT, 1.8) + T(40, 240, 'h(t-1)', 12, SOFT, 'start', MONO, 700) +
        L(110, 310, 110, 250, SOFT, 1.8) + T(118, 306, 'x(t): today&#39;s inputs'.replace('&#39;', "'"), 12, SOFT, 'start', MONO, 700) +
        L(180, 250, 580, 250, SOFT, 1.8))
# step 1: forget gate
s += G_(1, L(240, 250, 240, 185, BLUE, 1.8) + gate(240, 170, 'σ', BLUE) + L(240, 155, 240, 84, BLUE, 1.8) + circ(240, 70, '×', BLUE) +
        T(240, 270, 'forget gate f', 11, BLUE, 'middle', MONO, 700) + T(250, 122, 'keep how much', 10, BLUE, 'start', MONO) +
        T(250, 135, 'of the old memory', 10, BLUE, 'start', MONO))
# step 2: input gate + candidate
s += G_(2, L(380, 250, 380, 185, GREEN, 1.8) + gate(380, 170, 'σ', GREEN) + L(460, 250, 460, 185, GREEN, 1.8) + gate(460, 170, 'tanh', GREEN) +
        L(380, 155, 380, 126, GREEN, 1.8) + L(460, 155, 460, 126, GREEN, 1.8) + L(380, 126, 460, 126, GREEN, 1.8) + circ(420, 126, '×', GREEN) +
        L(420, 112, 420, 84, GREEN, 1.8) + circ(420, 70, '+', GREEN) +
        T(372, 270, 'input gate i', 11, GREEN, 'middle', MONO, 700) + T(472, 270, 'candidate g', 11, GREEN, 'middle', MONO, 700) +
        T(436, 104, 'write i x g', 10, GREEN, 'start', MONO))
# step 3: output gate
s += G_(3, L(580, 250, 580, 185, RED, 1.8) + gate(580, 170, 'σ', RED) + L(670, 70, 670, 111, RED, 1.8) + gate(670, 126, 'tanh', RED) +
        L(604, 170, 656, 170, RED, 1.8) + L(670, 141, 670, 156, RED, 1.8) + circ(670, 170, '×', RED) +
        L(670, 184, 670, 250, RED, 1.8) + L(670, 250, 800, 250, RED, 2.4) + f'<path d="M800,250 L789,245 L789,255 z" fill="{RED}"/>' +
        T(800, 240, 'h(t)', 12, RED, 'end', MONO, 700) + T(580, 270, 'output gate o', 11, RED, 'middle', MONO, 700) +
        T(760, 270, 'what the cell shows', 10.5, RED, 'middle', MONO))
cellfig = svg(840, 322, s, 'An LSTM cell')
add(slide('LSTM', 'LSTM &#183; Mechanism', 'Mechanism',
          'One LSTM cell processes one day. Three gates, each a vector of numbers between 0 and 1, decide what the memory keeps, takes in and shows.',
          '<div class="cols w7030">' + fig(cellfig, 'one cell, one time step') +
          '<div class="colstack">' +
          out("""f = σ(W_f · [h(t-1), x(t)])       forget
i = σ(W_i · [h(t-1), x(t)])       input
g = tanh(W_g · [h(t-1), x(t)])    candidate
o = σ(W_o · [h(t-1), x(t)])       output

c(t) = f × c(t-1) + i × g
h(t) = o × tanh(c(t))""", '', 'the six lines') +
          '<div class="callout sm">The cell state changes by a multiplication and an addition, so a value written on day 1 can survive to day 56. A plain recurrent network rewrites its whole state every step and loses it.</div>'
          f"<div class=\"callout sm\">With a hidden size of 64, each gate is 64 numbers. The whole forecasting network on the next slide has {TS['lstm']['params']:,} weights.</div>"
          '</div></div>' + bld(3), 'build'))

# ================================================================ LSTM mechanism: encoder-decoder
s = ''
for i, lab in enumerate(['t-56', 't-55', '...', 't-2', 't-1']):
    x = 30 + i * 92
    s += box(x, 86, 76, 50, 'LSTM', '', BLUE, tsize=12) + T(x + 38, 166, lab, 10.5, SOFT, 'middle', MONO)
    s += arrow(x + 38, 150, x + 38, 138)
    if i:
        s += arrow(x - 16, 111, x, 111, BLUE)
s += T(30, 190, 'in: outflow, calendar, segment', 10.5, BLUE, 'start', MONO, 700)
s += arrow(474, 111, 540, 111, INK, 2.2) + T(507, 100, 'state', 10.5, INK, 'middle', MONO, 700)
for i, lab in enumerate(['t+1', 't+2', '...', 't+14']):
    x = 540 + i * 92
    s += box(x, 86, 76, 50, 'LSTM', '', RED, tsize=12) + T(x + 38, 166, lab, 10.5, SOFT, 'middle', MONO)
    s += arrow(x + 38, 150, x + 38, 138) + arrow(x + 38, 86, x + 38, 52)
    s += T(x + 38, 44, 'ŷ', 14, RED, 'middle', MONO, 700)
    if i:
        s += arrow(x - 16, 111, x, 111, RED)
s += T(540, 190, 'in: calendar of that day, segment', 10.5, RED, 'start', MONO, 700)
s += T(540, 22, 'out: one forecast per day', 10.5, RED, 'start', MONO, 700)
add(slide('LSTM', 'LSTM &#183; Mechanism', 'Mechanism continued 1',
          'The forecaster is two LSTMs. The encoder reads 56 days and hands its final state to the decoder, which steps through the next 14 days of calendar.',
          fig(svg(1160, 198, f'<g transform="translate(120,0)">{s}</g>', 'Encoder-decoder LSTM'), 'encoder in blue, decoder in red') +
          '<div class="cols w6040">' +
          code("""class Seq2Seq(nn.Module):
    def __init__(self, n_known=17, hidden=64, n_seg=6):
        super().__init__()
        self.emb = nn.Embedding(n_seg, 4)
        self.enc = nn.LSTM(1 + n_known + 4, hidden, batch_first=True)
        self.dec = nn.LSTM(n_known + 4, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, xe, xd, sid):
        e = self.emb(sid)                                  # segment -> 4 numbers
        _, state = self.enc(cat(xe, e))                    # read the past
        d, _ = self.dec(cat(xd, e), state)                 # step through the future
        return self.out(d).squeeze(-1)                     # 14 forecasts""", 'xs') +
          '<div class="colstack"><div class="callout sm">17 calendar inputs: 7 for the weekday, then bank holiday, payday, three flags for the days after payday, direct-debit day, day after a holiday, and three for the time of year.</div>'
          '<div class="callout sm">Loss: mean absolute error on the scaled outflow. Training stops when the validation loss has not improved for 8 epochs.</div></div>'
          '</div>'))

# ================================================================ LSTM example
hist = TS['lstm']['hist']
ax = Ax(540, 400, (1, len(hist)), (0, max(h_[0] for h_ in hist[1:]) * 1.6), pad=(46, 12, 12, 30))
ytk = [round(v, 2) for v in (0.1, 0.2, 0.3, 0.4, 0.5) if v < ax.yr[1]]
s = ax.frame(yticks=ytk, xticks=[(v, str(v)) for v in range(5, len(hist) + 1, 5)], ylab='MAE on the scaled series', xlab='epoch')
s += ax.line(range(1, len(hist) + 1), [min(h_[0], ax.yr[1]) for h_ in hist], BLUE, 2.2) + ax.line(range(1, len(hist) + 1), [min(h_[1], ax.yr[1]) for h_ in hist], RED, 2.2)
s += T(ax.l + ax.pw - 8, ax.t + 16, 'blue: training windows', 10.5, BLUE, 'end', MONO, 700) + T(ax.l + ax.pw - 8, ax.t + 31, 'red: validation windows', 10.5, RED, 'end', MONO, 700)
seeds = TS['lstm']['mase_seeds']
mean_s = sum(seeds) / len(seeds); sd_s = (sum((v - mean_s) ** 2 for v in seeds) / len(seeds)) ** .5
add(slide('LSTM', 'LSTM &#183; Example', 'Example',
          f"{TS['lstm']['n_train']:,} training windows, {len(hist)} epochs, {TS['time']['lstm']:.0f} seconds on a CPU. MASE {MASE['lstm']:.3f}, against {MASE['arimax']:.3f} for ARIMA with the calendar.",
          '<div class="cols">' + fig(svg(540, 400, s, 'LSTM training and validation loss'), 'loss by epoch, seed 0') +
          '<div class="colstack">' +
          fig(fcfig(K, ['arimax', 'lstm'], w=560, h=240), f'window 2 &#183; {LEGEND}',
              f"MAE in this window: ARIMA + calendar {window_mae('arimax', K):.1f}, LSTM {window_mae('lstm', K):.1f}.") +
          '<div class="kv k3">' + tile('MASE, seed 0', f"{MASE['lstm']:.3f}", 'ok') + tile('mean of 5 seeds', f'{mean_s:.3f}') + tile('weights', f"{TS['lstm']['params']:,}") + '</div>'
          '</div></div>'))

# ================================================================ LSTM drawbacks
ax = Ax(560, 130, (0.50, 0.60), (0, 1), pad=(14, 14, 26, 30))
s = Rect(ax.l, ax.t, ax.pw, ax.ph, '#FFFFFF', HAIR, 1)
for v in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
    s += L(ax.X(v), ax.t, ax.X(v), ax.t + ax.ph, HAIR, 1) + T(ax.X(v), ax.t + ax.ph + 14, f'{v:.2f}', 10, SOFT, 'middle', MONO)
for v in seeds:
    s += ax.dot(v, 0.32, PURP, 7, 'fill-opacity=".8"')
s += T(ax.X(max(seeds)) + 12, ax.Y(0.32) + 4, 'LSTM', 11, PURP, 'start', MONO, 700)
for m in ('arimax', 'prophet'):
    s += L(ax.X(MASE[m]), ax.t, ax.X(MASE[m]), ax.t + ax.ph, COL[m], 2.4)
    s += T(ax.X(MASE[m]) + (-6 if m == 'arimax' else 6), 18, f'{NAME[m]} {MASE[m]:.3f}', 10.5, COL[m], 'end' if m == 'arimax' else 'start', MONO, 700)
add(slide('LSTM', 'LSTM &#183; Drawbacks', 'Drawbacks',
          f"Five runs that differ only in the random seed score between {min(seeds):.3f} and {max(seeds):.3f}. ARIMA with the calendar scores {MASE['arimax']:.3f} every time.",
          '<div class="cols"><div class="colstack">' +
          fig(svg(560, 130, s, 'LSTM seeds against the classical baselines'), 'MASE of the same network trained five times') +
          fig(kindbars(['arimax', 'lstm'], w=560, label_w=160), 'MAE by kind of day: ARIMA + calendar in slate, LSTM in purple') +
          '</div><div class="colstack fill">' +
          card('It matched the baseline', f"The LSTM learned the calendar effects from windows: its payday error is {BYK['lstm']['payday'][0]:.1f}, ARIMA + calendar {BYK['arimax']['payday'][0]:.1f}. On six series with additive effects that gave no clear gain.", 'p') +
          card('One number per day', 'The output is a point forecast. A treasury desk sizing a buffer needs to know how wrong the forecast can be, and this network does not say.', 'r') +
          card('No account of itself', f"Prophet shows its curves and ARIMA its five weights. The LSTM has {TS['lstm']['params']:,} weights and no part of it reads as the payday effect.", 'o') +
          '<div class="callout sm">Deep forecasters pull ahead with hundreds of related series, inputs that interact, and long histories. With six series and effects that add up, a classical model given the right columns is hard to beat.</div>'
          '</div></div>'))

# ================================================================ TFT concept
s = ''
s += G_(0, box(20, 20, 250, 54, 'static', 'segment', SLATE) + box(20, 96, 250, 54, 'observed in the past', 'net outflow, 56 days', BLUE) +
        box(20, 172, 250, 54, 'known for the future', 'calendar, 56 + 14 days', GREEN))
s += G_(1, box(340, 80, 170, 86, 'variable selection', 'a weight per input', AMBER) +
        arrow(270, 47, 340, 100) + arrow(270, 123, 340, 123) + arrow(270, 199, 340, 146))
s += G_(2, box(570, 80, 160, 86, 'LSTM', 'encoder and decoder', PURP) + arrow(510, 123, 570, 123))
s += G_(3, box(790, 80, 170, 86, 'attention', 'over the 56 past days', RED) + arrow(730, 123, 790, 123))
s += G_(4, box(1020, 20, 130, 54, 'P90', '', GREEN) + box(1020, 96, 130, 54, 'P50', '', GREEN) + box(1020, 172, 130, 54, 'P10', '', GREEN) +
        arrow(960, 105, 1020, 50) + arrow(960, 123, 1020, 123) + arrow(960, 141, 1020, 196) +
        T(1085, 246, 'for each of 14 days', 10.5, SOFT, 'middle', MONO))
add(slide('TFT', 'TFT &#183; Concept', 'Concept',
          'The Temporal Fusion Transformer keeps the LSTM and adds three parts around it: typed inputs with learned weights, attention over the past, and quantile outputs.',
          fig(svg(1170, 256, s, 'Structure of the Temporal Fusion Transformer'), 'Lim et al., 2021, simplified') +
          '<div class="cols4">' +
          '<div class="card sm o" data-at="1"><h3>Variable selection</h3><p>A small network gives each input a weight at every step. The weights can be read afterwards.</p></div>'
          '<div class="card sm p" data-at="2"><h3>LSTM</h3><p>The same encoder and decoder as the last section, for local patterns such as the weekly cycle.</p></div>'
          '<div class="card sm r" data-at="3"><h3>Attention</h3><p>Each forecast day looks back over all 56 encoder days and weights them. The weights show which past days it used.</p></div>'
          '<div class="card sm g" data-at="4"><h3>Quantiles</h3><p>Three outputs per day. P10 to P90 is a range the outflow should fall inside on 8 days in 10.</p></div>'
          '</div>' + bld(4), 'build'))

# ================================================================ quantile loss + widget
wed = [Y[i] for i in range(TEST0, n) if CAL['dow'][i] == 2 and not CAL['is_holiday'][i] and CAL['since_payday'][i] > 3 and not CAL['is_dd_day'][i]]
lo_w = math.floor(min(wed) / 10) * 10 - 10; hi_w = math.ceil(max(wed) / 10) * 10 + 10
lmax = math.ceil(0.95 * (hi_w - lo_w) / 2 / 10) * 10
pb = {'y': wed, 'lo': lo_w, 'hi': hi_w, 'lmax': lmax, 'xlab': 'candidate forecast, GBP million'}
add(slide('TFT', 'TFT &#183; Mechanism', 'Mechanism',
          f"A quantile forecast is trained with the pinball loss, which charges q per unit for forecasting too low and 1 - q for too high. Below: the outflow on {len(wed)} ordinary Wednesdays of the test period.",
          '<div class="cols w6040"><div class="fig"><div class="cap">dots: the Wednesdays &#183; curve: the loss of forecasting each value for all of them</div>'
          '<svg id="pb-svg" viewBox="0 0 700 400" width="100%" role="img" aria-label="Pinball loss"></svg></div>'
          '<div class="colstack">'
          f'<div class="wctl"><label class="rg">quantile q: <b id="pb-ql">0.50</b><input type="range" id="pb-q" min="0.05" max="0.95" step="0.05" value="0.5"/></label>{JSB}</div>'
          '<div class="out" id="pb-out"></div>' +
          out("""loss = q × (y - ŷ)        if y ≥ ŷ
       (1 - q) × (ŷ - y)  otherwise""", 'xs', 'pinball loss for one day') +
          '<div class="callout sm">At q = 0.5 the two costs are equal and the best forecast is the median. At q = 0.9 a forecast that is too low costs nine times as much, so the minimum moves up until 90% of the values lie below it.</div>'
          '</div></div>' + jdata('pb-data', pb)))

# ================================================================ TFT example
TF = TS['tft']
add(slide('TFT', 'TFT &#183; Example', 'Example',
          f"{TF['params']:,} weights, {TF['epochs']} epochs, {TS['time']['tft']:.0f} seconds on a CPU. The median forecast scores MASE {MASE['tft']:.3f}, and the P10 to P90 band holds the actual outflow on {pct(TF['cover'])} of test days.",
          '<div class="cols w4060"><div class="colstack">' +
          code("""training = TimeSeriesDataSet(
    long[long.time_idx < VAL0],
    time_idx='time_idx', target='net_outflow',
    group_ids=['segment'],
    max_encoder_length=56, max_prediction_length=14,
    static_categoricals=['segment'],
    time_varying_known_categoricals=['dow', 'since_payday'],
    time_varying_known_reals=[...calendar flags...],
    time_varying_unknown_reals=['net_outflow'],
    target_normalizer=GroupNormalizer(groups=['segment']))

tft = TemporalFusionTransformer.from_dataset(
    training, hidden_size=32, attention_head_size=4,
    loss=QuantileLoss([0.1, 0.5, 0.9]))""", 'xs') +
          f"<div class=\"callout warn sm\">A band built to hold 80% of days held {pct(TF['cover'])}. The outflow was above P90 on {pct(TF['above'])} of days and below P10 on {pct(TF['below'])}. A quantile forecast has to be backtested before a buffer is sized from it.</div>"
          '</div>' +
          fig(fcfig(K, ['tft'], w=660, h=420, band=True), f'window 2, median and P10 to P90 band &#183; {LEGEND}',
              f"MAE of the median in this window {window_mae('tft', K):.1f}.") +
          '</div>'))

# ================================================================ TFT: what it used
dec = sorted(TF['dec_imp'].items(), key=lambda t: -t[1])
enc = sorted(TF['enc_imp'].items(), key=lambda t: -t[1])
dfig = hbars([(k, v, GREEN, '') for k, v in dec], w=520, label_w=130, bar_h=19, gap=7, unit='%', label='Decoder variable weights', valfmt=lambda v: f'{v:.1f}%')
efig = hbars([(k, v, BLUE, '') for k, v in enc[:5]], w=520, label_w=130, bar_h=19, gap=7, unit='%', label='Encoder variable weights', valfmt=lambda v: f'{v:.1f}%')
at = TF['attn']
ax = Ax(560, 250, (-56.5, -0.5), (0, max(at) * 1.2), pad=(46, 12, 12, 32))
s = ax.frame(yticks=[round(v, 2) for v in (0.01, 0.02, 0.03, 0.04) if v < ax.yr[1]], xticks=[(-v, f'-{v}') for v in (56, 49, 42, 35, 28, 21, 14, 7, 1)], ylab='share of attention', xlab='days before the forecast origin')
for i, v in enumerate(at):
    d = i - 56
    s += ax.vbar(d - 0.38, d + 0.38, v, RED if (-d) % 7 == 0 else '#9FB0B9')
top3 = sorted(range(56), key=lambda i: -at[i])[:3]
add(slide('TFT', 'TFT &#183; Example', 'Example continued 1',
          f"What the model reports about itself. For the future days the largest selection weight is on {dec[0][0]} ({dec[0][1]:.1f}%); for the past, {enc[0][1]:.0f}% is on the outflow itself.",
          '<div class="cols"><div class="colstack">' +
          fig(dfig, 'selection weights, future days (decoder)') +
          fig(efig, 'selection weights, past days (encoder), top 5') +
          '</div><div class="colstack">' +
          fig(svg(560, 250, s, 'Attention over the encoder days'), 'attention over the 56 past days, red bars are multiples of 7',
              f"The three most attended days are {', '.join(str(56 - i) for i in top3)} days before the origin.") +
          '<div class="callout sm">These weights describe where the network looked. They are a starting point for a conversation with the treasury desk, not a causal statement about what drives outflow.</div>'
          '</div></div>'))

# ================================================================ widget: forecast explorer
fx = {'y': Y, 'dates': DATES, 'dow': CAL['dow'], 'hol': CAL['is_holiday'], 'pay': CAL['is_payday'], 'dd': CAL['is_dd_day'],
      'origins': ORIG, 'H': H, 'fc': TS['fc_head'], 'q': TS['q_head'], 'col': COL, 'name': NAME, 'mase': MASE}
btns = ''.join(f'<button data-v="{m}"{" class=\"on\"" if m == "arima" else ""}>{NAME[m]}</button>' for m in ('naive', 'arima', 'prophet', 'arimax', 'lstm', 'tft'))
add(slide('TFT', 'Forecasts &#183; Forecast explorer', 'Forecast explorer',
          'Retail current accounts. Choose a model and move the origin through the 13 test windows.',
          f'<div class="wctl"><div class="seg" id="fx-model">{btns}</div>'
          f'<label class="rg">forecast origin<input type="range" id="fx-origin" min="0" max="12" step="1" value="1"/></label>{JSB}</div>'
          '<div class="fig"><div class="cap">grey: the 28 days before the origin &#183; black: actual &#183; colour: forecast &#183; ' + LEGEND + '</div>'
          '<svg id="fx-svg" viewBox="0 0 1120 290" width="100%" role="img" aria-label="Forecast explorer"></svg></div>'
          '<div class="cols nat"><div class="out" id="fx-out"></div>'
          '<div class="callout">Windows to try: origin 2 holds a payday, origin 5 the August bank holiday, origin 13 Christmas.</div></div>' + jdata('fx-data', fx)))

# ================================================================ verdict: all models
order = sorted(NAME, key=lambda m: MASE[m])
items = [(NAME[m], MASE[m], COL[m], f"fit {TS['time'][m]:.0f} s" if m in TS['time'] else '') for m in ('naive', 'arima', 'prophet', 'arimax', 'lstm', 'tft')]
vfig = hbars(items, w=700, label_w=170, bar_h=44, gap=20, maxv=1.2, label='MASE of every model', valfmt=lambda v: f'{v:.3f}')
segrows = ''
for i_, sg in enumerate(SEGS):
    best = min(NAME, key=lambda m: TS['mase_seg'][m][i_])
    segrows += f'<tr><td>{sg}</td>' + ''.join(f'<td class="r"{" style=\"font-weight:700;background:#F3FBF7\"" if m == best else ""}>{TS["mase_seg"][m][i_]:.3f}</td>' for m in ('arima', 'prophet', 'arimax', 'lstm', 'tft')) + '</tr>'
spread = max(MASE[m] for m in ('prophet', 'arimax', 'lstm', 'tft')) - min(MASE[m] for m in ('prophet', 'arimax', 'lstm', 'tft'))
add(slide('TFT', 'Forecasts &#183; Verdict', 'Verdict',
          f"Four models that are given the calendar finish within {spread:.3f} of each other: {', '.join(f'{NAME[m]} {MASE[m]:.3f}' for m in order[:4])}.",
          '<div class="cols">' + fig(vfig, 'MASE over 13 windows and 6 segments, lower is better') +
          '<div class="colstack">' +
          f'<table class="d xs"><tr><th>MASE by segment</th><th class="r">ARIMA</th><th class="r">Prophet</th><th class="r">ARIMA+cal</th><th class="r">LSTM</th><th class="r">TFT</th></tr>{segrows}</table>'
          f"<div class=\"callout sm\">The step that mattered was from {MASE['arima']:.3f} to about {MASE[order[0]]:.2f}, and it came from the calendar. Which model consumed the calendar changed the score by {spread:.3f}.</div>"
          '<div class="callout good sm">What the deep models add here is in the output: one model for all segments, a quantile band, and weights that can be inspected. Start with the baseline, and move up when one of those is needed.</div>'
          '</div></div>'))

# ================================================================ applications
add(slide('Applications', 'Applications', 'Forecasting problems in a bank',
          'The models are the same. The target, the horizon and the inputs known in advance change with the problem.',
          '<div class="cols3">' +
          '<div class="card b big"><h3>Cash-flow forecasting</h3><p><b>Target:</b> daily net outflow by segment or legal entity.<br/><b>Horizon:</b> 1 to 30 days.<br/><b>Known in advance:</b> payday, direct-debit dates, bank holidays, tax deadlines, scheduled maturities.<br/><b>Feeds:</b> intraday funding and the treasury plan.</p></div>'
          '<div class="card g big"><h3>Demand forecasting</h3><p><b>Target:</b> cash drawn per ATM, calls per hour, logins per minute, mortgage applications per week.<br/><b>Horizon:</b> hours to weeks.<br/><b>Known in advance:</b> calendar, campaigns, rate announcements.<br/><b>Feeds:</b> cash replenishment, staffing, capacity.</p></div>'
          '<div class="card r big"><h3>Market forecasting</h3><p><b>Target:</b> volatility, volumes, spreads. Price direction has almost no forecastable signal at a daily horizon.<br/><b>Horizon:</b> minutes to days.<br/><b>Known in advance:</b> central bank dates, data releases, expiries.<br/><b>Feeds:</b> risk limits, margin, hedging.</p></div>'
          '</div>'
          '<div class="cols nat"><div class="callout"><b>Many related series favour one shared deep model.</b> 5,000 ATMs or 400 branches are where an LSTM or TFT trained across all of them beats 5,000 separate ARIMA fits, in accuracy and in the number of models to maintain.</div>'
          '<div class="callout"><b>The cost of an error is lopsided.</b> An empty ATM costs more than a full one, and a funding shortfall more than idle cash. That asymmetry is the q in the pinball loss, which is why these problems are forecast as quantiles.</div></div>'
          '<div class="callout warn sm">In every case the test is the one used today: hold back the most recent period, forecast it from an earlier origin, and compare against the seasonal naive forecast.</div>'))

# ================================================================ liquidity
LQ = TS['liq']
lo_l = math.floor(min(LQ['cum_act'] + LQ['cum_p50']) / 100) * 100; hi_l = math.ceil(max(LQ['cum_act'] + LQ['cum_p90']) / 100) * 100
ax = Ax(700, 430, (0.4, 13.6), (lo_l, hi_l), pad=(52, 12, 12, 32))
s = ax.frame(yticks=range(lo_l, hi_l + 1, 100), xticks=[(k, str(k)) for k in range(1, 14)], ylab='GBP million over 14 days', xlab='test window', yfmt=lambda v: f'{v:,}')
for k in range(13):
    x = k + 1; a_, p5, p9 = LQ['cum_act'][k], LQ['cum_p50'][k], LQ['cum_p90'][k]
    s += Rect(ax.X(x) - 13, ax.Y(p9), 26, ax.Y(p5) - ax.Y(p9), GREEN, extra='fill-opacity=".2"')
    s += L(ax.X(x) - 13, ax.Y(p9), ax.X(x) + 13, ax.Y(p9), GREEN, 2.6) + L(ax.X(x) - 13, ax.Y(p5), ax.X(x) + 13, ax.Y(p5), GREEN, 1.4)
    s += ax.dot(x, a_, RED if a_ > p9 else INK, 6)
s += T(ax.l + 8, ax.t + 30, 'thick green: sum of the daily P90 forecasts', 10.5, GREEN, 'start', MONO, 700)
s += T(ax.l + 8, ax.t + 44, 'thin green: sum of the daily medians', 10.5, GREEN, 'start', MONO)
s += T(ax.l + 8, ax.t + 58, 'dot: actual 14-day outflow, red when above the P90 line', 10.5, INK, 'start', MONO, 700)
add(slide('Applications', 'Applications &#183; Liquidity risk', 'Liquidity risk',
          f"A liquidity buffer is a quantile of future outflow. Sized from the TFT&#8217;s P90, the buffer was exceeded in {LQ['breach_p90']} of 13 two-week windows; sized from the median, in {LQ['breach_p50']}.",
          '<div class="cols w6040">' + fig(svg(700, 430, s, 'Cumulative 14-day outflow against two buffers'), 'retail current accounts, cumulative outflow per window') +
          '<div class="colstack fill"><div class="kv" style="grid-template-columns:1fr 1fr">' +
          tile('windows above the P90 buffer', f"{LQ['breach_p90']} of 13", 'bad' if LQ['breach_p90'] > 1 else 'ok') +
          tile('days above the daily P90', f"{LQ['day_breach_p90']} of {LQ['days']}", 'bad' if LQ['day_breach_p90'] > 0.15 * LQ['days'] else 'ok') + '</div>' +
          (f"<div class=\"callout warn sm\">A P90 should be exceeded on 10% of days. Here it was exceeded on {LQ['day_breach_p90'] / LQ['days'] * 100:.0f}% for this segment: the model was trained on 2022 to early 2025 and the book kept growing. Backtest the quantile on recent data, and recalibrate or retrain when the breach rate drifts.</div>"
           if LQ['day_breach_p90'] > 0.25 * LQ['days'] else
           f"<div class=\"callout sm\">A P90 should be exceeded on 10% of days. Here it was exceeded on {LQ['day_breach_p90'] / LQ['days'] * 100:.0f}% for this segment. That backtest is the evidence a treasury desk needs before sizing a buffer from the forecast, and it is repeated as new data arrives.</div>") +
          '<div class="callout sm">Adding 14 daily P90s gives more than the P90 of the 14-day total, because the days do not all run high together. To size a two-week buffer exactly, train the model on the cumulative outflow or sample whole paths.</div>'
          '</div></div>'))
