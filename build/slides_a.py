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
  <div class="kicker">Machine Learning for Finance</div>
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
    <tr><td>Code</td><td>forecasting_and_generative_models.ipynb</td></tr>
  </table>
</section>
''')

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
          '<div class="cols nat"><div class="callout"><b>Many related series favour one shared model.</b> With 5,000 ATMs or 400 branches, one model trained across all of them replaces 5,000 separate fits. The ATM slides measure this on 200 ATMs.</div>'
          '<div class="callout"><b>The cost of an error is lopsided.</b> An empty ATM costs more than a full one, and a funding shortfall more than idle cash. That asymmetry is the q in the pinball loss, which is why these problems are forecast as quantiles.</div></div>'
          '<div class="callout warn sm">In every case the test is the one used today: hold back the most recent period, forecast it from an earlier origin, and compare against the seasonal naive forecast.</div>'))


# ================================================================ liquidity: concept (staged)
AP = TS['app']
KC = 8                                  # window 9: a payday and a direct-debit day inside it
oc = ORIG[KC]; qc = TS['q_head'][KC]; ac = Y[oc:oc + H]
shortc = [max(0, a_ - q_[2]) for a_, q_ in zip(ac, qc)]; idlec = [max(0, q_[2] - a_) for a_, q_ in zip(ac, qc)]
n_short = sum(1 for v in shortc if v > 0); n_med = sum(1 for a_, q_ in zip(ac, qc) if a_ > q_[1])
ax = Ax(700, 450, (oc - 0.7, oc + H - 0.3), (0, 260), pad=(46, 12, 14, 50))
s = ax.frame(yticks=range(0, 261, 50), ylab='GBP million, one day')
bars = ''; med = ''; p90 = ''; res = ''
for h in range(H):
    i = oc + h; x0, x1 = ax.X(i - 0.36), ax.X(i + 0.36)
    bars += Rect(x0, ax.Y(ac[h]), x1 - x0, ax.Y(0) - ax.Y(ac[h]), '#C9D3D9')
    bars += T(ax.X(i), ax.t + ax.ph + 14, DATES[i][8:], 10, SOFT, 'middle', MONO)
    bars += T(ax.X(i), ax.t + ax.ph + 26, DOW[CAL['dow'][i]], 9, SOFT, 'middle', MONO)
    med += L(x0 - 3, ax.Y(qc[h][1]), x1 + 3, ax.Y(qc[h][1]), GREEN, 1.6, 'stroke-dasharray="4 3"')
    p90 += L(x0 - 3, ax.Y(qc[h][2]), x1 + 3, ax.Y(qc[h][2]), GREEN, 3.2)
    if shortc[h] > 0:
        res += Rect(x0, ax.Y(ac[h]), x1 - x0, ax.Y(qc[h][2]) - ax.Y(ac[h]), RED)
        res += T(ax.X(i), ax.Y(ac[h]) - 5, f'{shortc[h]:.0f}', 10.5, RED, 'middle', MONO, 700)
    else:
        res += Rect(x0, ax.Y(qc[h][2]), x1 - x0, ax.Y(ac[h]) - ax.Y(qc[h][2]), GREEN, extra='fill-opacity=".16"')
s += G_(0, bars + tags(ax, oc, oc + H, ax.t + ax.ph + 32) + T(ax.l + 8, ax.t + 30, 'grey bar: outflow that happened', 10.5, INK, 'start', MONO, 700))
s += G_(1, med + T(ax.l + 8, ax.t + 44, 'dashed green: median forecast (P50)', 10.5, GREEN, 'start', MONO))
s += G_(2, p90 + T(ax.l + 8, ax.t + 58, 'thick green: P90 forecast, the cash held', 10.5, GREEN, 'start', MONO, 700))
s += G_(3, res + T(ax.l + 8, ax.t + 72, 'red: short that day · pale green: cash held and not used', 10.5, RED, 'start', MONO, 700))
add(slide('Applications', 'Applications &#183; Liquidity risk', 'Concept',
          'A treasury desk holds cash today against tomorrow&#8217;s outflow. The forecast it acts on is a high quantile: an amount that should be enough on 9 days in 10.',
          '<div class="cols w6040">' +
          fig(svg(700, 450, s, 'One two-week window: outflow, median forecast and P90 buffer'),
              f'retail current accounts, forecast made on {dm(oc - 1)} for {dm(oc)} to {dm(oc + H - 1)} &#183; ' + LEGEND) +
          '<div class="colstack fill">'
          f'<div class="kv" style="grid-template-columns:1fr 1fr" data-at="1">' + tile('days above the median', f'{n_med} of 14', 'b') + tile('mean outflow per day', f'{sum(ac) / H:.0f}', '') + '</div>'
          f'<div class="kv" style="grid-template-columns:1fr 1fr" data-at="3">' + tile('days short of the P90', f'{n_short} of 14', 'bad') + tile('short, total', f'{sum(shortc):.1f}', 'bad') + '</div>'
          f'<div class="kv" style="grid-template-columns:1fr 1fr" data-at="3">' + tile('cash held, not used', f'{sum(idlec):.1f}', 'ok') + tile('per day', f'{sum(idlec) / H:.1f}', 'ok') + '</div>'
          '<div class="callout sm" data-at="2"><b>The decision is the green line.</b> A point forecast answers what will happen. The desk asks how much to hold, and that is a quantile of the same forecast.</div>'
          '<div class="callout warn sm" data-at="3"><b>Two costs pull against each other.</b> Cash held and not used earns less than it would elsewhere. A short day is covered by borrowing at a worse rate. A higher line trades one for the other.</div>'
          '</div></div>' + bld(3), 'build'))
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
add(slide('Applications', 'Applications &#183; Liquidity risk', 'Example',
          f"Sized from the TFT&#8217;s P90, the two-week buffer was exceeded in {LQ['breach_p90']} of 13 windows; sized from the median, in {LQ['breach_p50']}.",
          '<div class="cols w6040">' + fig(svg(700, 430, s, 'Cumulative 14-day outflow against two buffers'), 'retail current accounts, cumulative outflow per window') +
          '<div class="colstack fill"><div class="kv" style="grid-template-columns:1fr 1fr">' +
          tile('windows above the P90 buffer', f"{LQ['breach_p90']} of 13", 'bad' if LQ['breach_p90'] > 1 else 'ok') +
          tile('windows above the median buffer', f"{LQ['breach_p50']} of 13", 'bad') + '</div>' +
          '<div class="callout sm">A buffer sized from the median forecast is too small about half the time by construction. The desk holds a high quantile, and the question for the next slide is whether the P90 behaves like one.</div>'
          '<div class="callout sm">Adding 14 daily P90s gives more than the P90 of the 14-day total, because the days do not all run high together. To size a two-week buffer exactly, train the model on the cumulative outflow or sample whole paths.</div>'
          '</div></div>'))

# ================================================================ liquidity: test (backtest of the P90)
BT = AP['bt']; ND = AP['n_days']
SEGLAB = {'retail_current': 'retail current', 'retail_savings': 'retail savings', 'sme': 'SME', 'corporate': 'corporate',
          'cards': 'cards', 'mortgages': 'mortgages'}
cw, ch, gx, gy = 30, 22, 92, 26
s = ''
for h in range(H):
    s += T(gx + h * cw + cw / 2, gy - 8, str(h + 1), 9.5, SOFT, 'middle', MONO)
s += T(gx + H * cw / 2, 9, 'day ahead', 9.5, SOFT, 'middle', MONO) + T(gx - 8, gy - 8, 'forecast made', 9.5, SOFT, 'end', MONO)
tot_up = 0
for k in range(13):
    o = ORIG[k]; up = 0
    s += T(gx - 8, gy + k * ch + 15, dm(o - 1), 10, INK, 'end', MONO)
    for h in range(H):
        a_, q_ = Y[o + h], TS['q_head'][k][h]
        c = RED if a_ > q_[2] else BLUE if a_ < q_[0] else '#E4EAED'
        up += a_ > q_[2]
        s += Rect(gx + h * cw + 1, gy + k * ch + 1, cw - 2, ch - 2, c)
    tot_up += up
    s += T(gx + H * cw + 10, gy + k * ch + 15, f'{up} above', 10.5, RED if up > 2 else SOFT, 'start', MONO, 700 if up > 2 else 400)
gh = gy + 13 * ch + 8
assert tot_up == BT[J][1]
grid = svg(gx + H * cw + 80, gh, s, 'Each of 182 test days against the P10 to P90 band')
bw, lw = 250, 118
s = ''
for i_, r_ in enumerate(BT):
    y = 30 + i_ * 40; v = r_[2]; sig = r_[5] < 0.05
    col = (RED if v > 10 else AMBER) if sig else '#8FA0A9'
    s += T(lw - 8, y + 16, SEGLAB[r_[0]], 11.5, INK, 'end', MONO, 500)
    s += Rect(lw, y, bw * v / 20, 24, col)
    s += T(lw + bw * v / 20 + 6, y + 16, f'{v:.1f}%', 11.5, INK, 'start', MONO, 700)
    s += T(lw + bw + 66, y + 16, f'{r_[1]} days &#183; p {r_[5]:.4f}'.replace('&#183;', '·'), 10, RED if sig else SOFT, 'start', MONO)
s += L(lw + bw / 2, 18, lw + bw / 2, 30 + 6 * 40 - 8, INK, 1.4, 'stroke-dasharray="4 3"') + T(lw + bw / 2, 12, '10% expected', 9.5, INK, 'middle', MONO, 700)
segbars = svg(lw + bw + 190, 30 + 6 * 40, s, 'Share of days above the P90 by segment')
n_fail = sum(1 for r_ in BT if r_[5] < 0.05)
add(slide('Applications', 'Applications &#183; Liquidity risk', 'Test',
          f"Backtest the quantile. A P90 should be exceeded on 10% of days; for retail current accounts it was exceeded on {BT[J][1]} of {ND}, which is {BT[J][2]:.1f}%.",
          '<div class="cols">' +
          fig(grid, 'retail current accounts &#183; <span style="color:#B02F3B;font-weight:700">red</span> above P90 &#183; <span style="color:#125F9E;font-weight:700">blue</span> below P10 &#183; grey inside the band',
              f'<b>{BT[J][1]} red and {BT[J][3]} blue of {ND}.</b> The band was meant to hold 80% of days and held {(ND - BT[J][1] - BT[J][3]) / ND * 100:.0f}%, so it is too narrow on both sides.') +
          '<div class="colstack" data-at="1">' +
          fig(segbars, 'days above the P90, all six segments, with a binomial test against 10%') +
          f'<div class="callout warn sm" data-at="2"><b>{n_fail} of 6 segments fail the test</b>, three with a P90 that is too low and SME with one that is too high. The test counts days as independent. Red days here arrive together, 0 to 6 in a window, so the p-values overstate the evidence.</div>'
          '<div class="callout good sm" data-at="2">A desk runs this count every week on the newest days. When the rate drifts from 10% it widens the margin from recent errors, which is the next slide, or retrains.</div>'
          '</div></div>' + bld(2), 'build'))

# ================================================================ liquidity: cost (widget)
MG = {r_[0]: r_ for r_ in AP['margin']}
mk = {'naive': 'Seasonal naive', 'arima': 'ARIMA', 'prophet': 'Prophet', 'arimax': 'ARIMA + calendar', 'lstm': 'LSTM', 'tft': 'TFT (median)'}
cost = {'err': AP['err'], 'col': COL, 'name': NAME, 'mean': AP['mean_out']}
cbt = ''.join(f'<button data-v="{m}"{" class=\"on\"" if m == "naive" else ""}>{NAME[m]}</button>' for m in ('naive', 'arima', 'prophet', 'arimax', 'lstm', 'tft'))
rbt = ''.join(f'<button data-v="{r_}"{" class=\"on\"" if r_ == 9 else ""}>{r_}x</button>' for r_ in (1, 4, 9, 19))
add(slide('Applications', 'Applications &#183; Liquidity risk', 'Cost',
          f"Hold the forecast plus a fixed margin. To be short on 9 of {ND} days the margin is {MG[mk['naive']][1]:.1f} with the seasonal naive forecast, {MG[mk['arima']][1]:.1f} with ARIMA and {MG[mk['tft']][1]:.1f} with the TFT.",
          f'<div class="wctl"><div class="seg" id="cs-model">{cbt}</div>'
          '<label class="rg"><span>margin above the forecast: <b id="cs-ml">58.0</b> GBP m</span><input type="range" id="cs-margin" min="0" max="90" step="0.5" value="58"/></label>'
          f'<div class="seg" id="cs-ratio" title="cost of a pound short, in pounds idle">{rbt}</div>{JSB}</div>'
          '<div class="fig"><div class="cap">left: actual minus forecast, 182 test days, retail current accounts &#183; red: short that day &#160;|&#160; right: cost per day against the margin</div>'
          '<svg id="cs-svg" viewBox="0 0 1120 280" width="100%" role="img" aria-label="Margin against shortfall and idle cash"></svg></div>'
          '<div class="cols nat"><div class="out" id="cs-out"></div>'
          f'<div class="colstack"><div class="callout sm"><b>Accuracy is paid back as cash.</b> For the same 9 short days the TFT desk holds {MG[mk["naive"]][3] - MG[mk["tft"]][3]:.0f} million a day less idle cash than the seasonal naive desk, on one segment with a mean outflow of {AP["mean_out"]:.0f} million a day.</div>'
          '<div class="callout sm"><b>The multiple picks the quantile.</b> If a pound short costs 9 times a pound idle, the cheapest margin is the 90th percentile of the errors: 9 / (9 + 1). That multiple is the q in the pinball loss.</div></div></div>' + jdata('cs-data', cost)))

# ================================================================ whole bank (staged)
BK = AP['bank']
SHORT = ['current', 'savings', 'SME', 'corporate', 'cards', 'mortg.']
SEGCOL = [BLUE, SLATE, PURP, AMBER, GREEN, '#7C8C95']
def stack(y, vals, scale, h=52, names=True):
    x = 150; o = ''
    for v, nm, c in zip(vals, SHORT, SEGCOL):
        w_ = abs(v) * scale
        o += Rect(x, y, w_ - 1.5, h, c)
        if names and w_ > 34:
            o += T(x + w_ / 2, y + h / 2 + 4, nm, 10.5, '#FFFFFF', 'middle', MONO, 700)
        x += w_
    return o, x
sc1 = 500 / BK['sum_seg_mae']; sc2 = 500 / BK['sum_margin']
b1, x1 = stack(40, BK['seg_mae'], sc1)
b3, x3 = stack(290, BK['seg_margin'], sc2)
s = T(10, 22, 'MEAN ABSOLUTE ERROR, GBP MILLION A DAY', 11, SOFT, 'start', MONO, 700)
s += G_(0, T(142, 64, 'six segments', 12, INK, 'end', MONO) + T(142, 79, 'added', 12, INK, 'end', MONO) + b1 + T(x1 + 8, 73, f"{BK['sum_seg_mae']:.1f}", 18, INK, 'start', MONO, 700))
s += G_(1, T(142, 142, 'bank total', 12, INK, 'end', MONO) + Rect(150, 110, BK['mae_total'] * sc1, 52, INK) +
        T(150 + BK['mae_total'] * sc1 + 8, 143, f"{BK['mae_total']:.1f}", 18, RED, 'start', MONO, 700) +
        T(150, 184, 'errors in different segments partly cancel', 11.5, SOFT, 'start', MONO))
s += G_(2, T(10, 272, 'CASH HELD ABOVE THE MEDIAN FORECAST, GBP MILLION A DAY', 11, SOFT, 'start', MONO, 700) +
        T(142, 314, 'six P90', 12, INK, 'end', MONO) + T(142, 329, 'margins added', 12, INK, 'end', MONO) + b3 +
        T(x3 + 8, 323, f"{BK['sum_margin']:.1f}", 18, INK, 'start', MONO, 700))
s += G_(3, T(142, 384, 'needed for 10%', 12, INK, 'end', MONO) + T(142, 399, 'at bank level', 12, INK, 'end', MONO) +
        Rect(150, 360, BK['need_margin'] * sc2, 52, GREEN) +
        T(150 + BK['need_margin'] * sc2 + 8, 393, f"{BK['need_margin']:.1f}", 18, GREEN, 'start', MONO, 700) +
        T(150, 434, f"exceeded on {BK['above_need']} of {ND} days", 11.5, SOFT, 'start', MONO))
add(slide('Applications', 'Applications &#183; The whole bank', 'Example',
          f"The desk funds the bank, which is the six segments added. The TFT&#8217;s six errors add to {BK['sum_seg_mae']:.1f} million a day, and its error on the total is {BK['mae_total']:.1f}.",
          '<div class="cols w6040">' + fig(svg(740, 450, s, 'Segment errors and margins against the bank total'), f"TFT, 182 test days &#183; mean net outflow of the bank {BK['mean_total']:.0f} million a day") +
          '<div class="colstack fill">'
          '<div class="kv" style="grid-template-columns:1fr 1fr" data-at="2">' + tile('days above the six P90s added', f"{BK['above_sum_p90']} of {ND}", 'b') + tile('that is', f"{BK['above_sum_p90'] / ND * 100:.1f}%", 'b') + '</div>'
          f"<div class=\"callout sm\" data-at=\"2\"><b>Six P90s added are far above a P90.</b> The segments do not all run high on the same day, so the sum was exceeded on {BK['above_sum_p90'] / ND * 100:.1f}% of days and the bank holds more than the 10% target asks for.</div>"
          f"<div class=\"callout good sm\" data-at=\"3\"><b>{BK['sum_margin'] - BK['need_margin']:.0f} million a day is released</b> by sizing the margin on the total: {BK['need_margin']:.1f} against {BK['sum_margin']:.1f}. Forecast the quantity the decision is made on, or sample the six segments together.</div>"
          '<div class="callout sm" data-at="3">The same arithmetic applied across days on the earlier slide: 14 daily P90s added were never exceeded in 13 windows.</div>'
          '</div></div>' + bld(3), 'build'))

# ================================================================ horizon
WK = {r_[0]: r_ for r_ in AP['week']}
ax = Ax(700, 450, (-0.6, 5.6), (0, 1.2), pad=(46, 12, 14, 44))
s = ax.frame(yticks=[0, .2, .4, .6, .8, 1.0, 1.2], ylab='MASE over 6 segments', yfmt=lambda v: f'{v:.1f}')
for i_, m in enumerate(('naive', 'arima', 'arimax', 'prophet', 'lstm', 'tft')):
    w1, w2 = WK[mk[m]][1], WK[mk[m]][2]
    s += ax.vbar(i_ - 0.36, i_ - 0.02, w1, COL[m], 0, .55) + ax.vbar(i_ + 0.02, i_ + 0.36, w2, COL[m], 0)
    s += T(ax.X(i_ - 0.19), ax.Y(w1) - 5, f'{w1:.3f}', 10, SOFT, 'middle', MONO)
    s += T(ax.X(i_ + 0.19), ax.Y(w2) - 5, f'{w2:.3f}', 10, INK, 'middle', MONO, 700)
    for q_, part in enumerate(NAME[m].replace(' + ', ' +|').split('|') if '+' in NAME[m] else NAME[m].split(' ', 1)):
        s += T(ax.X(i_), ax.t + ax.ph + 15 + 12 * q_, part, 10, INK, 'middle', MONO)
s += T(ax.l + ax.pw - 8, ax.t + 16, 'pale: days 1 to 7 · solid: days 8 to 14', 10.5, INK, 'end', MONO, 700)
hold = [m for m in ('prophet', 'lstm', 'tft') if WK[mk[m]][2] <= WK[mk[m]][1] + 0.002]
add(slide('Applications', 'Applications &#183; Horizon', 'Example',
          f"The second week against the first. The TFT scores {WK[mk['tft']][1]:.3f} on days 1 to 7 and {WK[mk['tft']][2]:.3f} on days 8 to 14; ARIMA goes from {WK[mk['arima']][1]:.3f} to {WK[mk['arima']][2]:.3f}.",
          '<div class="cols w6040">' + fig(svg(700, 450, s, 'MASE in the first and second forecast week by model'), '13 windows, 6 segments; each week holds every weekday once') +
          '<div class="colstack fill">'
          f"<div class=\"callout\"><b>{len(hold)} models lose nothing in week 2:</b> {', '.join(NAME[m] for m in hold)}. Paydays, direct-debit days and bank holidays for day 14 are known on day 0, and most of this series is calendar.</div>"
          f"<div class=\"callout\"><b>Both ARIMA models lose about {((WK[mk['arima']][2] - WK[mk['arima']][1]) + (WK[mk['arimax']][2] - WK[mk['arimax']][1])) / 2:.2f}.</b> Part of their forecast comes from the last few days&#8217; errors, and that information is used up within a week.</div>"
          '<div class="callout warn sm"><b>Which horizon the desk needs</b> depends on the decision: tomorrow for intraday funding, two weeks for the funding plan, 30 days for a liquidity ratio. Only 14 days was tested here. A 30-day claim needs its own backtest.</div>'
          '</div></div>'))


# ================================================================ ATM cash: concept
AT = AP['atm']
KINDLAB = {'high_street': 'high street', 'residential': 'residential', 'transport': 'transport hub'}
KCOL = {'high_street': BLUE, 'residential': GREEN, 'transport': PURP}
s = ''
for i_, kd in enumerate(('high_street', 'residential', 'transport')):
    e = AT['ex'][kd]; y0 = 8 + i_ * 146; nd = len(e['y']); top = math.ceil(max(e['y']) / 10) * 10
    px = lambda d: 52 + (d + 0.5) / nd * 636
    py = lambda v: y0 + 16 + (top - v) / top * 96
    s += Rect(52, y0 + 16, 636, 96, '#FFFFFF', HAIR, 1)
    s += T(58, y0 + 29, f"{KINDLAB[kd]} · {e['id']} · {AT['kinds'][kd]} ATMs of this kind", 10.5, KCOL[kd], 'start', MONO, 700)
    s += T(46, y0 + 20, str(top), 9.5, SOFT, 'end', MONO) + T(46, y0 + 115, '0', 9.5, SOFT, 'end', MONO)
    for d in range(nd):
        wkend = e['dow'][d] >= 5
        s += Rect(px(d) - 6, py(e['y'][d]), 12, y0 + 112 - py(e['y'][d]), KCOL[kd], extra=f'fill-opacity="{.4 if wkend else .85}"')
        tag = 'H' if e['hol'][d] else 'P' if e['pay'][d] else ''
        if tag:
            s += Rect(px(d) - 6, y0 + 115, 12, 12, RED if tag == 'H' else GREEN) + T(px(d), y0 + 124.5, tag, 8.5, '#FFFFFF', 'middle', MONO, 700)
        elif e['dow'][d] == 0:
            s += T(px(d), y0 + 124.5, e['dates'][d][8:] + '/' + e['dates'][d][5:7], 8.5, SOFT, 'middle', MONO)
add(slide('Applications', 'Applications &#183; ATM cash', 'Concept',
          f"A second problem with the same shape. {AT['n']} ATMs, cash withdrawn per day for {AT['days']} days, and a van that loads each one for the days until its next visit.",
          '<div class="cols w6040">' +
          fig(svg(700, 446, s, 'Six weeks of daily withdrawals at three ATMs'), 'GBP thousand withdrawn per day, the 6 weeks before the test &#183; pale bars are weekends &#183; <span class="tag g">P</span>payday <span class="tag r">H</span>bank holiday') +
          '<div class="colstack fill"><div class="kv" style="grid-template-columns:1fr 1fr">' +
          tile('ATMs', str(AT['n']), 'b') + tile('opened 90 days before the test', str(AT['n_new']), 'bad') +
          tile('mean withdrawn per day', f"{AT['mean_day']:.1f}", '') + tile('test days', str(AT['test']), '') + '</div>'
          '<div class="callout sm"><b>The decision is the load.</b> Too little and the ATM is empty before the van returns. Too much and cash sits in a box earning nothing. A third cost is new here: each visit is paid for.</div>'
          '<div class="callout sm"><b>The week differs by site.</b> A transport hub is busy Monday to Friday, a high street on Friday and Saturday, and a residential ATM after payday.</div>'
          f"<div class=\"callout warn sm\"><b>{AT['n_new']} ATMs are new.</b> Each has {AT['new_train_days']} days to train on, so a model fitted to one of them alone learns its payday and its week from those {AT['new_train_days']} days.</div>"
          '</div></div>'))

# ================================================================ ATM cash: one shared model (staged)
AM = {r_[0]: r_ for r_ in AT['mase']}
order = ['Seasonal naive', 'One model per ATM', 'One shared model']; mcol = ['#7C8C95', AMBER, GREEN]
def mgroup(y, idx, title):
    o = T(10, y, title, 11, SOFT, 'start', MONO, 700)
    for i_, (nm, c) in enumerate(zip(order, mcol)):
        v = AM[nm][idx]; yy = y + 14 + i_ * 46
        o += T(178, yy + 23, nm, 12, INK, 'end', MONO, 500) + Rect(186, yy, 420 * v / 1.1, 36, c)
        o += T(186 + 420 * v / 1.1 + 8, yy + 24, f'{v:.3f}', 16, INK, 'start', MONO, 700)
    return o
s = G_(0, mgroup(22, 1, f"MASE, {AT['n'] - AT['n_new']} ESTABLISHED ATMS")) + G_(1, mgroup(232, 2, f"MASE, {AT['n_new']} NEW ATMS, {AT['new_train_days']} TRAINING DAYS EACH"))
add(slide('Applications', 'Applications &#183; ATM cash', 'Example',
          f"One model trained on all {AT['n']} ATMs scores MASE {AM['One shared model'][3]:.3f}. {AT['n']} models, one per ATM, score {AM['One model per ATM'][3]:.3f}, and the seasonal naive forecast {AM['Seasonal naive'][3]:.3f}.",
          '<div class="cols w6040">' + fig(svg(700, 420, s, 'MASE of three forecasts for established and new ATMs'), 'gradient-boosted trees, median forecast, 7 days ahead, 13 weekly origins') +
          '<div class="colstack fill">'
          '<div class="callout sm"><b>Same model class, same inputs.</b> Weekday, payday, bank holiday, December, days ahead and the ATM&#8217;s mean over the last 28 days. The shared model also reads the kind of site and the ATM id.</div>'
          f"<div class=\"callout sm\"><b>Pooling helps the established ATMs too:</b> {AM['One shared model'][1]:.3f} against {AM['One model per ATM'][1]:.3f}. A payday at one residential ATM teaches the model about the other {AT['kinds']['residential'] - 1}.</div>"
          f"<div class=\"callout good sm\" data-at=\"1\"><b>New ATMs: {AM['One shared model'][2]:.3f} against {AM['One model per ATM'][2]:.3f}.</b> The shared model forecasts an ATM with {AT['new_train_days']} days of history almost as well as the per-ATM models forecast one with two years ({AM['One model per ATM'][1]:.3f}).</div>"
          f"<div class=\"callout sm\" data-at=\"1\"><b>One model to maintain.</b> The shared fit took {AT['t_shared']} seconds for five quantiles; the {AT['n']} separate fits took {AT['t_local']} seconds for the median alone.</div>"
          '</div></div>' + bld(1), 'build'))

# ================================================================ ATM cash: test
PA = AT['per_atm']; mxd = max(PA); exp_d = 0.05 * AT['test']
cw_ = 31; x0_ = 44; base = 372
s = ''
for d in range(mxd + 1):
    k = sum(1 for v in PA if v == d)
    for r_ in range(k):
        s += Rect(x0_ + d * cw_ + 2, base - (r_ + 1) * 9, cw_ - 4, 7.5, RED if d >= 10 else '#8FA0A9')
    s += T(x0_ + d * cw_ + cw_ / 2, base + 14, str(d), 10, INK if d % 5 == 0 or d >= 10 else SOFT, 'middle', MONO)
    if k:
        s += T(x0_ + d * cw_ + cw_ / 2, base - k * 9 - 5, str(k), 10, RED if d >= 10 else SOFT, 'middle', MONO, 700)
s += L(x0_, base, x0_ + (mxd + 1) * cw_, base, SOFT, 1)
xe = x0_ + exp_d * cw_ + cw_ / 2
s += L(xe, 20, xe, base, INK, 1.4, 'stroke-dasharray="4 3"') + T(xe + 6, 28, f'expected {exp_d:.2f} days, which is 5% of {AT["test"]}', 10.5, INK, 'start', MONO, 700)
s += T(x0_ + (mxd + 1) * cw_, base + 30, 'days above the P95, out of 91', 10, SOFT, 'end', MONO)
n_bad = sum(1 for v in PA if v >= 10)
qrows = ''.join(f"<tr><td>P{q}</td><td class=\"r\">{100 - int(q)}%</td><td class=\"r\">{AT['above'][q]:.1f}%</td></tr>" for q in ('50', '80', '90', '95', '99'))
add(slide('Applications', 'Applications &#183; ATM cash', 'Test',
          f"Backtest the P95 one ATM at a time. Over all {AT['n'] * AT['test']:,} ATM-days it was exceeded on {AT['above']['95']:.1f}%, close to the 5% it promises. {n_bad} ATMs were above it on 10 days or more.",
          '<div class="cols w6040">' + fig(svg(x0_ + (mxd + 1) * cw_ + 16, base + 36, s, 'ATMs by number of days above the P95'), 'one square is one ATM &#183; red: 10 or more days above') +
          '<div class="colstack fill">'
          f'<table class="d sm"><tr><th>forecast</th><th class="r">should be exceeded</th><th class="r">was exceeded</th></tr>{qrows}</table>'
          f"<div class=\"callout warn sm\"><b>The average passes and {n_bad} ATMs fail.</b> With a true 5%, ten or more days above in 91 has a probability under 2% for one ATM. These {n_bad} run out more than their share however the fleet average looks.</div>"
          '<div class="callout good sm">The backtest is run at the level the cash is held, which here is the single ATM. The red ATMs get a higher quantile or a look at what changed at the site.</div>'
          '</div></div>'))

# ================================================================ ATM cash: cost (widget)
GR = AT['grid']
atw = {'grid': GR, 'qs': ['50', '80', '90', '95', '99'], 'keep': AT['keep'], 'mean': AT['mean_day']}
qb = ''.join(f'<button data-v="{q}"{" class=\"on\"" if q == "95" else ""}>P{q}</button>' for q in atw['qs'])
rb = ''.join(f'<button data-v="{r_}"{" class=\"on\"" if r_ == 3 else ""}>every {r_} {"day" if r_ == 1 else "days"}</button>' for r_ in (1, 2, 3, 7))
add(slide('Applications', 'Applications &#183; ATM cash', 'Cost',
          f"Two choices: the quantile to load to and the days between visits. The P95 every 3 days leaves {GR['95_3']['out']:.2f}% of ATM-days empty and {GR['95_3']['idle']:.1f} thousand in each ATM at day end.",
          f'<div class="wctl"><div class="seg" id="at-q">{qb}</div><div class="seg" id="at-r">{rb}</div>{JSB}</div>'
          f"<div class=\"fig\"><div class=\"cap\">left: cash in one residential ATM over the 91 test days under the chosen policy &#160;|&#160; right: all 20 policies for the {AT['n']} ATMs, the chosen one circled</div>"
          '<svg id="at-svg" viewBox="0 0 1120 280" width="100%" role="img" aria-label="ATM replenishment policies"></svg></div>'
          '<div class="cols nat"><div class="out" id="at-out"></div>'
          '<div class="colstack">'
          f"<div class=\"callout sm\"><b>Fewer visits cost cash.</b> At the P95, a daily visit leaves {GR['95_1']['idle']:.1f} thousand in each ATM at day end and a weekly visit {GR['95_7']['idle']:.1f}. The van is saved 6 trips a week per ATM.</div>"
          f"<div class=\"callout sm\"><b>A longer cycle is safer than its quantile says.</b> Loading to the P95 daily leaves {GR['95_1']['out']:.1f}% of ATM-days empty, and weekly {GR['95_7']['out']:.2f}%, because 7 daily P95s added are far above the P95 of a week.</div></div></div>" + jdata('at-data', atw)))

# ================================================================ call volumes: concept
CC = AP['calls']; CST = {r_[0]: r_ for r_ in CC['staff']}; CV = dict(CC['curve'])
wk_ = CC['week']['calls']; top = math.ceil(max(wk_) / 50) * 50
px = lambda i: 52 + (i + 0.5) / 168 * 636
py = lambda v: 24 + (top - v) / top * 170
s = Rect(52, 24, 636, 170, '#FFFFFF', HAIR, 1)
for v in range(0, top + 1, 100):
    s += L(52, py(v), 688, py(v), HAIR, 1) + T(46, py(v) + 3.5, str(v), 9.5, SOFT, 'end', MONO)
for d in range(7):
    if d:
        s += L(52 + d * 636 / 7, 24, 52 + d * 636 / 7, 194, LINE, 1)
    s += T(52 + (d + 0.5) * 636 / 7, 208, DOW[CC['week']['dow'][d]] + ' ' + CC['week']['dates'][d][8:] + '/' + CC['week']['dates'][d][5:7], 10, INK, 'middle', MONO)
for i_, v in enumerate(wk_):
    s += Rect(px(i_) - 1.5, py(v), 3, 194 - py(v), BLUE)
s += T(58, 37, 'calls per half hour, 08:00 to 20:00, the week before the test', 10.5, INK, 'start', MONO, 700)
# service level against agents, 100 calls in a half hour
s += T(52, 250, 'SERVICE LEVEL AGAINST AGENTS, 100 CALLS IN A HALF HOUR', 10.5, SOFT, 'start', MONO, 700)
ns = sorted(CV); bw_ = 636 / len(ns)
for i_, n_ in enumerate(ns):
    v = CV[n_]; x = 52 + i_ * bw_; hh = v / 100 * 130
    s += Rect(x + 8, 400 - hh, bw_ - 16, hh, GREEN if v >= 80 else RED)
    s += T(x + bw_ / 2, 400 - hh - 5, f'{v:.0f}%', 11, INK, 'middle', MONO, 700) + T(x + bw_ / 2, 415, f'{n_}', 11, INK, 'middle', MONO)
s += L(52, 400 - 104, 688, 400 - 104, INK, 1.2, 'stroke-dasharray="4 3"') + T(56, 400 - 109, 'target: 80% answered in 20 s', 10, INK, 'start', MONO, 700)
s += T(688, 430, 'agents on the phones', 10, SOFT, 'end', MONO)
n_ok = min(n_ for n_ in ns if CV[n_] >= 80)
add(slide('Applications', 'Applications &#183; Call volumes', 'Concept',
          f"A third problem. One contact centre, calls counted every half hour, and a roster written a week ahead that needs a number of agents for each half hour.",
          '<div class="cols w6040">' + fig(svg(700, 436, s, 'A week of half-hourly calls and the service level curve'), 'simulated contact centre &#183; mean handling time 5 minutes') +
          '<div class="colstack fill"><div class="kv" style="grid-template-columns:1fr 1fr">' +
          tile('half hours in the test', f"{CC['n_test']:,}", 'b') + tile('mean calls per half hour', f"{CC['mean']:.0f}", '') + '</div>'
          '<div class="callout sm"><b>Two cycles at once.</b> Monday is the busy day and each weekday has a morning and an afternoon peak. The forecast is needed per half hour because agents are rostered per half hour.</div>'
          f"<div class=\"callout warn sm\"><b>The cost of being short is steep.</b> With 100 calls in a half hour, {n_ok} agents answer {CV[n_ok]:.0f}% within 20 seconds. Two fewer answer {CV[n_ok - 2]:.0f}%, and {n_ok - 4} answer {CV[n_ok - 4]:.0f}%.</div>"
          '<div class="callout sm"><b>The formula is Erlang C.</b> It takes calls per half hour and handling time and returns the agents needed. The forecast is its input, so a forecast error becomes a queue.</div>'
          '</div></div>'))

# ================================================================ call volumes: test (heatmap)
UP = CC['up']; cw_, ch_ = 7, 13.5; hx, hy = 50, 16
s = ''
for d in range(len(UP)):
    for i_ in range(24):
        s += Rect(hx + d * cw_, hy + i_ * ch_, cw_ - 0.8, ch_ - 0.8, RED if UP[d][i_] else '#E4EAED')
    if d % 7 == 0:
        s += T(hx + d * cw_ + 24, hy + 24 * ch_ + 13, f'week {d // 7 + 1}', 9, SOFT, 'middle', MONO)
for i_ in range(0, 24, 4):
    s += T(hx - 6, hy + i_ * ch_ + 10, f'{8 + i_ // 2:02d}:00', 9.5, SOFT, 'end', MONO)
s += T(hx + 91 * cw_, hy + 24 * ch_ + 28, 'one column is one day, one cell is one half hour', 10, SOFT, 'end', MONO)
RW = CC['raw']
add(slide('Applications', 'Applications &#183; Call volumes', 'Test',
          f"The P90 was exceeded in {CC['up_n']} of {CC['n_test']:,} half hours, {CC['up_pct']:.1f}%. Half of those misses fall on {CC['top_days']} of the 91 days.",
          '<div class="cols w6040">' + fig(svg(hx + 91 * cw_ + 8, hy + 24 * ch_ + 34, s, 'Half hours above the P90 forecast, by day'), '<span style="color:#B02F3B;font-weight:700">red</span>: calls above the P90 forecast for that half hour',
              f"<b>The red comes in columns.</b> {CC['day0']} days have no red half hour and {CC['day8']} days have 8 or more.") +
          '<div class="colstack fill"><div class="kv" style="grid-template-columns:1fr 1fr">' +
          tile('MASE, seasonal naive', f"{CC['mase']['naive']:.3f}", 'bad') + tile('MASE, model', f"{CC['mase']['model']:.3f}", 'ok') + '</div>'
          f"<div class=\"callout sm\"><b>A busy day is busy all day.</b> The day&#8217;s total ran between {CC['ratio'][0]:.2f} and {CC['ratio'][1]:.2f} times its forecast on 9 days in 10, and every half hour of that day moved with it.</div>"
          '<div class="callout warn sm"><b>Half hours are not independent trials.</b> A count of red cells understates how bad a red day is for the people on the phones: the queue from 10:00 is still there at 10:30.</div>'
          f"<div class=\"callout sm\"><b>A first version failed this test.</b> Trees fed raw call counts were exceeded on {RW['above90']:.1f}% of half hours. The level in the test reached {RW['lev_test']:.0f} calls and training had seen {RW['lev_train']:.0f}, and a tree cannot forecast above what it has seen. The model here forecasts calls divided by the recent level.</div>"
          '</div></div>'))

# ================================================================ call volumes: intraday update (staged)
U = CC['upd']; ua, uf, un = U['act'], U['fc'], U['new']; ud = U['day']
ax = Ax(700, 440, (-0.6, 23.6), (0, math.ceil(max(ua + un + uf) / 50) * 50 + 50), pad=(46, 12, 14, 34))
s = ax.frame(yticks=range(0, int(ax.yr[1]) + 1, 50), xticks=[(i_ - 0.5, f'{8 + i_ // 2}:00') for i_ in range(0, 24, 2)], ylab='calls per half hour')
am_b = ''.join(ax.vbar(i_ - 0.4, i_ + 0.4, ua[i_], '#8FA0A9', 0) for i_ in range(8))
pm_b = ''.join(ax.vbar(i_ - 0.4, i_ + 0.4, ua[i_], '#C9D3D9', 0) for i_ in range(8, 24))
step = lambda v, i0: ' '.join(f'{ax.X(i0 + k - 0.5):.1f},{ax.Y(x):.1f} {ax.X(i0 + k + 0.5):.1f},{ax.Y(x):.1f}' for k, x in enumerate(v))
wline = f'<polyline points="{step(uf, 0)}" fill="none" stroke="{BLUE}" stroke-width="2.4"/>'
nline = f'<polyline points="{step(un, 8)}" fill="none" stroke="{GREEN}" stroke-width="3"/>'
noon = L(ax.X(7.5), ax.t, ax.X(7.5), ax.t + ax.ph, INK, 1.4, 'stroke-dasharray="4 3"')
s += G_(0, am_b + wline + noon + T(ax.X(7.5) + 6, ax.t + 14, '12:00', 10.5, INK, 'start', MONO, 700) +
        T(ax.l + ax.pw - 8, ax.t + 14, 'blue: forecast issued earlier in the week', 10.5, BLUE, 'end', MONO, 700) +
        T(ax.l + ax.pw - 8, ax.t + 28, 'dark bars: calls received by 12:00', 10.5, INK, 'end', MONO, 700))
s += G_(1, nline + T(ax.l + ax.pw - 8, ax.t + 42, f"green: afternoon forecast x {U['r']:.2f}", 10.5, GREEN, 'end', MONO, 700))
s += G_(2, pm_b + T(ax.l + ax.pw - 8, ax.t + 56, 'pale bars: what the afternoon did', 10.5, SOFT, 'end', MONO, 700))
add(slide('Applications', 'Applications &#183; Call volumes', 'Example',
          f"The morning is a measurement of the day. Scaling the afternoon forecast at 12:00 by the morning&#8217;s ratio takes the afternoon error from {U['before']:.1f} to {U['after']:.1f} calls per half hour.",
          '<div class="cols w6040">' + fig(svg(700, 440, s, 'Intraday update on one busy day'), f"{DOW[CC['dow'][ud]]} {CC['dates'][ud]}, the test day with the largest morning ratio") +
          '<div class="colstack fill">'
          f"<div class=\"kv\" style=\"grid-template-columns:1fr 1fr\" data-at=\"1\">" + tile('morning, actual / forecast', f"{U['r']:.2f}", 'bad') + tile('days the update helped', f"{U['helped']} of 91", 'b') + '</div>'
          f"<div class=\"kv\" style=\"grid-template-columns:1fr 1fr\" data-at=\"2\">" + tile('afternoon MAE, weekly forecast', f"{U['before']:.1f}", 'bad') + tile('after the 12:00 update', f"{U['after']:.1f}", 'ok') + '</div>'
          '<div class="callout sm" data-at="1"><b>The update is one multiplication:</b> calls received by 12:00 divided by calls forecast by 12:00, applied to every half hour left.</div>'
          f"<div class=\"callout warn sm\" data-at=\"2\"><b>It made {91 - U['helped']} days worse.</b> On an ordinary day the morning ratio carries more noise than signal, so a production version shrinks it toward 1.</div>"
          '<div class="callout good sm" data-at="2">The roster is fixed, so the update buys the afternoon&#8217;s levers: moved breaks, overtime, call-backs offered.</div>'
          '</div></div>' + bld(2), 'build'))

# ================================================================ call volumes: cost (widget)
ccw = {'act': CC['act'], 'dates': CC['dates'], 'dow': CC['dow'], 'q': CC['q'], 'staff': CC['staff'], 'known': CC['known']}
qb = ''.join(f'<button data-v="{q}"{" class=\"on\"" if q == "80" else ""}>P{q}</button>' for q in ('50', '80', '90', '95'))
add(slide('Applications', 'Applications &#183; Call volumes', 'Cost',
          f"Roster to the median forecast and {CST['P50'][2]:.0f}% of half hours meet the target. Roster to the P90 and {CST['P90'][2]:.0f}% do, for {CST['P90'][1] - CST['P50'][1]:.0f} more agent-hours a day.",
          f'<div class="wctl"><div class="seg" id="cc-q">{qb}</div>'
          f'<label class="rg"><span>test day: <b id="cc-dl"></b></span><input type="range" id="cc-day" min="0" max="90" step="1" value="{CC['dow'].index(0)}"/></label>{JSB}</div>'
          '<div class="fig"><div class="cap">left: one test day, calls received against the forecast the roster was written for &#160;|&#160; right: the four rosters over all 91 days</div>'
          '<svg id="cc-svg" viewBox="0 0 1120 290" width="100%" role="img" aria-label="Staffing to a forecast quantile"></svg></div>'
          '<div class="cols nat"><div class="out" id="cc-out"></div>'
          '<div class="colstack">'
          f"<div class=\"callout sm\"><b>The price of not knowing.</b> With the calls known in advance the target needs {CC['known']:.0f} agent-hours a day. The P90 roster uses {CST['P90'][1]:.0f} and still misses in {100 - CST['P90'][2]:.0f}% of half hours.</div>"
          f"<div class=\"callout sm\"><b>The median is the wrong forecast to roster to.</b> Being 10% short costs far more service than being 10% over costs in wages, so the roster follows a high quantile. On {CST['P50'][4]} of 91 days the median roster had 6 or more half hours below target.</div></div></div>" + jdata('cc-data', ccw)))
# ================================================================ drawbacks
pay_no, pay_cal = BYK['arima']['payday'][0], BYK['tft']['payday'][0]
rows = [
    ('The band is the wrong width', f"P90 exceeded on {BT[J][2]:.1f}% of days for retail current and {BT[SEGS.index('sme')][2]:.1f}% for SME, against 10%",
     'Backtest each segment on the newest days.'),
    ('Quantiles do not add', f"Six P90s added: exceeded on {BK['above_sum_p90'] / ND * 100:.1f}% of days. 14 daily P90s added: exceeded in {LQ['breach_p90']} of 13 windows",
     'Forecast the total and the period the buffer is held against.'),
    ('The calendar is an input', f"Payday error {pay_no:.1f} without the calendar and {pay_cal:.1f} with it",
     'A moved payday or a new bank holiday has to reach the model before the day does.'),
    ('Trees do not extrapolate', f"Call model on raw counts: P90 exceeded in {AP['calls']['raw']['above90']:.1f}% of half hours. On calls divided by the recent level: {AP['calls']['up_pct']:.1f}%",
     'Scale the target by a recent level when the series trends.'),
    ('The history holds ordinary days', f"{TS['n']:,} days, none of them a run on deposits",
     'The forecast sizes the day-to-day buffer. A stress buffer comes from regulatory scenarios.'),
]
add(slide('Applications', 'Applications &#183; Drawbacks', 'Drawbacks',
          'Where a forecast misleads the people who act on it, with the number measured in this session.',
          table(['What goes wrong', 'Measured here', 'What the desk does'],
                [(f'<b>{a_}</b>', b_, c_) for a_, b_, c_ in rows], 'd free lg') +
          '<div class="cols nat"><div class="callout"><b>The desk is scored on the breach rate and the cash held.</b> MASE ranks the models; the backtest and the margin decide how much money sits idle.</div>'
          '<div class="callout good"><b>Four steps served all three problems:</b> forecast a quantile, backtest it, price the two errors, forecast at the level of the decision.</div></div>'))
