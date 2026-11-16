"""Simulated bank data for the day 1 session.

Three files, one simulated bank:
  data/cashflow.csv      daily net outflow, 6 segments of the deposit book, 2022-2025
  data/transactions.csv  card transactions over 180 days with a fraud label
  data/atm.csv           cash withdrawn per day at 200 ATMs, 2024-2025

Everything is simulated. Nothing here comes from a real bank.
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data') + '/'

# ------------------------------------------------------------------ calendar
# England and Wales bank holidays, 2022-2025
BANK_HOLIDAYS = pd.to_datetime([
    '2022-01-03', '2022-04-15', '2022-04-18', '2022-05-02', '2022-06-02', '2022-06-03',
    '2022-08-29', '2022-09-19', '2022-12-26', '2022-12-27',
    '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-08', '2023-05-29',
    '2023-08-28', '2023-12-25', '2023-12-26',
    '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-06', '2024-05-27', '2024-08-26',
    '2024-12-25', '2024-12-26',
    '2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05', '2025-05-26', '2025-08-25',
    '2025-12-25', '2025-12-26',
])


def calendar(start='2022-01-01', end='2025-12-31'):
    d = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
    d['dow'] = d.date.dt.dayofweek                      # 0 = Monday
    d['is_holiday'] = d.date.isin(BANK_HOLIDAYS).astype(int)
    working = (d.dow < 5) & (d.is_holiday == 0)
    d['is_working'] = working.astype(int)

    # payday: the 25th, moved back to the last working day when the 25th is not one
    d['is_payday'] = 0
    # direct-debit day: the 1st, moved forward to the next working day
    d['is_dd_day'] = 0
    idx = d.set_index('date').index
    for per in pd.period_range(start, end, freq='M'):
        p = pd.Timestamp(per.year, per.month, 25)
        while not working[idx.get_loc(p)]:
            p -= pd.Timedelta(days=1)
        d.loc[d.date == p, 'is_payday'] = 1
        q = pd.Timestamp(per.year, per.month, 1)
        while not working[idx.get_loc(q)]:
            q += pd.Timedelta(days=1)
        d.loc[d.date == q, 'is_dd_day'] = 1

    # days since the last payday, capped at 4 (0 on payday itself)
    since, k = [], 9
    for v in d.is_payday:
        k = 0 if v else k + 1
        since.append(min(k, 4))
    d['since_payday'] = since
    # first working day after a bank holiday
    prev_h = d.is_holiday.shift(1, fill_value=0)
    run = []
    pending = 0
    for h, w in zip(d.is_holiday, d.is_working):
        if h:
            pending = 1; run.append(0)
        elif w and pending:
            run.append(1); pending = 0
        else:
            run.append(0)
    d['post_holiday'] = run
    doy = d.date.dt.dayofyear
    d['yr_sin'] = np.sin(2 * np.pi * doy / 365.25)
    d['yr_cos'] = np.cos(2 * np.pi * doy / 365.25)
    d['december'] = ((d.date.dt.month == 12) & (d.date.dt.day <= 24)).astype(int) * d.date.dt.day / 24
    return d


# ------------------------------------------------------------------ cash flow
# net outflow in GBP million; positive = money leaving the bank that day
SEGMENTS = {
    #                 level  trend  weekday profile (Mon..Sun)            payday  dd   hol  yearly dec  noise
    'retail_current': (120, .012, [18, 6, 0, 4, 22, -38, -52],           -62,   55,  -50,  9,    16,   8.0),
    'retail_savings': (35,  .004, [6, 2, 0, 1, 5, -20, -24],             -30,   8,   -22,  5,    -6,   5.0),
    'sme':            (80,  .010, [14, 8, 4, 6, 16, -46, -58],           28,    34,  -60,  6,    8,    7.5),
    'corporate':      (210, .020, [10, 4, 0, 6, 30, -120, -150],         45,    60,  -150, 14,   20,   17.0),
    'cards':          (60,  .015, [-6, -4, 0, 4, 16, 22, 6],             10,    0,   12,   4,    26,   5.5),
    'mortgages':      (-45, -.006, [3, 1, 0, 1, 2, 12, 14],              -4,    -70, 14,   2,    0,    3.0),
}


def simulate_cashflow(seed=7):
    rng = np.random.default_rng(seed)
    cal = calendar()
    n = len(cal)
    t = np.arange(n)
    frames = []
    for seg, (level, trend, wk, pay, dd, hol, yr, dec, sd) in SEGMENTS.items():
        growth = 1 + 0.00018 * t                      # effects grow with the book
        y = level + trend * t
        wk_eff = np.array(wk)[cal.dow.values].astype(float)
        # a bank holiday behaves like a Sunday; the weekday effect is switched off
        wk_eff = np.where(cal.is_holiday.values == 1, hol, wk_eff)
        y = y + growth * wk_eff
        y = y + growth * yr * cal.yr_sin.values + growth * dec * cal.december.values
        # payday: salary lands (inflow for retail, outflow for the employers),
        # then three days of spending that decay
        sp = cal.since_payday.values
        y = y + growth * pay * (sp == 0)
        spend = np.select([sp == 1, sp == 2, sp == 3], [0.40, 0.25, 0.12], 0.0)
        # the spending is larger when payday was a Friday
        fri_pay = (cal.is_payday.values == 1) & (cal.dow.values == 4)
        boost = np.zeros(n)
        for i in np.where(fri_pay)[0]:
            boost[i + 1:i + 4] = 0.45
        y = y - growth * pay * spend * (1 + boost)
        y = y + growth * dd * cal.is_dd_day.values
        # payments held over a bank holiday settle on the next working day
        y = y - growth * 0.55 * hol * cal.post_holiday.values
        # AR(1) noise, wider in the days after payday
        e = np.zeros(n)
        z = rng.standard_normal(n)
        scale = sd * (1 + 0.7 * (sp <= 3))
        for i in range(1, n):
            e[i] = 0.55 * e[i - 1] + scale[i] * z[i]
        y = y + e
        f = cal.copy()
        f.insert(1, 'segment', seg)
        f.insert(2, 'net_outflow', np.round(y, 2))
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ transactions
CATS = ['grocery', 'fuel', 'restaurant', 'online_retail', 'travel', 'electronics', 'cash_withdrawal']
CAT_P = [0.30, 0.10, 0.17, 0.22, 0.05, 0.06, 0.10]
CAT_MU = [3.2, 3.7, 3.0, 3.5, 5.2, 5.0, 3.9]      # log GBP
CAT_SD = [0.7, 0.4, 0.6, 0.9, 0.9, 0.8, 0.6]
CHANNELS = ['contactless', 'chip_pin', 'online', 'atm']


def simulate_transactions(n=100_000, days=180, seed=11):
    rng = np.random.default_rng(seed)
    cat = rng.choice(len(CATS), n, p=CAT_P)
    day = np.sort(rng.integers(0, days, n))
    # hour: lunch and evening peaks for most, flat-ish for online
    comp = rng.choice(3, n, p=[0.45, 0.40, 0.15])
    hour = np.where(comp == 0, rng.normal(12.5, 2.0, n),
                    np.where(comp == 1, rng.normal(18.5, 2.2, n), rng.uniform(0, 24, n)))
    hour = np.clip(np.round(hour), 0, 23).astype(int)
    amount = np.exp(rng.normal(np.array(CAT_MU)[cat], np.array(CAT_SD)[cat]))
    # channel depends on category and amount
    ch = np.empty(n, dtype=int)
    for i in range(n):
        c = cat[i]
        if c == 3:
            ch[i] = 2
        elif c == 6:
            ch[i] = 3
        elif c == 4:
            ch[i] = rng.choice([1, 2], p=[0.35, 0.65])
        else:
            ch[i] = 0 if (amount[i] <= 100 and rng.random() < 0.8) else 1
    age = np.clip(np.round(rng.gamma(9, 4.6, n) + 18), 18, 90).astype(int)
    tenure = np.clip(np.round(rng.exponential(70, n) * (0.4 + age / 80)), 1, 480).astype(int)
    dist = np.exp(rng.normal(1.2, 1.1, n))
    dist = np.where(cat == 4, dist * 25, dist)
    dist = np.where(ch == 2, 0.0, dist)                 # online: no card-present distance
    vel = rng.poisson(1.6, n)

    # fraud: a logistic score with interactions, calibrated to roughly 0.9%
    night = ((hour <= 5) | (hour >= 23)).astype(float)
    z = (-6.3
         + 1.5 * (ch == 2) + 0.9 * (cat == 5) + 0.5 * (cat == 6)
         + 1.5 * night
         + 0.75 * np.clip(np.log(amount) - 4.0, 0, None)
         + 1.3 * (amount < 3)                           # card-testing
         + 0.55 * np.clip(vel - 3, 0, None)
         + 0.8 * (np.log1p(dist) > 4.0)
         + 0.7 * (tenure < 6)
         + 1.2 * night * (ch == 2)
         + 0.6 * (cat == 5) * (ch == 2))
    p = 1 / (1 + np.exp(-z))
    fraud = (rng.random(n) < p).astype(int)
    # fraudsters also change behaviour: faster, further, rounder amounts
    vel = np.where(fraud == 1, vel + rng.poisson(2.2, n), vel)
    amount = np.where((fraud == 1) & (rng.random(n) < 0.35), np.round(amount, -1) + 0.0, amount)
    amount = np.clip(np.round(amount, 2), 0.5, None)

    return pd.DataFrame({
        'day': day, 'hour': hour, 'amount': amount,
        'category': np.array(CATS)[cat], 'channel': np.array(CHANNELS)[ch],
        'customer_age': age, 'tenure_months': tenure,
        'distance_km': np.round(dist, 1), 'txns_last_24h': vel, 'is_fraud': fraud,
    })


# ------------------------------------------------------------------ ATM cash
# cash withdrawn per ATM per day, GBP thousand. Three kinds of site with different weeks.
ATM_KINDS = {
    #               share  weekday profile (Mon..Sun), mean 1        payday  holiday  december
    'high_street': (.40, [0.92, 0.95, 1.00, 1.05, 1.30, 1.38, 0.40],  0.35,   0.45,    0.30),
    'residential': (.35, [0.95, 0.90, 0.92, 0.98, 1.25, 1.15, 0.85],  0.70,   0.90,    0.18),
    'transport':   (.25, [1.22, 1.25, 1.24, 1.22, 1.32, 0.45, 0.30],  0.15,   0.30,    0.08),
}


def simulate_atm(n_atm=200, n_new=20, start='2024-01-01', end='2025-12-31', test_days=91, seed=23):
    """n_new of the ATMs open 90 days before the test period, so they have a short history."""
    rng = np.random.default_rng(seed)
    cal = calendar(start, end)
    n = len(cal); t = np.arange(n)
    kinds = rng.choice(list(ATM_KINDS), size=n_atm, p=[v[0] for v in ATM_KINDS.values()])
    new = set(rng.choice(n_atm, size=n_new, replace=False).tolist())
    sp = cal.since_payday.values
    frames = []
    for a in range(n_atm):
        _, wk, pay, hol, dec = ATM_KINDS[kinds[a]]
        level = float(np.exp(rng.normal(np.log(14), 0.5)))            # GBP thousand a day
        prof = np.array(wk) * np.exp(rng.normal(0, 0.08, 7)); prof = prof / prof.mean()
        pay_a = pay * float(np.exp(rng.normal(0, 0.25)))
        y = level * (1 - 0.00022 * t)                                 # cash use falls about 8% a year
        y = y * np.where(cal.is_holiday.values == 1, hol, prof[cal.dow.values])
        y = y * (1 + pay_a * np.select([sp == 0, sp == 1, sp == 2, sp == 3], [1.0, 0.6, 0.35, 0.15], 0.0))
        y = y * (1 + dec * cal.december.values)
        e = np.zeros(n); z = rng.standard_normal(n)                   # a slow local level, then day noise
        for i in range(1, n):
            e[i] = 0.9 * e[i - 1] + 0.05 * z[i]
        y = y * np.exp(e) * rng.gamma(shape=22, scale=1 / 22, size=n)
        f = pd.DataFrame({'date': cal.date, 'atm_id': f'ATM{a:03d}', 'kind': kinds[a],
                          'withdrawn': np.round(y, 2)})
        for c in ['dow', 'is_holiday', 'is_payday', 'since_payday', 'december']:
            f[c] = cal[c].values
        if a in new:
            f = f.iloc[n - test_days - 90:]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    cf = simulate_cashflow()
    cf.to_csv(OUT + 'cashflow.csv', index=False)
    tx = simulate_transactions()
    tx.to_csv(OUT + 'transactions.csv', index=False)
    simulate_atm().to_csv(OUT + 'atm.csv', index=False)
    print(cf.shape, tx.shape, 'fraud rate', round(tx.is_fraud.mean() * 100, 3), '%', tx.is_fraud.sum())
    print(cf.groupby('segment').net_outflow.agg(['mean', 'std', 'min', 'max']).round(1))
    print(tx.groupby('channel').is_fraud.agg(['mean', 'sum', 'size']).round(4))
