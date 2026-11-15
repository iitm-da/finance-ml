"""Second half of the deck: GANs for synthetic data, then the fraud detection pipeline."""
import random
from fx import *

B = []
add = B.append
JSB = '<span class="jsbadge">browser JS</span>'
N, RATE, GAN, FID, PRIV, FR = GN['n'], GN['rate'], GN['gan'], GN['fid'], GN['priv'], GN['fraud']
NUMC = ['hour', 'amount', 'customer_age', 'tenure_months', 'distance_km', 'txns_last_24h']
COLS = NUMC + ['category', 'channel', 'is_fraud']


def rowtable(rows, cols=COLS, cls='d sm tight', hl=None):
    def cell(c, v):
        if c == 'amount':
            return f'{v:.2f}'
        if c in ('distance_km',):
            return f'{v:.1f}'
        if c == 'score':
            return f'{v:.3f}'
        return str(v)
    h = ''.join(f'<th>{c}</th>' for c in cols)
    b = ''
    for r in rows:
        k = ' class="hl"' if hl and r.get(hl) == 1 else ''
        b += f'<tr{k}>' + ''.join(f'<td class="m">{cell(c, r[c])}</td>' for c in cols) + '</tr>'
    return f'<table class="{cls}"><tr>{h}</tr>{b}</table>'


# ================================================================ GAN · concept
s = ''
s += G_(0, box(30, 34, 160, 58, 'noise', f"{GAN['d_z']} random numbers", SLATE) +
        box(30, 214, 160, 58, 'fraud flag', '0 or 1', AMBER) +
        box(280, 114, 180, 78, 'Generator', '2 hidden layers of 256', RED) +
        box(560, 34, 200, 58, 'synthetic row', f"{GAN['d_x']} encoded values", RED) +
        arrow(190, 63, 280, 132) + arrow(190, 243, 280, 176) + arrow(460, 140, 560, 70))
s += G_(1, box(560, 214, 200, 58, 'real row', 'from the training table', GREEN) +
        box(850, 114, 170, 78, 'Critic', '2 hidden layers of 256', BLUE) +
        arrow(760, 63, 850, 136) + arrow(760, 243, 850, 172) +
        arrow(1020, 153, 1070, 153) + T(1076, 150, 'score', 13, INK, 'start', BODY, 700) +
        T(1076, 166, 'one number', 10, SOFT, 'start', MONO))
s += G_(2, T(935, 100, 'critic step', 11, BLUE, 'middle', MONO, 700) +
        T(935, 216, 'raise the score of real rows,', 10.5, BLUE, 'middle', MONO) +
        T(935, 231, 'lower it for generated rows', 10.5, BLUE, 'middle', MONO))
s += G_(3, f'<path d="M1095,174 C1095,318 370,318 370,200" fill="none" stroke="{RED}" stroke-width="1.8" stroke-dasharray="5 4"/>' +
        f'<path d="M370,192 L365,203 L375,203 z" fill="{RED}"/>' +
        T(690, 306, 'generator step: change the weights so its own rows score higher', 11, RED, 'middle', MONO, 700))
gfig = svg(1140, 320, s, 'Generator and critic of a conditional GAN')
add(slide('GAN', 'GAN &#183; Concept', 'Concept',
          'A generator turns random numbers into a table row. A critic scores rows, and each network is trained against the other.',
          fig(gfig, 'conditional GAN on one transaction row',
              'The generator does not read real rows. What reaches it is the gradient of the critic&#8217;s score.') +
          f'<div class="callout sm">Use case at a bank: a vendor, a test environment or another team needs transaction data, and the real table cannot leave. A generator trained on the real table writes rows with the same statistics and no real customer in them. The three tests later in this section measure how far that holds.</div>'
          + bld(3), 'build'))

# ================================================================ GAN · the data
s = ''
tot = N['tx']
x0, wid = 10, 1080
wa = wid * N['fit'] / tot; wb = wid * N['hold'] / tot; wc = wid * N['test'] / tot
s += Rect(x0, 26, wa, 40, RED) + Rect(x0 + wa, 26, wb, 40, GREY) + Rect(x0 + wa + wb, 26, wc, 40, INK)
s += T(x0 + 8, 51, f"GAN training rows  {comma(N['fit'])}", 12.5, '#FFFFFF', 'start', MONO, 700)
s += T(x0 + wa + 8, 51, f"held out  {comma(N['hold'])}", 12.5, INK, 'start', MONO, 700)
s += T(x0 + wa + wb + 8, 51, f"test  {comma(N['test'])}", 12.5, '#FFFFFF', 'start', MONO, 700)
s += T(x0, 16, 'days 0 to 119', 10.5, SOFT, 'start', MONO) + T(x0 + wa + wb, 16, 'days 120 to 179', 10.5, SOFT, 'start', MONO)
s += T(x0, 84, f"fraud rows {N['fraud_fit']}", 10.5, SOFT, 'start', MONO) + T(x0 + wa, 84, f"fraud rows {N['fraud_hold']}", 10.5, SOFT, 'start', MONO)
s += T(x0 + wa + wb, 84, f"fraud rows {N['fraud_test']}", 10.5, SOFT, 'start', MONO)
split = svg(1100, 92, s, 'How the transaction table is split')
add(slide('GAN', 'GAN &#183; The transaction table', 'The transaction table',
          f"{comma(N['tx'])} simulated card transactions over 180 days. {N['fraud_all']} are fraud, {pct(RATE['all'], 2)} of the table.",
          '<div class="outlbl">the first 8 rows</div>' + rowtable(GN['rows_real']) +
          fig(split, 'rows by use') +
          '<div class="callout sm">The held-out rows are real rows the GAN does not see. They give every test a reference: the score two samples of real data produce against each other.</div>'))

# ================================================================ GAN · mechanism
ax = Ax(520, 300, (1, GAN['epochs']), (0, max(GAN['wdist']) * 1.05), pad=(44, 12, 12, 28))
s = ax.frame(yticks=[0, 1, 2, 3], xticks=[(1, '1'), (50, '50'), (100, '100'), (150, '150')], ylab='critic(real) - critic(generated)', xlab='epoch')
s += ax.line(range(1, GAN['epochs'] + 1), GAN['wdist'], BLUE, 2.2)
wfig = svg(520, 300, s, 'Critic distance estimate per epoch')
add(slide('GAN', 'GAN &#183; Mechanism', 'Mechanism',
          f"The training loop, abridged from the notebook. {GAN['epochs']} epochs of {GAN['steps']} batches took {GAN['time']:.0f} seconds on a laptop CPU.",
          '<div class="cols w6040"><div class="colstack">' +
          code("""for step in range(STEPS):
    x, c = real_batch()                        # 512 rows, 20% of them fraud
    fake = G(noise(), c).detach()
    gap = C(x, c).mean() - C(fake, c).mean()   # critic's distance estimate
    lossC = -gap + 10 * gradient_penalty(C, x, fake, c)
    optC.zero_grad(); lossC.backward(); optC.step()

    if step % 5 == 0:                          # 1 generator step per 5 critic steps
        lossG = -C(G(noise(), c), c).mean()
        optG.zero_grad(); lossG.backward(); optG.step()""", 'sm') +
          '<ol class="n"><li><b>Wasserstein loss.</b> The critic outputs an unbounded score, and the gap between its mean score on real and generated rows estimates the distance between the two tables.</li>'
          '<li><b>Gradient penalty.</b> It keeps the critic&#8217;s slope near 1, which is what makes that gap a distance.</li>'
          f"<li><b>Oversampled fraud.</b> Fraud is {pct(RATE['fit'], 2)} of the table, about 5 rows in a batch of 512. Each batch is drawn with 20% fraud so the generator learns that class.</li></ol>"
          '</div><div class="colstack">' +
          fig(wfig, 'the gap the critic measures', f"The gap falls from {max(GAN['wdist']):.1f} to {GAN['wdist'][-1]:.2f} as generated rows become harder to separate.") +
          f"<div class=\"callout sm\">Generator {comma(GAN['params_g'])} weights, critic {comma(GAN['params_c'])}. Numeric columns pass through a quantile transform and the two categorical columns are one-hot.</div>"
          '</div></div>'))

# ================================================================ GAN · widget
floor = sum(FID['hold'][c] for c in NUMC) / len(NUMC)
random.seed(4)
gs = {'snaps': [{'epoch': sn['epoch'], 'ks': {k: round(v, 3) for k, v in sn['ks'].items()},
                 'amount': [round(a, 2) for a in sn['amount']], 'hour': sn['hour']} for sn in GN['snaps']],
      'real': GN['real_pts'], 'jit': [round(random.uniform(-5.5, 5.5), 1) for _ in range(53)],
      'epochs': GAN['epochs'], 'wdist': GAN['wdist'], 'floor': round(floor, 3)}
last = GN['snaps'][-1]
mean_ks = lambda sn: sum(sn['ks'].values()) / len(sn['ks'])
add(slide('GAN', 'GAN &#183; Training, snapshot by snapshot', 'Training, snapshot by snapshot',
          f"600 generated rows against 600 real ones at {len(GN['snaps'])} points in training. The mean KS statistic falls from {mean_ks(GN['snaps'][0]):.2f} to {mean_ks(last):.2f}.",
          '<div class="cols w6040"><div class="fig"><div class="cap">amount against hour of day</div>'
          '<svg id="gs-svg" viewBox="0 0 640 420" width="100%" role="img" aria-label="Real and synthetic rows"></svg></div>'
          '<div class="colstack"><div class="wctl"><button class="wbtn" id="gs-play">play</button>'
          f'<label class="rg">snapshot<input type="range" id="gs-ep" min="0" max="{len(GN["snaps"]) - 1}" step="1" value="0"/></label>{JSB}</div>'
          '<div class="out" id="gs-out"></div>'
          '<div class="fig"><div class="cap">critic&#8217;s distance estimate</div>'
          '<svg id="gs-w" viewBox="0 0 340 120" width="100%" role="img" aria-label="Critic distance"></svg></div>'
          '<div class="callout sm">Real rows concentrate around midday and early evening. The generator matches the amount range first and the hours later.</div>'
          '</div></div>' + jdata('gs-data', gs)))

# ================================================================ GAN · synthetic rows
add(slide('GAN', 'GAN &#183; Example', 'Example',
          f"The first 8 of {comma(PRIV['n_synth'])} generated rows. {PRIV['exact']} of them are identical to a training row.",
          '<div class="outlbl">synthetic</div>' + rowtable(GN['rows_synth']) +
          '<div class="cols nat">' +
          code("""cond  = (torch.rand(n, 1) < fraud_rate).float()
synth = decode(G(torch.randn(n, 64), cond), cond)""", 'sm') +
          f"<div class=\"callout sm\">The fraud flag is an input, so the fraud rate of the synthetic table is chosen at generation time: {pct(RATE['synth'], 2)} here against {pct(RATE['fit'], 2)} in the training rows. A table of fraud rows alone is one line of code.</div>"
          '</div>'
          '<div class="callout warn sm">A table that looks plausible has passed no test yet. The next four slides test it for fidelity, utility and privacy.</div>'))

# ================================================================ GAN · test 1 fidelity
H = GN['hist']
def mini(title, real, synth, xt, w=380, h=190):
    n = len(real)
    ax = Ax(w, h, (0, n), (0, max(max(real), max(synth)) * 1.12), pad=(8, 8, 20, 20))
    s = Rect(ax.l, ax.t, ax.pw, ax.ph, '#FFFFFF', HAIR, 1)
    for i, v in enumerate(real):
        s += ax.vbar(i + 0.12, i + 0.88, v, '#B9C4CA')
    s += ax.line([i + 0.5 for i in range(n)], synth, RED, 1.8)
    for i, v in enumerate(synth):
        s += ax.dot(i + 0.5, v, RED, 2.2)
    for i, lab in xt:
        s += T(ax.X(i + 0.5), h - 6, lab, 9.5, SOFT, 'middle', MONO)
    s += T(ax.l + 2, 13, title, 11, INK, 'start', MONO, 700)
    return svg(w, h, s, title)
ed = H['edges_amount']
amt_ticks = [(min(range(40), key=lambda i: abs(ed[i] - v)), f'£{v}') for v in (1, 10, 100, 1000)]
m1 = mini('amount', H['amount_real'], H['amount_synth'], amt_ticks)
m2 = mini('hour', H['hour_real'], H['hour_synth'], [(0, '0'), (6, '6'), (12, '12'), (18, '18'), (23, '23')])
m3 = mini('category', H['cat_real'], H['cat_synth'], [(i, {'cash_withdrawal': 'cash', 'electronics': 'electr.', 'online_retail': 'online', 'restaurant': 'rest.'}.get(c, c)) for i, c in enumerate(H['cat'])])
m4 = mini('txns_last_24h', H['vel_real'], H['vel_synth'], [(i, str(i)) for i in range(0, 12, 2)])
frows = ''
for c in NUMC + ['category', 'channel']:
    frows += f'<tr><td>{c}</td><td class="r">{FID["synth"][c]:.3f}</td><td class="r">{FID["hold"][c]:.3f}</td></tr>'
worst = max(FID['synth'], key=FID['synth'].get)
add(slide('GAN', 'GAN &#183; Test 1: fidelity', 'Test',
          f"Column by column, the synthetic table against the training rows. The largest gap is {worst} at {FID['synth'][worst]:.3f}; two samples of real data differ by about {max(FID['hold'].values()):.3f}.",
          '<div class="cols w6040"><div class="fig"><div class="cap">grey bars real &#183; red line synthetic</div>'
          f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">{m1}{m2}{m3}{m4}</div></div>'
          '<div class="colstack">'
          f'<table class="d sm"><tr><th>column</th><th class="r">synthetic vs real</th><th class="r">held-out vs real</th></tr>{frows}</table>'
          '<div class="callout sm">Numeric columns: the KS statistic, the largest gap between two cumulative distributions. Categorical columns: total variation distance. 0 is identical, 1 is disjoint.</div>'
          '</div></div>'))

# ================================================================ GAN · test 1 continued: structure
CR = GN['corr']
SHORT = ['hour', 'amt', 'age', 'tenure', 'dist', 'vel', 'fraud']
RDBU = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
def cmat(M, x0, title):
    c = 38
    s = T(x0 + 50, 14, title, 11, INK, 'start', MONO, 700)
    for i in range(7):
        s += T(x0 + 44, 38 + i * c + 22, SHORT[i], 10, SOFT, 'end', MONO)
        s += T(x0 + 50 + i * c + c / 2, 32, SHORT[i][:4], 9.5, SOFT, 'middle', MONO)
        for j in range(7):
            v = M[i][j]
            k = int(round((max(-0.6, min(0.6, v)) + 0.6) / 1.2 * 8))
            s += Rect(x0 + 50 + j * c, 38 + i * c, c - 2, c - 2, RDBU[k])
            if i != j:
                s += T(x0 + 50 + j * c + c / 2 - 1, 38 + i * c + 22, f'{v:.2f}'.replace('0.', '.'), 9.5,
                       '#FFFFFF' if abs(v) > 0.4 else INK, 'middle', MONO)
    return s
cfig = svg(680, 312, cmat(CR['real'], 0, 'real') + cmat(CR['synth'], 340, 'synthetic'), 'Spearman correlation, real and synthetic')
RU = GN['rules']
rule_items = [(k, RU['synth'][k]) for k in RU['synth']]
rb = ''
for k, v in rule_items:
    nrows = round(v / 100 * PRIV['n_synth'])
    rb += f'<tr><td>{k}</td><td class="r">{RU["real"][k]:.2f}%</td><td class="r">{v:.2f}%</td><td class="r">{comma(nrows)}</td></tr>'
worst_rule = max(rule_items, key=lambda t: t[1])
add(slide('GAN', 'GAN &#183; Test 1: fidelity', 'Test continued 1',
          f"Relationships between columns. Correlations match to within {CR['max']:.3f}, and {worst_rule[1]:.2f}% of synthetic rows break a rule that holds in every real row.",
          '<div class="cols">' +
          fig(cfig, 'Spearman correlation of the numeric columns and the fraud flag',
              f"Mean absolute gap {CR['mean']:.3f}, largest {CR['max']:.3f}.") +
          '<div class="colstack">' +
          code("""((df.category == 'cash_withdrawal') &
 (df.channel != 'atm')).mean()""", 'sm') +
          f'<table class="d sm free"><tr><th>rule broken</th><th class="r">real</th><th class="r">synthetic</th><th class="r">rows</th></tr>{rb}</table>'
          '<div class="callout warn sm">A GAN learns that cash withdrawals happen at ATMs as a strong tendency. It is not told the rule, so a share of rows breaks it. A system that validates its input will reject those rows; filter them or build the rule into the generator.</div>'
          '</div></div>'))

# ================================================================ GAN · test 2 utility
U = GN['util']
items = [('real, ROC-AUC', U['trtr'][0], GREEN, ''), ('synthetic, ROC-AUC', U['tstr'][0], RED, ''),
         ('real, PR-AUC', U['trtr'][1], GREEN, ''), ('synthetic, PR-AUC', U['tstr'][1], RED, '')]
ufig = hbars(items, w=640, label_w=190, bar_h=52, gap=26, maxv=1.0, label='Classifier trained on real and on synthetic rows',
             valfmt=lambda v: f'{v:.3f}')
add(slide('GAN', 'GAN &#183; Test 2: utility', 'Test continued 2',
          f"Train the fraud classifier on the synthetic table and test it on the real last 60 days. ROC-AUC drops from {U['trtr'][0]:.3f} to {U['tstr'][0]:.3f} and PR-AUC from {U['trtr'][1]:.3f} to {U['tstr'][1]:.3f}.",
          '<div class="cols w6040">' +
          fig(ufig, f"gradient boosting trained on {comma(N['fit'])} rows, scored on {comma(N['test'])} real test rows",
              'Same model, same test rows. The training table is the one thing that changes.') +
          '<div class="colstack">' +
          code("""m_real  = fraud_model(fit)
m_synth = fraud_model(synth)
evaluate(m_real), evaluate(m_synth)""", 'sm') +
          f"<div class=\"callout sm\">ROC-AUC keeps {U['tstr'][0] / U['trtr'][0] * 100:.0f}% of its value: the synthetic table ranks transactions almost as well. PR-AUC keeps {U['tstr'][1] / U['trtr'][1] * 100:.0f}%: the precise shape of the {N['fraud_fit']} fraud rows, which decides the top of the ranking, is the part the generator reproduces least well.</div>"
          '<div class="callout good sm">This test is the one to quote to a team that will build models on the synthetic table. It answers their question directly.</div>'
          '</div></div>'))

# ================================================================ GAN · test 3 privacy
ed = PRIV['edges']
ymax = max(max(PRIV['h_hold']), max(PRIV['h_synth'])) * 1.25
ax = Ax(660, 380, (0, 2.5), (0, ymax), pad=(14, 12, 14, 30))
s = ax.frame(xticks=[(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5, 2.0, 2.5)], xlab='distance to the closest training row')
for i, v in enumerate(PRIV['h_hold']):
    s += ax.vbar(ed[i], ed[i + 1], v, '#B9C4CA')
mid = [(ed[i] + ed[i + 1]) / 2 for i in range(len(ed) - 1)]
s += ax.line(mid, PRIV['h_synth'], RED, 2.2)
s += ax.line(mid, [min(v, ymax) for v in PRIV['h_copy']], AMBER, 2.2, dash='5 3')
s += L(ax.X(PRIV['p5']), ax.t, ax.X(PRIV['p5']), ax.t + ax.ph, INK, 1.2, 'stroke-dasharray="3 3"')
s += T(ax.X(PRIV['p5']) + 5, ax.t + 14, '5th percentile of held-out', 10, INK, 'start', MONO)
s += T(ax.X(1.45), ax.t + 44, 'grey: held-out real rows', 11, SOFT, 'start', MONO, 700)
s += T(ax.X(1.45), ax.t + 60, 'red: GAN rows', 11, RED, 'start', MONO, 700)
s += T(ax.X(1.45), ax.t + 76, 'amber: training rows + noise', 11, AMBER, 'start', MONO, 700)
pfig = svg(660, 380, s, 'Distance to closest record')
add(slide('GAN', 'GAN &#183; Test 3: privacy', 'Test continued 3',
          f"Distance to the closest record. {pct(PRIV['below_p5']['synth'])} of GAN rows sit closer to a training row than the 5th percentile of held-out real rows; for a noisy copy of the training table it is {pct(PRIV['below_p5']['copy'])}.",
          '<div class="cols w6040">' +
          fig(pfig, f"{comma(N['hold'])} rows of each kind, distance in the encoded space",
              f"Median distance: held-out {PRIV['median']['hold']:.3f}, GAN {PRIV['median']['synth']:.3f}, noisy copy {PRIV['median']['copy']:.3f}.") +
          '<div class="colstack">'
          '<ol class="n"><li>For each synthetic row, find the nearest training row and record the distance.</li>'
          '<li>Do the same for real rows the generator did not see. Those distances are what an honest table produces.</li>'
          '<li>Compare. A generator that memorised its training rows produces distances near 0.</li></ol>'
          f"<div class=\"callout sm\">Second check: is a synthetic row nearer to a training row or to a held-out row? GAN rows choose the training row {pct(PRIV['share']['synth'])} of the time, the noisy copy {pct(PRIV['share']['copy'])}. 50% means the generator shows no preference for rows it was trained on.</div>"
          f"<div class=\"callout good sm\">The test clears the GAN table and condemns the copy. {PRIV['exact']} of {comma(PRIV['n_synth'])} synthetic rows equal a training row.</div>"
          '</div></div>'))

# ================================================================ GAN · drawbacks
add(slide('GAN', 'GAN &#183; Drawbacks', 'Drawbacks',
          'What the three tests found, and what they cannot show.',
          '<div class="cols3">' +
          card('Rules are learned as tendencies', f"{worst_rule[1]:.2f}% of rows break a rule that every real row obeys. Validate synthetic rows with the checks the production table already passes.", 'o big') +
          card('The rare class is the weakest', f"PR-AUC on fraud falls to {U['tstr'][1] / U['trtr'][1] * 100:.0f}% of the real-data value. {N['fraud_fit']} fraud rows are a small sample to learn a distribution from, and they are the rows a fraud team cares about.", 'r big') +
          card('Distance is evidence, not proof', "The closest-record test shows no copying on this run. It gives no bound on what an attacker who holds the generator can infer about one customer.", 'b big') +
          '</div>'
          '<div class="cols nat"><div class="callout"><b>A formal guarantee</b> needs differential privacy: train with DP-SGD, which clips each row&#8217;s gradient and adds noise, and report the privacy budget &#949;. Fidelity and utility fall as &#949; shrinks, so the three tests are run again at the chosen &#949;.</div>'
          '<div class="callout"><b>Before a synthetic table leaves the bank</b>, three numbers go with it: the fidelity gaps, the train-on-synthetic score, and the closest-record result. Rare categories and outliers are checked by hand, because one extreme row can identify one customer.</div></div>'
          '<div class="callout warn sm">Other generators for tables: CTGAN and TVAE (the SDV library) handle mixed column types out of the box, and diffusion models now match or beat GANs on tabular fidelity. The three tests apply to all of them unchanged.</div>'))

# ================================================================ fraud · concept
s = ''
bx = [(20, 'transaction', 'card, amount, time'), (215, 'features', 'velocity, distance, tenure'),
      (410, 'model', 'score from 0 to 1'), (605, 'threshold', 'set by the alert budget'),
      (800, 'analyst queue', 'review, then confirm'), (995, 'label', 'fraud or genuine')]
cols_ = [SLATE, BLUE, PURP, AMBER, RED, GREEN]
for i, (x, t_, sub) in enumerate(bx):
    piece = box(x, 60, 165, 70, t_, sub, cols_[i])
    if i:
        piece += arrow(x - 30, 95, x, 95)
    s += G_(min(i, 4) if i < 5 else 4, piece)
s += G_(5, f'<path d="M1077,130 C1077,226 492,226 492,138" fill="none" stroke="{GREEN}" stroke-width="1.8" stroke-dasharray="5 4"/>' +
        f'<path d="M492,130 L487,141 L497,141 z" fill="{GREEN}"/>' +
        T(785, 216, 'confirmed labels retrain the model', 11, GREEN, 'middle', MONO, 700))
s += G_(1, T(297, 44, 'computed from past events only', 10.5, BLUE, 'middle', MONO))
s += G_(2, T(492, 44, 'milliseconds per transaction', 10.5, PURP, 'middle', MONO))
s += G_(4, T(882, 44, 'days to weeks for a chargeback', 10.5, RED, 'middle', MONO))
pfig2 = svg(1180, 240, s, 'Fraud detection pipeline')
add(slide('Fraud pipeline', 'Fraud pipeline &#183; Concept', 'Concept',
          'A fraud model is one box in a loop. The threshold, the analysts and the delay before a label arrives are design decisions of the same weight.',
          fig(pfig2, 'from a transaction to a confirmed label',
              'Labels arrive late and only for transactions somebody reviewed or disputed, so the training table is always behind the fraud.') +
          '<div class="cols3">' +
          card('Scoring', 'The score is needed before the payment is authorised. Features that need a lookup, such as transactions in the last 24 hours, are kept current by a stream job.', 'p sm') +
          card('Threshold', 'A team of analysts reviews a fixed number of alerts a day. The threshold is whatever score fills that number.', 'o sm') +
          card('Feedback', 'Fraudsters change behaviour once a pattern is blocked. The model is retrained on a schedule and its score distribution is monitored between retrains.', 'g sm') +
          '</div>' + bld(5), 'build'))

# ================================================================ fraud · imbalance
random.seed(2)
per1000 = round(RATE['test'] * 1000)
reds = set(random.sample(range(1000), per1000))
s = ''
for i in range(1000):
    r, c = divmod(i, 40)
    s += Rect(c * 18, r * 18, 15, 15, RED if i in reds else '#C9D3D8')
gfig2 = svg(718, 448, s, 'One thousand transactions, the fraud ones in red')
acc = 1 - N['fraud_test'] / N['test']
add(slide('Fraud pipeline', 'Fraud pipeline &#183; Class imbalance', 'Class imbalance',
          f"{N['fraud_test']} of {comma(N['test'])} test transactions are fraud: {per1000} in every 1,000.",
          '<div class="cols w6040">' + fig(gfig2, '1,000 transactions at the test fraud rate') +
          '<div class="colstack fill"><div class="kv" style="grid-template-columns:1fr">' +
          tile('accuracy of flagging nothing', pct(acc, 2), 'bad') +
          tile('fraud caught by flagging nothing', f"0 of {N['fraud_test']}", 'bad') + '</div>' +
          '<div class="callout sm">Accuracy rewards the model for the 992 grey squares. Fraud models are scored on the red ones: <b>recall</b>, the share of fraud caught, and <b>precision</b>, the share of alerts that are fraud. PR-AUC summarises the two over every threshold.</div>'
          '</div></div>'))

# ================================================================ fraud · example: two models
CU = FR['curve']
ax = Ax(600, 400, (0, 50), (0, N['fraud_test'] * 1.16), pad=(46, 14, 14, 32))
s = ax.frame(yticks=[0, 50, 100, 150, 200, 250], xticks=[(v, str(v)) for v in (5, 10, 20, 30, 40, 50)], ylab='fraud caught', xlab='alerts reviewed per day')
s += L(ax.l, ax.Y(N['fraud_test']), ax.l + ax.pw, ax.Y(N['fraud_test']), INK, 1, 'stroke-dasharray="4 3"')
s += T(ax.l + ax.pw - 4, ax.Y(N['fraud_test']) + 14, f"all {N['fraud_test']} fraud rows", 10, INK, 'end', MONO)
xs = list(range(1, 51))
for key, col, name in [('iso', AMBER, 'isolation forest'), ('gb', GREEN, 'gradient boosting')]:
    ys = [CU[key]['tp'][x * 60 - 1] for x in xs]
    s += ax.line(xs, ys, col, 2.4)
    for x in (5, 10, 20, 50):
        s += ax.dot(x, ys[x - 1], col, 4)
        s += T(ax.X(x) + (-8 if x == 50 else 6), ax.Y(ys[x - 1]) + (16 if key == 'iso' else -8), str(ys[x - 1]), 11, col, 'end' if x == 50 else 'start', MONO, 700)
    s += T(ax.X(30), ax.Y(ys[29]) + (18 if key == 'iso' else -10), name, 11, col, 'middle', MONO, 700)
cfig2 = svg(600, 400, s, 'Fraud caught against alerts per day')
A = FR['auc']
toprows = rowtable(FR['top'], ['amount', 'hour', 'category', 'channel', 'txns_last_24h', 'score', 'is_fraud'], 'd xs tight', hl='is_fraud')
ntop = sum(r['is_fraud'] for r in FR['top'])
add(slide('Fraud pipeline', 'Fraud pipeline &#183; Example', 'Example',
          f"Two detectors on the same test days. With 10 alerts a day the isolation forest catches {FR['budget']['10']['iso']['caught']} of {N['fraud_test']} fraud rows and gradient boosting {FR['budget']['10']['gb']['caught']}.",
          '<div class="cols">' + fig(cfig2, 'train on days 0 to 119, score days 120 to 179',
                                    f"PR-AUC: isolation forest {A['iso'][1]:.3f}, gradient boosting {A['gb'][1]:.3f}.") +
          '<div class="colstack">' +
          code("""iso = IsolationForest(n_estimators=300).fit(X_train)       # no labels
gb  = HistGradientBoostingClassifier().fit(X_train, y_train)""", 'xs') +
          f'<div class="outlbl">the 8 highest gradient-boosting scores in the test days: {ntop} are fraud</div>' + toprows +
          '<div class="callout sm">The isolation forest flags rows that are unusual. Unusual and fraudulent overlap only in part: a large travel purchase is rare and genuine. It is the detector to run while labels do not exist yet, and a feature for the supervised model afterwards.</div>'
          '</div></div>'))

# ================================================================ fraud · widget
ab = {'curve': {k: {'tp': CU[k]['tp'], 'gbp': CU[k]['gbp']} for k in ('iso', 'gb')}, 'days': 60, 'fraud': N['fraud_test'],
      'gbp': round(FR['gbp_total']), 'name': {'iso': 'isolation forest', 'gb': 'gradient boosting'}}
add(slide('Fraud pipeline', 'Fraud pipeline &#183; The alert budget', 'The alert budget',
          f"The threshold follows from how many alerts the analysts can review. Move the budget and read what it buys on the {N['fraud_test']} fraud rows worth £{comma(round(FR['gbp_total']))}.",
          '<div class="wctl"><div class="seg" id="ab-model"><button data-v="gb" class="on">gradient boosting</button><button data-v="iso">isolation forest</button></div>'
          '<label class="rg">alerts reviewed per day: <b id="ab-kl">10</b><input type="range" id="ab-k" min="1" max="50" step="1" value="10"/></label>'
          f'{JSB}</div>'
          '<div class="cols w6040"><div class="fig"><div class="cap">60 test days</div>'
          '<svg id="ab-svg" viewBox="0 0 700 280" width="100%" role="img" aria-label="What the alert budget catches"></svg></div>'
          '<div class="colstack"><div class="out" id="ab-out"></div>'
          '<div class="callout sm">Each extra alert a day costs analyst time and finds less fraud than the one before it. The budget is set where the value of the next fraud caught equals the cost of the reviews needed to find it.</div>'
          '<div class="callout warn sm">A blocked genuine payment has a cost as well: a customer at a till with a declined card. That cost sits in the precision bar.</div>'
          '</div></div>' + jdata('ab-data', ab)))

# ================================================================ fraud · augmentation
AUG = FR['aug']
ks_ = ['0', '500', '2000', '5000']
lo_ = min(min(AUG[k][0] - AUG[k][1], AUG[k][2] - AUG[k][3]) for k in ks_) - 0.012
hi_ = max(max(AUG[k][0] + AUG[k][1], AUG[k][2] + AUG[k][3]) for k in ks_) + 0.012
ax = Ax(620, 310, (-0.5, 3.5), (lo_, hi_), pad=(50, 14, 14, 32))
yt = [round(lo_ + (hi_ - lo_) * i / 4, 3) for i in range(5)]
s = ax.frame(yticks=yt, xticks=[(i, f'+{int(k):,}' if k != '0' else 'none') for i, k in enumerate(ks_)], yfmt=lambda v: f'{v:.3f}', ylab='PR-AUC on the test days', xlab='fraud rows added to training')
for off, (mi, si, col, name) in zip((-0.09, 0.09), [(0, 1, RED, 'GAN fraud rows'), (2, 3, BLUE, 'repeated real fraud rows')]):
    xs_ = [i + off for i in range(4)]
    ys_ = [AUG[k][mi] for k in ks_]
    s += ax.line(xs_, ys_, col, 2)
    for x, k in zip(xs_, ks_):
        m, sd = AUG[k][mi], AUG[k][si]
        s += L(ax.X(x), ax.Y(m - sd), ax.X(x), ax.Y(m + sd), col, 1.6) + ax.dot(x, m, col, 4.5)
    s += T(ax.X(3) + 0.3 * (ax.X(3) - ax.X(2)), ax.Y(ys_[3]) + (26 if mi == 0 else -22), name, 11, col, 'end', MONO, 700)
afig = svg(620, 310, s, 'PR-AUC with added fraud rows')
g5, o5, g0 = AUG['5000'][0], AUG['5000'][2], AUG['0'][0]
best_g = max(AUG[k][0] for k in ks_)
add(slide('Fraud pipeline', 'Fraud pipeline &#183; GAN fraud rows as training data', 'Example continued 1',
          f"Adding generated fraud rows to the training set moved PR-AUC from {g0:.3f} to at most {best_g:.3f}. Repeating real fraud rows reached {max(AUG[k][2] for k in ks_):.3f}.",
          '<div class="cols w6040">' + fig(afig, 'mean of 5 runs, bars show one standard deviation',
                                          f"class_weight='balanced' on the same model scores {FR['cw']:.3f}.") +
          '<div class="colstack">' +
          code("""fake_fraud = sample(G, k, cond=torch.ones(k, 1))
model = fraud_model(pd.concat([train, fake_fraud]))""", 'xs') +
          f"<div class=\"callout sm\">The generator learned fraud from {N['fraud_fit']} rows. Its samples carry no information beyond those rows, and they blur the boundary the classifier is trying to find. On this table the cheap method, repeating real fraud rows, did at least as well.</div>"
          '<div class="callout good sm">Where generated fraud does earn its place: testing. A rule engine, an alert queue or a new model can be load-tested on a stream with 10% fraud without waiting months for real cases.</div>'
          '</div></div>'))

# ================================================================ fraud · checklist
b10 = FR['budget']['10']['gb']
add(slide('Fraud pipeline', 'Fraud pipeline &#183; Checks to run before deployment', 'Checks to run before deployment',
          'Each check is one line on the tables in the notebook, with the output it gave.',
          '<div class="cols nat"><div class="colstack">'
          '<h2 class="s">1. Split by time.</h2>' +
          code("train, test = tx[tx.day < 120], tx[tx.day >= 120]", 'sm') +
          out(f"{comma(N['train'])} rows before day 120, {comma(N['test'])} after", '', '') +
          '<h2 class="s">2. Score on the fraud rows.</h2>' +
          code("average_precision_score(test.is_fraud, score)", 'sm') +
          out(f"{A['gb'][1]:.3f}   (accuracy of flagging nothing: {acc * 100:.2f}%)", '', '') +
          '</div><div class="colstack">'
          '<h2 class="s">3. Set the threshold from the alert budget.</h2>' +
          code("top = np.argsort(-score)[:10 * 60]     # 10 alerts a day", 'sm') +
          out(f"{b10['caught']} of {N['fraud_test']} fraud rows caught, precision {b10['precision'] * 100:.0f}%", '', '') +
          '<h2 class="s">4. Watch the score distribution after release.</h2>' +
          code("psi(score[first_30_days], score[last_30_days])", 'sm') +
          out(f"{FR['psi']:.4f}   (0.2 is the conventional alarm level)", '', '') +
          '</div></div>'
          '<div class="callout">A random split lets the model train on transactions from after the ones it is tested on. In production it scores the future, so the test has to be the future as well.</div>'
          '<div class="callout warn">The simulated table has no drift, which is why the PSI is near 0. Real fraud moves, and the PSI of the score is the first place that shows.</div>'))
