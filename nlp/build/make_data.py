"""Simulate the text data for the NLP session. Nothing here comes from a real bank or a real company.

    python3 nlp/build/make_data.py        (run from the repo folder)

Writes into nlp/data/:
    ner_train.jsonl, ner_test.jsonl   sentences with entity tags (tokens, BIO tags, genre)
    pages_train.csv, pages_test.csv   single pages of eight document types
    loans.jsonl                       400 loan agreements with the six fields to extract
"""
import json, os, re
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
OUT = 'nlp/data/'
os.makedirs(OUT, exist_ok=True)
pick = lambda xs: xs[int(rng.integers(len(xs)))]

# ---------------------------------------------------------------- names (train and test lists are disjoint)
FIRST = ('Aisha Daniel Priya Tomasz Grace Imran Sophie Kwame Hannah Rahul Eleanor Mateo Chloe Farid Niamh Oluwaseun '
         'Marta Callum Yasmin Gareth Ingrid Dev Lucia Stefan Bronwen Tariq Elsa Marcus Anjali Fergus Zofia Hamza '
         'Rosalind Kofi Meera Declan Astrid Nikhil Carys Emeka').split()
LAST = ('Okafor Whitfield Banerjee Kowalski Hargreaves Siddiqui Lindqvist Mensah Thornbury Iyer Castellano Fairweather '
        'Nakamura Abernethy Petrov Ogunleye Ashdown Rahman Delacroix Pemberton Chaudhry Winterbourne Osei Radcliffe '
        'Venkataraman Holloway Achterberg Kilbride Sandhu Marchetti Trevelyan Adeyemi Blackwood Qureshi Stanhope '
        'Lindholm Baptiste Greenhalgh Mukherjee Farrow').split()
STEM = ('Harrowgate Castlebrook Finchley Northmere Aldwick Brightwater Kestrel Thornfield Silverline Meridian Oakhaven '
        'Redcliffe Larkspur Stonebridge Wexcombe Halcyon Ravensworth Tidewell Amberley Coppergate Elmhurst Fenwick '
        'Glenmoor Hollins Ironbridge Juniper Kingsmead Lowther Marlow Netherby Orchard Penhallow Quayside Rosedale '
        'Saltmarsh Tamworth Underhill Valebrook Westmarch Yarrow').split()
TRADE = ('Energy Logistics Foods Pharma Holdings Capital Motors Textiles Software Building Supplies Insurance Media '
         'Engineering Hotels Freight Packaging Dental Care Marine Aerospace').split()
SUFFIX = ['Ltd', 'plc', 'Limited', 'LLP', 'Group', '& Sons', 'Partners']
LOCS = ('Leeds Bristol Cardiff Glasgow Norwich Sheffield Aberdeen Exeter Belfast Durham Swansea Lincoln Plymouth '
        'Dundee Coventry Wrexham Carlisle Ipswich Inverness Truro Hull Reading Stirling Derby Bath York Perth '
        'Chester Luton Newport').split()
BANK4 = ['NRBK', 'HLFX', 'CSBK', 'TDWL', 'MRDN', 'OKHV']
MONTHS = 'January February March April May June July August September October November December'.split()

def split(xs):
    xs = list(xs); rng.shuffle(xs); h = len(xs) // 2
    return xs[:h], xs[h:]

POOL = {'train': {}, 'test': {}}
for key, xs in (('first', FIRST), ('last', LAST), ('stem', STEM), ('loc', LOCS)):
    POOL['train'][key], POOL['test'][key] = split(xs)

def person(p):
    f, l = pick(p['first']), pick(p['last'])
    r = rng.random()
    return f'{f} {l}' if r < .7 else f'{f[0]} {l}' if r < .85 else f'Mr {l}' if r < .93 else f'Dr {f} {l}'

def org(p):
    s = pick(p['stem']); r = rng.random()
    if r < .45: return f'{s} {pick(TRADE)} {pick(SUFFIX)}'
    if r < .75: return f'{s} {pick(SUFFIX)}'
    if r < .9:  return f'{s} {pick(TRADE)}'
    return f'{pick(p["last"])} {pick(p["last"])} {pick(SUFFIX)}'       # a firm named after two people

def money(new=False):
    v = float(np.round(np.exp(rng.normal(7.5, 2.2)), 2)); r = rng.random()
    if new and rng.random() < .25:           # formats that appear only in the test sentences
        return pick([f'{v:,.0f} pounds', f'USD {v:,.0f}', f'£{max(1, v / 1000):.0f}k', f'{v:,.2f} sterling'])
    if v > 2e6 or r < .12:
        m = max(1.1, v / 1e6 if v > 1e6 else rng.uniform(1.1, 90))
        return pick([f'£{m:.1f}m', f'{m:.1f} million pounds', f'GBP {m:.1f} million', f'£{m:.1f} million'])
    if r < .45: return f'£{v:,.2f}'
    if r < .65: return f'GBP {v:,.2f}'
    if r < .8:  return f'£{v:,.0f}'
    if r < .9:  return f'EUR {v:,.0f}'
    return f'{v:,.2f} GBP'

def date(new=False):
    d, m, y = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(2022, 2026)); r = rng.random()
    if new and rng.random() < .25:           # formats that appear only in the test sentences
        th = 'th' if 4 <= d <= 20 or 24 <= d <= 28 else {1: 'st', 2: 'nd', 3: 'rd'}[d % 10]
        return pick([f'{d}{th} {MONTHS[m - 1]} {y}', f'{d:02d}.{m:02d}.{y}', f'{d} {MONTHS[m - 1][:3]} {str(y)[2:]}', f'{d:02d}-{MONTHS[m - 1][:3]}-{y}'])
    if r < .4:  return f'{d} {MONTHS[m - 1]} {y}'
    if r < .6:  return f'{d:02d}/{m:02d}/{y}'
    if r < .75: return f'{y}-{m:02d}-{d:02d}'
    if r < .9:  return f'{d} {MONTHS[m - 1][:3]} {y}'
    return f'{MONTHS[m - 1]} {d}, {y}'

def acct():
    dg = lambda k: ''.join(str(x) for x in rng.integers(0, 10, k))
    if rng.random() < .5:
        return f'GB{dg(2)} {pick(BANK4)} {dg(4)} {dg(4)} {dg(4)} {dg(2)}'
    return f'{dg(2)}-{dg(2)}-{dg(2)} {dg(8)}'

GEN = {'PER': person, 'ORG': org, 'LOC': lambda p: pick(p['loc']), 'MONEY': lambda p: money(p is POOL['test']),
       'DATE': lambda p: date(p is POOL['test']), 'ACCT': lambda p: acct()}

# ---------------------------------------------------------------- sentence templates (train and test differ)
TEMPL = {
 'train': {
  'narrative': ['FASTER PAYMENT TO {PER} REF RENT {DATE}', 'DIRECT DEBIT {ORG} {ACCT}', 'CARD PAYMENT {ORG} {LOC} {MONEY}',
                'STANDING ORDER {PER} {ACCT} {MONEY}', 'BACS CREDIT {ORG} SALARY {DATE}', 'CASH WITHDRAWAL {LOC} {MONEY} {DATE}',
                'TRANSFER FROM {PER} {MONEY}', 'REFUND {ORG} {MONEY} {DATE}'],
  'note': ['{PER} opened a business account for {ORG} at the {LOC} branch on {DATE}.',
           'Source of funds for {PER} is the sale of {ORG}, completed on {DATE} for {MONEY}.',
           'Customer {PER} asked for payments to {ACCT} to be stopped from {DATE}.',
           '{ORG} is registered in {LOC} and {PER} is the only director.',
           'The relationship manager met {PER} of {ORG} in {LOC} to review the {MONEY} facility.',
           'A payment of {MONEY} from {ACCT} to {ORG} was held for review on {DATE}.',
           '{PER} confirmed that {ORG} will repay {MONEY} by {DATE}.',
           'Address check failed for {PER}, who moved from {LOC} to {LOC} in {DATE}.'],
  'news': ['{ORG} agreed to buy {ORG} for {MONEY}, the {LOC} lender said on {DATE}.',
           'Shares in {ORG} fell after chief executive {PER} warned on profits.',
           '{ORG} will close its {LOC} office and cut costs by {MONEY} a year.',
           '{PER} joins {ORG} as finance director from {ORG} on {DATE}.',
           'The regulator fined {ORG} {MONEY} over failings at its {LOC} branch.',
           '{ORG} raised {MONEY} in new debt, {PER} told investors in {LOC}.'],
  'complaint': ['I transferred {MONEY} to {PER} on {DATE} and it never arrived.',
                '{ORG} took {MONEY} from my account {ACCT} without my permission.',
                'On {DATE} your {LOC} branch told me that {PER} would call me back.',
                'My card was charged {MONEY} twice by {ORG} on {DATE}.',
                'I wrote to {PER} at your {LOC} office about the {MONEY} fee and had no reply.',
                'Please refund the {MONEY} that went to {ACCT} on {DATE}.'],
 },
 'test': {
  'narrative': ['BILL PAYMENT {ORG} {ACCT} {MONEY}', 'FPS CREDIT FROM {PER} {DATE}', 'CONTACTLESS {ORG} {LOC}',
                'CHAPS OUT {ORG} {MONEY} {DATE}', 'ATM {LOC} {DATE} {MONEY}', 'SO TO {PER} {ACCT}'],
  'note': ['On {DATE} {PER} asked to add {PER} as a signatory for {ORG}.',
           '{ORG}, based in {LOC}, expects turnover of {MONEY} this year according to {PER}.',
           'Funds of {MONEY} arrived in {ACCT} from {ORG} and were not explained.',
           'Enhanced checks were completed for {PER} at the {LOC} office on {DATE}.',
           'The loan of {MONEY} to {ORG} was approved by {PER}.'],
  'news': ['{LOC} firm {ORG} named {PER} as chair on {DATE}.',
           '{ORG} said it lost {MONEY} after the collapse of {ORG}.',
           'Lender {ORG} is to open a branch in {LOC}, creating jobs worth {MONEY} in wages.',
           '{PER}, who founded {ORG}, sold a stake for {MONEY} on {DATE}.'],
  'complaint': ['{PER} at the {LOC} branch promised a refund of {MONEY} by {DATE}.',
                'A payment to {ORG} left {ACCT} on {DATE} and I did not make it.',
                'I was charged {MONEY} by {ORG} even though I cancelled on {DATE}.',
                'Nobody in {LOC} could tell me why {MONEY} was sent to {PER}.'],
 }}

TOK = re.compile(r"£?\d[\d,./-]*\w*|£|[A-Za-z&]+(?:'[a-z]+)?|\S")

def make_sentence(part):
    genre = pick(['narrative', 'note', 'news', 'complaint'])
    t = pick(TEMPL[part][genre])
    tokens, tags = [], []
    for piece in re.split(r'(\{[A-Z]+\})', t):
        if not piece:
            continue
        if piece.startswith('{'):
            typ = piece[1:-1]
            val = GEN[typ](POOL[part])
            if genre == 'narrative':
                val = val.upper()
            tk = val.split() if typ in ('ACCT',) else TOK.findall(val) if typ in ('PER', 'ORG', 'LOC') else val.split()
            tokens += tk
            tags += [f'B-{typ}'] + [f'I-{typ}'] * (len(tk) - 1)
        else:
            tk = TOK.findall(piece)
            tokens += tk
            tags += ['O'] * len(tk)
    return {'tokens': tokens, 'tags': tags, 'genre': genre}

for part, n in (('train', 2400), ('test', 800)):
    rows = [make_sentence(part) for _ in range(n)]
    with open(f'{OUT}ner_{part}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ents = sum(t.startswith('B-') for r in rows for t in r['tags'])
    print(f'ner_{part}: {n} sentences, {ents} entities')

# ---------------------------------------------------------------- document pages
P = POOL['train']        # document text may reuse any name
def anyorg(): return org(pick([POOL['train'], POOL['test']]))
def anyper(): return person(pick([POOL['train'], POOL['test']]))
def amt(lo=5e3, hi=5e6):
    v = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    v = round(v, -3) if v > 2e4 else round(v, -2)
    return f'GBP {v:,.0f}'
def small(): return f'{rng.uniform(4, 900):,.2f}'
def longdate():
    return f'{int(rng.integers(1, 29))} {pick(MONTHS)} {int(rng.integers(2023, 2027))}'
def pctv(lo=1.5, hi=9.5): return f'{rng.uniform(lo, hi):.2f}%'
LAW = ['England and Wales', 'Scotland', 'Northern Ireland']

SHARED = {
 'legal': [lambda: f'This agreement is governed by the law of {pick(LAW)} and the parties submit to the jurisdiction of its courts.',
           lambda: 'Any notice under this document must be in writing and is deemed received two working days after posting.',
           lambda: 'In this document, words in the singular include the plural and references to a party include its successors.',
           lambda: 'If any provision is found to be unenforceable, the remaining provisions continue in full force.',
           lambda: f'Signed for and on behalf of the parties on {longdate()}.'],
 'statement': [lambda: (lambda m, y: f'Statement period 1 {MONTHS[m]} {y} to 28 {MONTHS[m]} {y}. Page')(int(rng.integers(12)), int(rng.integers(2023, 2026))) + f' {int(rng.integers(1, 6))} of {int(rng.integers(6, 9))}.',
               lambda: 'Please check this statement and tell us within 60 days about any entry you do not recognise.',
               lambda: f'{anyper()}, {int(rng.integers(1, 120))} {pick(["High Street", "Mill Lane", "Station Road", "Church Walk"])}, {pick(LOCS)}.',
               lambda: 'Your deposits are protected up to the limit set by the Financial Services Compensation Scheme.'],
 'letter': [lambda: f'{pick(LOCS)}, {longdate()}.', lambda: 'Yours faithfully,', lambda: 'Our reference is shown at the top of this letter; please quote it when you contact us.'],
}
SPEC = {
 'loan_agreement': ('legal', [
    lambda: f'The Lender agrees to make available to the Borrower a term loan facility in an aggregate principal amount of {amt()}.',
    lambda: f'The Borrower shall repay the loan in equal quarterly instalments, the first falling due on {longdate()}.',
    lambda: f'Interest accrues on the outstanding principal at {pctv()} per annum and is payable in arrear on each repayment date.',
    lambda: 'The Borrower may prepay the whole or any part of the loan on giving not less than ten working days notice to the Lender.',
    lambda: 'An event of default occurs if the Borrower fails to pay any sum when due or breaches a financial covenant.',
    lambda: f'The Borrower shall pay an arrangement fee of {amt(1e3, 5e4)} on the date of first drawdown.']),
 'mortgage_offer': ('legal', [
    lambda: f'We are pleased to offer you a mortgage of {amt(8e4, 9e5)} secured on the property described below.',
    lambda: f'The initial rate is fixed at {pctv(3, 6.5)} until {longdate()}, after which our standard variable rate applies.',
    lambda: f'Your monthly payment will be GBP {rng.uniform(450, 3900):,.2f} over a term of {int(rng.integers(15, 36))} years.',
    lambda: 'Your home may be repossessed if you do not keep up repayments on your mortgage.',
    lambda: f'An early repayment charge of {pctv(1, 5)} of the amount repaid applies during the fixed rate period.',
    lambda: f'This offer is valid until {longdate()} and is subject to a satisfactory valuation of the property.']),
 'guarantee': ('legal', [
    lambda: f'The Guarantor irrevocably guarantees to the Lender the due payment of all sums owed by {anyorg()}.',
    lambda: f'The liability of the Guarantor under this guarantee shall not exceed {amt()}.',
    lambda: 'This guarantee is a continuing security and is not discharged by any intermediate payment.',
    lambda: 'The Lender may claim under this guarantee without first enforcing any other security it holds.',
    lambda: 'The Guarantor waives any right to require the Lender to proceed first against the principal debtor.',
    lambda: 'The Guarantor confirms that it has taken independent legal advice before signing this guarantee.']),
 'account_statement': ('statement', [
    lambda: f'Current account {acct()}. Balance brought forward {small()}.',
    lambda: f'{int(rng.integers(1, 29)):02d} {pick(MONTHS)[:3]}  DIRECT DEBIT {anyorg().upper()}  {small()}  balance {small()}',
    lambda: f'{int(rng.integers(1, 29)):02d} {pick(MONTHS)[:3]}  FASTER PAYMENT {anyper().upper()}  {small()}  balance {small()}',
    lambda: f'Total paid in {small()}. Total paid out {small()}. Balance carried forward {small()}.',
    lambda: f'Arranged overdraft limit GBP {int(rng.integers(1, 30)) * 100}. Overdraft interest charged this period {small()}.',
    lambda: f'{int(rng.integers(1, 29)):02d} {pick(MONTHS)[:3]}  BACS CREDIT {anyorg().upper()} SALARY  {small()}  balance {small()}']),
 'card_statement': ('statement', [
    lambda: f'Credit card ending {int(rng.integers(1000, 9999))}. Credit limit GBP {int(rng.integers(5, 150)) * 100}.',
    lambda: f'Minimum payment {small()} due by {longdate()}. Paying only the minimum will cost you more in interest.',
    lambda: f'{int(rng.integers(1, 29)):02d} {pick(MONTHS)[:3]}  {anyorg().upper()} {pick(LOCS).upper()}  {small()}',
    lambda: f'Purchase rate {pctv(17, 30)} a year. Cash advance rate {pctv(24, 35)} a year.',
    lambda: f'Previous balance {small()}. Payments received {small()}. New balance {small()}.',
    lambda: f'{int(rng.integers(1, 29)):02d} {pick(MONTHS)[:3]}  CASH ADVANCE {pick(LOCS).upper()}  {small()}  fee {small()}']),
 'annual_report': ('none', [
    lambda: f'Revenue for the year rose {pctv(1, 18)} to {amt(5e6, 9e8)}, driven by growth in the {pick(TRADE).lower()} division.',
    lambda: f'Profit before tax was {amt(1e6, 2e8)} compared with {amt(1e6, 2e8)} in the prior year.',
    lambda: f'The directors recommend a final dividend of {rng.uniform(1, 40):.1f} pence per share.',
    lambda: f'Net debt at the year end was {amt(1e6, 5e8)} and the group has undrawn facilities of {amt(1e6, 3e8)}.',
    lambda: 'The directors have a reasonable expectation that the group has adequate resources to continue as a going concern.',
    lambda: 'The principal risks facing the group are set out in the risk report, including credit, market and operational risk.']),
 'complaint_letter': ('letter', [
    lambda: f'I am writing to complain about a payment of {small()} taken from my account on {longdate()}.',
    lambda: 'I have called your helpline three times and each time I was told someone would call me back.',
    lambda: 'I want the charge refunded and an explanation of how this was allowed to happen.',
    lambda: 'If I do not hear from you within eight weeks I will refer the matter to the Financial Ombudsman Service.',
    lambda: f'{anyorg()} took the money without my authority and your branch in {pick(LOCS)} refused to help.',
    lambda: 'This has caused me distress and I have had to borrow money to pay my rent.']),
 'kyc_form': ('none', [
    lambda: f'Full name: {anyper()}. Date of birth: {int(rng.integers(1, 29)):02d}/{int(rng.integers(1, 13)):02d}/{int(rng.integers(1950, 2004))}.',
    lambda: f'Nationality: {pick(["British", "Irish", "Polish", "Indian", "Nigerian", "French"])}. Country of tax residence: United Kingdom.',
    lambda: f'Occupation: {pick(["engineer", "nurse", "teacher", "company director", "accountant", "self-employed builder"])}. Annual income: {amt(1.5e4, 2.5e5)}.',
    lambda: f'Source of funds: {pick(["salary", "sale of property", "inheritance", "business income", "savings"])}.',
    lambda: f'Identity document: {pick(["passport", "driving licence"])} number {int(rng.integers(10**8, 10**9))}, verified on {longdate()}.',
    lambda: f'Is the customer a politically exposed person? {pick(["No", "No", "No", "Yes"])}. Risk rating: {pick(["low", "medium", "high"])}.']),
}

def page(kind):
    grp, spec = SPEC[kind]
    k = int(rng.choice([0, 1, 2, 3], p=[.12, .28, .36, .24])) if grp != 'none' else int(rng.choice([1, 2, 3], p=[.35, .4, .25]))
    paras = [spec[i]() for i in rng.choice(len(spec), k, replace=False)]
    if grp != 'none':
        sh = SHARED[grp]             # a page with k = 0 holds shared paragraphs only
        paras += [sh[i]() for i in rng.choice(len(sh), int(rng.integers(1, 3)) + (k == 0), replace=False)]
    rng.shuffle(paras)
    return ' '.join(paras)

rows = [{'doc_type': k, 'text': page(k)} for k in SPEC for _ in range(250)]
pages = pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)
pages.iloc[:1400].to_csv(OUT + 'pages_train.csv', index=False)
pages.iloc[1400:].to_csv(OUT + 'pages_test.csv', index=False)
print('pages:', len(pages), 'of', pages.doc_type.nunique(), 'types | median words', int(pages.text.str.split().str.len().median()))

# ---------------------------------------------------------------- loan agreements with fields to extract
def choose(variants, p):
    i = int(rng.choice(len(variants), p=p)); return i, variants[i]

def loan():
    borrower, lender, guarantor = anyorg(), pick(['Northmere Bank plc', 'Halcyon Commercial Finance Ltd', 'Tidewell Bank plc', 'Kestrel Lending Limited']), anyorg()
    while guarantor == borrower: guarantor = anyorg()
    principal, fee, rate, default = amt(5e4, 2e7), amt(1e3, 9e4), pctv(2.5, 9.5), pctv(1, 3)
    y0 = int(rng.integers(2023, 2026)); signed = f'{int(rng.integers(1, 29))} {pick(MONTHS)} {y0}'
    first = f'{int(rng.integers(1, 29))} {pick(MONTHS)} {y0 + 1}'
    maturity = f'{int(rng.integers(1, 29))} {pick(MONTHS)} {y0 + int(rng.integers(2, 8))}'
    law = pick(LAW); margin = f'{rng.uniform(1, 4):.2f}%'
    floating = rng.random() < .25
    rate_txt = f'SONIA plus {margin}' if floating else rate
    vi = {}
    vi['parties'], parties = choose([
        f'This agreement is dated {signed} and made between {borrower} (the "Borrower") and {lender} (the "Lender").',
        f'THIS FACILITY AGREEMENT is made on {signed}. PARTIES: (1) {lender} as lender; (2) {borrower} as borrower.',
        f'{lender} (referred to below as "we") agrees with {borrower} (referred to below as "you") on {signed} as follows.'], [.6, .25, .15])
    vi['principal'], prin = choose([
        f'The Lender makes available to the Borrower a term loan in a principal amount of {principal}.',
        f'Facility amount: {principal}. An arrangement fee of {fee} is payable on first drawdown.',
        f'The total sum advanced under this agreement will not exceed {principal}, drawn in one amount.'], [.55, .3, .15])
    feeS = '' if vi['principal'] == 1 else f'The Borrower shall pay an arrangement fee of {fee} on the date of first drawdown.'
    vi['rate'], rt = choose([
        f'Interest accrues on the loan at {rate_txt} per annum.',
        f'The rate of interest is {rate_txt} a year, calculated on the daily balance.',
        f'Interest: {rate_txt}. Default interest is charged at {default} above that rate on overdue sums.'], [.5, .3, .2])
    dflt = '' if vi['rate'] == 2 else f'If a sum is not paid when due, default interest of {default} above the agreed rate applies.'
    vi['maturity'], mat = choose([
        f'The loan shall be repaid in full on {maturity} (the "Final Repayment Date").',
        f'Repayments begin on {first} and the last repayment is due on {maturity}.',
        f'Termination date: {maturity}. All amounts outstanding are due on that date.'], [.5, .3, .2])
    firstS = '' if vi['maturity'] == 1 else f'The first instalment falls due on {first}.'
    vi['law'], lw = choose([
        f'This agreement is governed by the law of {law}.',
        f'Governing law and jurisdiction: {law}.',
        f'The courts of {law} have exclusive jurisdiction and its law applies to this agreement.'], [.6, .25, .15])
    filler = ['The Borrower may prepay the loan on ten working days notice without penalty.',
              f'The obligations of the Borrower are guaranteed by {guarantor} (the "Guarantor").',
              'The Borrower shall supply its audited accounts within 180 days of each financial year end.',
              'An event of default occurs if any sum is unpaid for more than five working days.',
              'Any notice must be in writing and is deemed received two working days after posting.',
              'The Borrower shall not create any security over its assets without the consent of the Lender.']
    mid = [prin, feeS, rt, dflt, firstS, mat] + [filler[i] for i in rng.choice(len(filler), 4, replace=False)]
    mid = [m for m in mid if m]
    rng.shuffle(mid)
    text = ' '.join([parties] + mid + [lw])
    return {'text': text, 'borrower': borrower, 'lender': lender, 'principal': principal, 'rate': rate_txt,
            'maturity': maturity, 'law': law, 'variant': vi}

loans = [loan() for _ in range(400)]
with open(OUT + 'loans.jsonl', 'w', encoding='utf-8') as f:
    for r in loans:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('loans:', len(loans), '| median words', int(np.median([len(r['text'].split()) for r in loans])))
