"""Write data/questions_multi.jsonl: comparison questions whose answer needs one section from every product of a class.

    python3 build/make_multi.py        (run from llm/)

Each value is read out of data/kb.jsonl with a pattern, so the questions follow the knowledge base. A question is kept
only when one product alone holds the highest or the lowest value.
"""
import json, re

kb = [json.loads(l) for l in open('data/kb.jsonl', encoding='utf-8')]
onto = json.load(open('data/ontology.json', encoding='utf-8'))
num = lambda s: float(s.replace(',', ''))

ACC = [  # attribute, pattern in the section text, highest wording, lowest wording
 ('cash machine limit', r'withdraw up to GBP ([\d,.]+) a day', 'has the highest daily cash machine limit', 'has the lowest daily cash machine limit'),
 ('online payment limit', r'limited to GBP ([\d,.]+) a day', 'allows the largest online payment in a day', 'allows the smallest online payment in a day'),
 ('monthly fee', r'monthly fee is GBP ([\d,.]+)|(no monthly fee)', 'has the highest monthly fee', 'has the lowest monthly fee'),
 ('overdraft buffer', r'interest-free buffer of GBP ([\d,.]+)', 'has the largest interest-free overdraft buffer', 'has the smallest interest-free overdraft buffer'),
 ('foreign currency fee', r'carry a fee of ([\d.]+)%|carry (no fee)', 'charges the highest fee on card payments in a foreign currency', 'charges the lowest fee on card payments in a foreign currency'),
 ('replacement card', r'arrives within (\d+) working days', 'takes the longest to deliver a replacement debit card', 'delivers a replacement debit card fastest'),
]
CARD = [
 ('purchase rate', r'purchase rate on the .*? is ([\d.]+)% APR', 'has the highest purchase rate', 'has the lowest purchase rate'),
 ('cash fee', r'costs ([\d.]+)% of the amount', 'charges the most for taking cash', 'charges the least for taking cash'),
 ('minimum payment', r'minimum payment on the .*? is ([\d.]+)% of the balance', 'has the highest minimum payment as a share of the balance', 'has the lowest minimum payment as a share of the balance'),
 ('late payment', r'after the due date costs GBP ([\d.]+)', 'charges the most for a late payment', 'charges the least for a late payment'),
]
SHORT = {'Standard Current Account': 'Standard', 'Premier Current Account': 'Premier', 'Student Current Account': 'Student',
         'Business Start Account': 'Business Start', 'Business Plus Account': 'Business Plus',
         'Classic Credit Card': 'Classic', 'Platinum Credit Card': 'Platinum', 'Business Credit Card': 'Business Credit'}

def under(cls):
    kids = [c['name'] for c in onto['classes'] if c['parent'] == cls]
    return [i for i in onto['instances'] if i['class'] == cls] + [i for k in kids for i in under(k)]

def value(inst, pat):
    for k in kb:
        if k['code'] == inst['code'] and k['status'] == 'current':
            m = re.search(pat, k['text'])
            if m:
                return (num(m.group(1)) if m.group(1) else 0.0), k['id']
    raise ValueError((inst['name'], pat))

QS = []
for cls, phrase, attrs in (('account', 'account', ACC), ('personal current account', 'personal current account', ACC),
                           ('business account', 'business account', ACC), ('credit card', 'credit card', CARD)):
    inst = under(cls)
    for attr, pat, hi, lo in attrs:
        vals = [value(i, pat) for i in inst]
        for word, pick in ((hi, max), (lo, min)):
            best = pick(v for v, _ in vals)
            win = [i for i, (v, _) in zip(inst, vals) if v == best]
            if len(win) != 1:
                continue
            QS.append({'question': f'Which {phrase} {word}?', 'class': cls, 'attribute': attr, 'gold': [s for _, s in vals],
                       'answers': [win[0]['name'], SHORT[win[0]['name']]],
                       'others': [SHORT[i['name']] for i in inst if i is not win[0]]})
with open('data/questions_multi.jsonl', 'w', encoding='utf-8') as f:
    for r in QS:
        f.write(json.dumps(r) + '\n')
print(len(QS), 'comparison questions |', {c: sum(r['class'] == c for r in QS) for c in dict.fromkeys(r['class'] for r in QS)})
for r in QS[:3] + QS[-2:]:
    print(r)
