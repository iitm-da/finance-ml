"""Write the knowledge base of a fictional bank ("Northmere Bank") and a question set with known answers.

    python3 llm/build/make_data.py        (run from the repo folder)

Every policy, product, limit and number here is invented for teaching. Writes into llm/data/:
    kb.jsonl          one section per line: doc code, doc title, section title, text, status (current / archived)
    questions.jsonl   question, kind, the section that answers it, the sentence that answers it, accepted answers
"""
import json, os

OUT = 'llm/data/'
os.makedirs(OUT, exist_ok=True)
KB, QS = [], []

def sec(code, doc, title, text, status='current'):
    KB.append({'id': f'{code}#{sum(k["code"] == code for k in KB) + 1}', 'code': code, 'doc': doc, 'section': title,
               'text': ' '.join(text.split()), 'status': status})
    return KB[-1]['id']

def q(sid, evidence, answers, direct, para, code=None):
    """answers: accepted answer strings. direct shares words with the section, para does not, code names the document."""
    s = next(k for k in KB if k['id'] == sid)
    assert evidence in s['text'], (sid, evidence)
    for kind, text in (('direct', direct), ('paraphrase', para), ('code', code)):
        if text:
            QS.append({'question': text, 'kind': kind, 'gold': sid, 'evidence': evidence, 'answers': answers})

# ------------------------------------------------------------------ current accounts: same attributes, five products
ACC = {  # name: (code, monthly fee, ATM/day, online payment/day, fee-free overdraft, foreign fee %, card days, who)
 'Standard Current Account': ('PG-01', 0, 450, 20000, 50, 2.75, 5, 'personal customers aged 18 or over'),
 'Premier Current Account': ('PG-02', 18, 900, 75000, 500, 0, 2, 'customers who pay in at least GBP 5,000 a month'),
 'Student Current Account': ('PG-03', 0, 300, 5000, 1500, 2.75, 5, 'students on a course of two years or longer'),
 'Business Start Account': ('PG-04', 7.5, 600, 40000, 0, 2.95, 4, 'businesses in their first two years of trading'),
 'Business Plus Account': ('PG-05', 25, 1500, 250000, 2000, 1.5, 2, 'businesses with turnover above GBP 2 million'),
}
for name, (code, fee, atm, online, od, fx, days, who) in ACC.items():
    doc = f'Product guide: {name}'
    short = name.replace(' Account', '').replace(' Current', '')
    fee_s = 'There is no monthly fee' if fee == 0 else f'The monthly fee is GBP {fee:g}'
    a = sec(code, doc, 'Eligibility and fees', f'''The {name} is open to {who}. {fee_s} for the {name}.
        The fee, where there is one, is taken on the first working day of the month. Statements are issued monthly and are free
        online; a paper statement costs GBP 2 each.''')
    b = sec(code, doc, 'Daily limits', f'''Holders of the {name} can withdraw up to GBP {atm:,} a day from cash machines.
        Payments made in online banking or the mobile app are limited to GBP {online:,} a day for the {name}.
        Limits reset at midnight UK time. A branch can raise a limit for a single day after an identity check.''')
    c = sec(code, doc, 'Overdraft', f'''An arranged overdraft on the {name} has an interest-free buffer of GBP {od:,}.
        Borrowing above the buffer is charged daily. Unarranged borrowing is refused where possible and no fee is charged for a
        refused payment.''')
    d = sec(code, doc, 'Cards and use abroad', f'''A replacement debit card for the {name} arrives within {days} working days.
        Card payments in a foreign currency on the {name} carry a fee of {fx:g}% of the amount{"" if fx else ", that is, no fee"}.
        A lost card can be frozen in the app at once.''')
    q(a, f'{fee_s} for the {name}.', ['no monthly fee', 'no fee', 'free', 'GBP 0', '£0'] if fee == 0 else [f'{fee:g}'],
      f'What is the monthly fee for the {name}?', f'How much does it cost each month to hold a {short} account?')
    q(b, f'Holders of the {name} can withdraw up to GBP {atm:,} a day from cash machines.', [f'{atm}'],
      f'How much can holders of the {name} withdraw a day from cash machines?',
      f'What is the daily ATM limit on the {short} account?', f'According to {code}, what is the daily cash withdrawal limit?')
    q(b, f'Payments made in online banking or the mobile app are limited to GBP {online:,} a day for the {name}.', [f'{online}'],
      f'What is the daily limit for payments made in online banking for the {name}?',
      f'What is the most a {short} customer can send in one day from the app?')
    q(c, f'An arranged overdraft on the {name} has an interest-free buffer of GBP {od:,}.', [f'{od}'],
      f'What is the interest-free buffer on an arranged overdraft for the {name}?',
      f'How far can a {short} customer go overdrawn before paying interest?')
    q(d, f'A replacement debit card for the {name} arrives within {days} working days.', [f'{days} working days', f'{days} days'],
      f'Within how many working days does a replacement debit card arrive for the {name}?',
      f'How long does a {short} customer wait for a new card after losing one?')
    q(d, f'Card payments in a foreign currency on the {name} carry a fee of {fx:g}% of the amount', [f'{fx:g}%'] if fx else ['no fee', '0%'],
      f'What fee do card payments in a foreign currency carry on the {name}?',
      f'What does a {short} customer pay for using the card overseas?')

# archived sections: superseded numbers that still sit in the knowledge base
sec('PG-01', 'Product guide: Standard Current Account', 'Daily limits (version of January 2023, superseded)',
    '''Holders of the Standard Current Account can withdraw up to GBP 300 a day from cash machines. Payments made in online
    banking are limited to GBP 10,000 a day. This version was replaced in March 2024.''', 'archived')
sec('PG-04', 'Product guide: Business Start Account', 'Eligibility and fees (version of June 2022, superseded)',
    '''The monthly fee is GBP 5 for the Business Start Account, taken on the first working day of the month. This version was
    replaced in April 2024.''', 'archived')

# ------------------------------------------------------------------ credit cards
CARDS = {'Classic Credit Card': ('CC-01', 24.9, 3, 2.5, 12), 'Platinum Credit Card': ('CC-02', 19.9, 2.5, 2, 12),
         'Business Credit Card': ('CC-03', 21.4, 3.5, 5, 15)}
for name, (code, apr, cash, minpay, late) in CARDS.items():
    doc = f'Product guide: {name}'
    a = sec(code, doc, 'Rates', f'''The purchase rate on the {name} is {apr}% APR. Taking cash on the {name} costs {cash}% of the
        amount, with a minimum of GBP 3. Interest on cash is charged from the day of the withdrawal.''')
    b = sec(code, doc, 'Payments', f'''The minimum payment on the {name} is {minpay}% of the balance or GBP 5, whichever is higher.
        A payment that arrives after the due date costs GBP {late}. Customers can set a direct debit for the minimum, a fixed
        sum or the full balance.''')
    q(a, f'The purchase rate on the {name} is {apr}% APR.', [f'{apr}%', f'{apr} %'], f'What is the purchase rate on the {name}?',
      f'What interest does the {name.replace(" Credit Card", "")} card charge on shopping?')
    q(a, f'Taking cash on the {name} costs {cash}% of the amount', [f'{cash}%'], f'What does taking cash on the {name} cost?',
      f'What is the cash advance fee for the {name.replace(" Credit Card", "")} card?')
    q(b, f'A payment that arrives after the due date costs GBP {late}.', [f'{late}'], f'What does a late payment cost on the {name}?',
      f'What is the penalty for missing the due date on the {name.replace(" Credit Card", "")} card?')

# ------------------------------------------------------------------ staff expenses
D = 'Staff expenses policy'
a = sec('EXP-12', D, 'Hotels', '''Hotel rooms are booked through the travel desk. The nightly cap is GBP 210 in London, GBP 140 in
    other UK cities and GBP 190 overseas. A room above the cap needs written approval from a director before travel.''')
q(a, 'The nightly cap is GBP 210 in London', ['210'], 'What is the nightly hotel cap in London?',
  'How much can staff spend on a room for a night in the capital?', 'Under EXP-12, what is the London hotel cap?')
q(a, 'GBP 140 in other UK cities', ['140'], 'What is the nightly hotel cap in UK cities other than London?',
  'What is the room allowance for a night in Manchester?')
b = sec('EXP-12', D, 'Travel by car', '''Staff who use their own car claim 45 pence a mile for the first 10,000 miles in a tax year
    and 25 pence a mile after that. Parking is claimed at cost. Fines are never reimbursed.''')
q(b, '45 pence a mile for the first 10,000 miles', ['45 pence', '45p', '0.45'], 'How much a mile do staff claim for the first 10,000 miles in their own car?',
  'What is the mileage rate for driving a personal vehicle on bank business?')
c = sec('EXP-12', D, 'Meals', '''When travelling overnight, staff may claim up to GBP 35 for an evening meal and GBP 12 for breakfast
    if it is not included in the room rate. Alcohol is not reimbursed.''')
q(c, 'up to GBP 35 for an evening meal', ['35'], 'How much may staff claim for an evening meal when travelling overnight?',
  'What is the dinner allowance on a work trip?')
d = sec('EXP-12', D, 'Submitting a claim', '''Claims are entered in the ExpenseHub system within 60 days of the spend. A receipt is
    required for any item above GBP 10. Claims older than 60 days are paid only with the approval of the finance director.''')
q(d, 'within 60 days of the spend', ['60 days'], 'Within how many days of the spend must a claim be entered in ExpenseHub?',
  'How long do employees have to put in an expense?', 'What is the deadline for claims in EXP-12?')
q(d, 'A receipt is required for any item above GBP 10.', ['10'], 'Above what amount is a receipt required?',
  'From what value does an expense need proof of purchase?')
e = sec('EXP-12', D, 'Approval', '''A line manager approves claims up to GBP 500. A claim above GBP 500 also needs a director.
    Approvers must not approve their own claims or those of their own manager.''')
q(e, 'A line manager approves claims up to GBP 500.', ['500'], 'Up to what amount does a line manager approve claims?',
  'At what claim size does a director have to sign off as well?')

# ------------------------------------------------------------------ complaints
D = 'Complaint handling procedure'
a = sec('CMP-03', D, 'Timescales', '''Every complaint is acknowledged in writing within 3 working days. A final response is sent
    within 8 weeks of the day the complaint was received. If the deadline will be missed, the customer is told why and when
    to expect a reply.''')
q(a, 'acknowledged in writing within 3 working days', ['3 working days', '3 days', 'three working days'], 'Within how many working days is a complaint acknowledged in writing?',
  'How quickly must we confirm to a customer that their grievance has been logged?', 'What acknowledgement deadline does CMP-03 set?')
q(a, 'A final response is sent within 8 weeks', ['8 weeks', 'eight weeks', '56 days'], 'Within how many weeks is a final response sent?',
  'What is the longest we can take to settle a complaint?')
b = sec('CMP-03', D, 'Goodwill payments', '''A complaint handler can offer a goodwill payment of up to GBP 250. A team leader can
    approve up to GBP 1,000. Anything higher goes to the head of customer relations.''')
q(b, 'A complaint handler can offer a goodwill payment of up to GBP 250.', ['250'], 'Up to what amount can a complaint handler offer a goodwill payment?',
  'How much compensation can a front-line agent give without asking anyone?')
q(b, 'A team leader can approve up to GBP 1,000.', ['1000'], 'Up to what goodwill payment can a team leader approve?',
  'What is the supervisor limit for a gesture of goodwill?')
c = sec('CMP-03', D, 'Recording', '''Complaints are logged in CaseTrack on the day they are received, including complaints
    resolved on the first call. The log records the product, the cause and the amount of any redress.''')
q(c, 'Complaints are logged in CaseTrack on the day they are received', ['CaseTrack'], 'In which system are complaints logged?',
  'Where do we record a customer grievance?')
d = sec('CMP-03', D, 'Vulnerable customers', '''A complaint from a customer flagged as vulnerable is passed to the specialist support
    team within 1 working day. The team can pause collections activity for up to 30 days while the complaint is open.''')
q(d, 'pause collections activity for up to 30 days', ['30 days'], 'For how long can the specialist support team pause collections activity?',
  'How long can debt chasing be put on hold for someone in a fragile situation who has complained?')

# ------------------------------------------------------------------ AML
D = 'Anti-money-laundering standard'
a = sec('AML-07', D, 'Enhanced due diligence', '''Enhanced due diligence is required when cash paid into an account exceeds
    GBP 9,000 in any 30 days, when the customer is a politically exposed person, or when the customer is established in a
    high-risk country.''')
q(a, 'cash paid into an account exceeds GBP 9,000 in any 30 days', ['9000'], 'Above what amount of cash paid in over 30 days is enhanced due diligence required?',
  'What level of cash deposits in a month triggers deeper checks on a customer?', 'What cash threshold does AML-07 set for enhanced due diligence?')
b = sec('AML-07', D, 'Periodic review', '''Customer records are reviewed every 12 months for high-risk customers, every 36 months
    for medium-risk customers and every 60 months for low-risk customers. A change of ownership triggers a review at once.''')
q(b, 'every 12 months for high-risk customers', ['12 months', 'annual', 'every year', 'yearly'], 'How often are records reviewed for high-risk customers?',
  'How frequently do we re-check the file of a client rated as dangerous?')
q(b, 'every 60 months for low-risk customers', ['60 months', '5 years', 'five years'], 'How often are records reviewed for low-risk customers?',
  'What is the refresh cycle for the safest category of client?')
c = sec('AML-07', D, 'Reporting suspicion', '''A member of staff who suspects money laundering sends an internal report to the
    nominated officer within 24 hours. The customer must not be told that a report has been made.''')
q(c, 'sends an internal report to the nominated officer within 24 hours', ['24 hours', 'one day', '1 day'], 'Within how many hours is an internal report sent to the nominated officer?',
  'How fast does an employee have to escalate a possible laundering case?')
d = sec('AML-07', D, 'Beneficial owners', '''Form KYC-7 is completed for every person who owns or controls more than 25% of a
    business customer. Identity documents are kept for 5 years after the relationship ends.''')
q(d, 'Form KYC-7 is completed for every person who owns or controls more than 25%', ['25%', '25 %', '25 per cent', '25 percent'], 'Above what ownership share is Form KYC-7 completed?',
  'What stake makes someone a beneficial owner we must identify?', 'When is form KYC-7 needed?')
q(d, 'Identity documents are kept for 5 years after the relationship ends.', ['5 years', 'five years'], 'For how long are identity documents kept after the relationship ends?',
  'How long do we hold on to passport copies once a customer has left?')

# ------------------------------------------------------------------ IT security
D = 'Information security policy'
a = sec('ISP-21', D, 'Passwords', '''Passwords are at least 14 characters long. Privileged accounts change their password every
    90 days; other accounts change it only after a suspected compromise. Multi-factor authentication is required for all
    remote access.''')
q(a, 'Passwords are at least 14 characters long.', ['14'], 'How many characters long must passwords be at least?',
  'What is the minimum length of a login secret?', 'What password length does ISP-21 require?')
q(a, 'Privileged accounts change their password every 90 days', ['90 days'], 'How often do privileged accounts change their password?',
  'What is the rotation period for admin credentials?')
b = sec('ISP-21', D, 'Lost devices', '''A lost or stolen laptop or phone is reported to the Service Desk on extension 4400 within
    1 hour. The device is wiped remotely as soon as it next connects.''')
q(b, 'reported to the Service Desk on extension 4400 within 1 hour', ['1 hour', 'one hour', '60 minutes'], 'Within what time is a lost laptop reported to the Service Desk?',
  'How soon must I tell someone that my work computer has gone missing?')
q(b, 'reported to the Service Desk on extension 4400', ['4400'], 'On which extension is the Service Desk reached to report a lost device?',
  'What number do I ring when my work phone is stolen?')
c = sec('ISP-21', D, 'Patching', '''Critical security patches are applied within 7 days of release and high-severity patches within
    30 days. A system that cannot be patched needs a risk acceptance signed by the chief information security officer.''')
q(c, 'Critical security patches are applied within 7 days of release', ['7 days', 'seven days', 'one week', '1 week'], 'Within how many days of release are critical security patches applied?',
  'What is the deadline for fixing the most severe vulnerabilities?')
d = sec('ISP-21', D, 'Removable media', '''USB storage devices are blocked on all bank laptops. An exception lasts at most 14 days
    and is approved by the head of security operations.''')
q(d, 'An exception lasts at most 14 days', ['14 days', 'two weeks', '2 weeks'], 'For how long at most does an exception for USB storage last?',
  'How long can a memory stick waiver run?')

# ------------------------------------------------------------------ mortgages
D = 'Mortgage servicing guide'
a = sec('MTG-05', D, 'Payment holidays', '''A borrower can take a payment holiday of up to 3 months in any 12-month period,
    provided the account is not in arrears. Interest continues to accrue during the holiday.''')
q(a, 'a payment holiday of up to 3 months in any 12-month period', ['3 months', 'three months'], 'How long a payment holiday can a borrower take in any 12-month period?',
  'For how long can a homeowner pause their repayments in a year?', 'What does MTG-05 allow as the maximum payment holiday?')
b = sec('MTG-05', D, 'Overpayments', '''Borrowers on a fixed rate can overpay up to 10% of the outstanding balance each year without
    a charge. Overpayments above that are subject to the early repayment charge.''')
q(b, 'overpay up to 10% of the outstanding balance each year', ['10%', '10 %', '10 per cent', '10 percent'], 'What share of the outstanding balance can fixed-rate borrowers overpay each year without a charge?',
  'How much extra can someone pay off their home loan annually for free?')
c = sec('MTG-05', D, 'Early repayment charge', '''The early repayment charge is 3% of the amount repaid in the first year of a fixed
    rate, 2% in the second year and 1% in the third year. No charge applies after the fixed rate ends.''')
q(c, 'The early repayment charge is 3% of the amount repaid in the first year', ['3%', '3 %', '3 per cent', '3 percent'], 'What is the early repayment charge in the first year of a fixed rate?',
  'What penalty applies for clearing a home loan within twelve months of fixing?')
d = sec('MTG-05', D, 'Arrears', '''The arrears team contacts the borrower after 1 missed payment. A field agent visit is arranged
    after 3 missed payments. Legal action is not started until at least 6 payments have been missed.''')
q(d, 'A field agent visit is arranged after 3 missed payments.', ['3 missed', 'three missed', '3 payments', 'three payments', '3'], 'After how many missed payments is a field agent visit arranged?',
  'When do we send someone to the door of a borrower who has fallen behind?')
e = sec('MTG-05', D, 'Porting', '''A borrower who moves home can port the existing rate to a new property if the new mortgage
    completes within 90 days of repaying the old one.''')
q(e, 'completes within 90 days of repaying the old one', ['90 days'], 'Within how many days must the new mortgage complete for a rate to be ported?',
  'How long does a mover have to carry their deal over to the next house?')
sec('MTG-05', D, 'Payment holidays (version of 2021, superseded)', '''A borrower can take a payment holiday of up to 6 months in any
    12-month period. This version applied during 2020 and 2021 and was withdrawn in January 2022.''', 'archived')

# ------------------------------------------------------------------ fraud and disputes
D = 'Fraud and disputes handbook'
a = sec('FRD-09', D, 'Card disputes', '''A card dispute can be raised up to 120 days after the transaction date. For claims under
    GBP 500 a provisional credit is applied within 2 working days while the dispute is investigated.''')
q(a, 'A card dispute can be raised up to 120 days after the transaction date.', ['120 days'], 'Up to how many days after the transaction date can a card dispute be raised?',
  'How long does a cardholder have to challenge a purchase?', 'What is the dispute window in FRD-09?')
q(a, 'a provisional credit is applied within 2 working days', ['2 working days', 'two working days', '2 days'], 'Within how many working days is a provisional credit applied for claims under GBP 500?',
  'How fast do we give the money back temporarily on a small disputed purchase?')
b = sec('FRD-09', D, 'Authorised push payment scams', '''A reimbursement decision on an authorised push payment scam is made within
    5 working days. Complex cases can take up to 35 working days. An excess of GBP 100 is deducted from the reimbursement.''')
q(b, 'made within 5 working days', ['5 working days', 'five working days', '5 days'], 'Within how many working days is a reimbursement decision on an authorised push payment scam made?',
  'How soon do we decide whether to refund a customer tricked into sending money?')
q(b, 'An excess of GBP 100 is deducted from the reimbursement.', ['100'], 'What excess is deducted from a scam reimbursement?',
  'How much of the loss does the victim of a transfer scam bear?')
c = sec('FRD-09', D, 'Blocking', '''A card reported as lost, stolen or compromised is blocked at once and cannot be unblocked.
    A replacement is ordered in the same call.''')
d = sec('FRD-09', D, 'Staff reporting', '''Staff who suspect internal fraud use the confidential line on 0800 555 0142 or write to
    the head of internal audit. Reports can be anonymous.''')
q(d, 'use the confidential line on 0800 555 0142', ['0800 555 0142', '08005550142'], 'What is the number of the confidential line for suspected internal fraud?',
  'Which phone line takes whistleblowing calls about colleagues?')

# ------------------------------------------------------------------ HR
D = 'Leave and absence policy'
a = sec('HR-02', D, 'Annual leave', '''Full-time staff have 27 days of annual leave, rising to 30 days after 5 years of service.
    Up to 5 unused days can be carried into the next year and must be used by 31 March.''')
q(a, 'Full-time staff have 27 days of annual leave', ['27'], 'How many days of annual leave do full-time staff have?',
  'What is the holiday entitlement of a new joiner working full hours?', 'How much annual leave does HR-02 give?')
q(a, 'Up to 5 unused days can be carried into the next year', ['5 days', 'five days', '5 unused', '5'], 'How many unused days can be carried into the next year?',
  'How much holiday can roll over?')
b = sec('HR-02', D, 'Sickness', '''Staff can self-certify sickness for up to 7 calendar days. A fit note from a doctor is needed
    from the eighth day. Absence is reported to the line manager before the start of the working day.''')
q(b, 'self-certify sickness for up to 7 calendar days', ['7 calendar days', '7 days', 'seven days'], 'For how many calendar days can staff self-certify sickness?',
  'How long can I be off ill before I need a letter from my GP?')
c = sec('HR-02', D, 'Study leave', '''Staff studying for a qualification agreed with their manager receive 5 days of paid study leave
    a year and the exam fee for the first attempt.''')
q(c, '5 days of paid study leave a year', ['5 days', 'five days'], 'How many days of paid study leave a year do staff receive?',
  'How much time off is given to prepare for professional exams?')
d = sec('HR-02', D, 'Notice periods', '''The notice period is 3 months for staff at grade D and above and 1 month for all other
    grades. Notice is given in writing to the line manager.''')
q(d, 'The notice period is 3 months for staff at grade D and above', ['3 months', 'three months'], 'What is the notice period for staff at grade D and above?',
  'How long must a senior employee work after resigning?')

# ------------------------------------------------------------------ change management
D = 'Technology change standard'
a = sec('CHG-30', D, 'Normal changes', '''A normal change to a production system is submitted to the change advisory board at least
    5 working days before the planned date. Each change carries a tested rollback plan and a named approver from the business.''')
q(a, 'at least 5 working days before the planned date', ['5 working days', 'five working days', '5 days'], 'How many working days before the planned date is a normal change submitted to the change advisory board?',
  'How much lead time does a production release need for sign-off?', 'What notice does CHG-30 require for a normal change?')
b = sec('CHG-30', D, 'Emergency changes', '''An emergency change can be made with the approval of the duty incident manager alone.
    It is reviewed by the change advisory board within 2 working days after it is made.''')
q(b, 'reviewed by the change advisory board within 2 working days', ['2 working days', 'two working days', '2 days'], 'Within how many working days is an emergency change reviewed by the change advisory board?',
  'How soon after an urgent hotfix does the review panel look at it?')
c = sec('CHG-30', D, 'Change freeze', '''No normal changes are made to payment systems between 15 December and 5 January. Emergency
    changes during the freeze need the approval of the chief technology officer.''')
q(c, 'between 15 December and 5 January', ['15 December', '15 Dec', 'December 15'], 'On what date does the change freeze for payment systems begin?',
  'When does the year-end deployment ban start?')
d = sec('CHG-30', D, 'Incidents', '''During a severity 1 incident the incident manager sends an update every 30 minutes. A
    post-incident review is published within 10 working days of resolution.''')
q(d, 'sends an update every 30 minutes', ['30 minutes', 'half an hour', 'half-hour'], 'How often does the incident manager send an update during a severity 1 incident?',
  'How frequently are stakeholders told what is happening in a major outage?')
q(d, 'published within 10 working days of resolution', ['10 working days', 'ten working days', '10 days'], 'Within how many working days of resolution is a post-incident review published?',
  'What is the deadline for the write-up after an outage is fixed?')

# ------------------------------------------------------------------ data retention, branch operations
D = 'Data retention schedule'
a = sec('DRP-11', D, 'Customer records', '''Transaction records are kept for 7 years after the account closes. Recordings of
    customer calls are kept for 6 years. Marketing consent is refreshed every 24 months.''')
q(a, 'Recordings of customer calls are kept for 6 years.', ['6 years', 'six years'], 'For how long are recordings of customer calls kept?',
  'How long do we store audio of conversations with clients?', 'What retention does DRP-11 set for call recordings?')
q(a, 'Transaction records are kept for 7 years after the account closes.', ['7 years', 'seven years'], 'For how long are transaction records kept after the account closes?',
  'How long is payment history held once a customer has gone?')
b = sec('DRP-11', D, 'Premises and recruitment', '''CCTV footage from branches is kept for 31 days unless it is needed for an
    investigation. Data about unsuccessful job applicants is deleted after 12 months.''')
q(b, 'CCTV footage from branches is kept for 31 days', ['31 days'], 'For how long is CCTV footage from branches kept?',
  'How long is security camera video stored?')
D = 'Branch operations manual'
a = sec('BRN-14', D, 'Cash limits', '''A cashier till holds no more than GBP 4,000. Cash above the limit is moved to the vault,
    which is opened only under dual control by two members of staff.''')
q(a, 'A cashier till holds no more than GBP 4,000.', ['4000'], 'How much does a cashier till hold at most?',
  'What is the ceiling on notes kept at a counter position?', 'What till limit does BRN-14 set?')
b = sec('BRN-14', D, 'Counterfeit notes', '''A suspected counterfeit note is kept, recorded on form BRN-14C and sent to the cash
    centre within 2 working days. The customer is given a receipt and is not reimbursed until the note is confirmed genuine.''')
q(b, 'recorded on form BRN-14C', ['BRN-14C'], 'On which form is a suspected counterfeit note recorded?',
  'What paperwork goes with a fake banknote?')

# ------------------------------------------------------------------ questions the knowledge base cannot answer
UNANSWERABLE = [
 'What is the daily cash machine limit on the Youth Saver Account?', 'What is the monthly fee for the Private Banking Account?',
 'What interest rate does the Premier Current Account pay on credit balances?', 'What is the purchase rate on the Student Credit Card?',
 'What is the nightly hotel cap for stays in New York specifically?', 'How much can staff claim for a taxi to the airport?',
 'How many days of paid paternity leave do staff receive?', 'What is the notice period during probation?',
 'How long are emails kept under the retention schedule?', 'What is the minimum length of a PIN for a mobile device?',
 'How often is a penetration test run on payment systems?', 'What is the goodwill payment limit for the chief executive?',
 'What is the maximum loan-to-value for a buy-to-let mortgage?', 'How long does a mortgage valuation take?',
 'What is the chargeback window for a direct debit?', 'What is the reward paid to staff who report internal fraud?',
 'What is the cash limit for the branch vault?', 'Who chairs the change advisory board?',
 'What is the fee for a CHAPS payment from the Business Plus Account?', 'How many complaints did the bank receive last year?',
 'What is the enhanced due diligence threshold for cryptocurrency transfers?', 'What is the dress code in branches?',
 'What cashback does the Platinum Credit Card pay?', 'What is the daily contactless limit on the Standard Current Account?',
 'How many days of annual leave do part-time contractors receive?', 'What is the bank\'s policy on working from another country?',
 'What is the overdraft buffer on the Charity Account?', 'How long is a severity 2 incident allowed to stay open?',
 'What is the mileage rate for an electric company car?', 'What is the retention period for board minutes?']
for t in UNANSWERABLE:
    QS.append({'question': t, 'kind': 'unanswerable', 'gold': None, 'evidence': None, 'answers': []})

with open(OUT + 'kb.jsonl', 'w', encoding='utf-8') as f:
    for k in KB:
        f.write(json.dumps(k, ensure_ascii=False) + '\n')
with open(OUT + 'questions.jsonl', 'w', encoding='utf-8') as f:
    for i, r in enumerate(QS):
        f.write(json.dumps({'qid': i, **r}, ensure_ascii=False) + '\n')
from collections import Counter
print('kb:', len(KB), 'sections in', len({k['code'] for k in KB}), 'documents |', sum(len(k['text'].split()) for k in KB), 'words')
print('questions:', dict(Counter(r['kind'] for r in QS)))
