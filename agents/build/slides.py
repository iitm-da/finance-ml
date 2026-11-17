"""The agents deck: the loop, recorded runs, cost, chain or agent, designs, guardrails, testing. Slides only."""
from common import *

TITLE = 'Agentic AI'
RAIL = 'Machine Learning for Finance &#183; Agentic AI'
OUT = 'agentic_ai.html'
CSS = '''
.cols3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.cols3 .out{white-space:pre-wrap;overflow-wrap:anywhere}
.wctl label.rg.nar{min-width:200px;flex:1}
.cols4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
#tx-tab table td{vertical-align:top}
#tx-tab{min-height:268px}
'''
S = []
add = S.append
TR = TRC['traces']; T0 = TR[0]
MU = ON['multi']['answer directly']; NM = ON['n_multi']; MC = ON['multi_code']
FO = MU['fan-out, current sections only']
SP = INJ['strict prompt']; N_INJ = INJ['n']
call_txt = lambda c: json.dumps({'name': c['name'], 'arguments': c['args']})

def path(pts, col=SOFT, sw=1.6):
    """a line through the points with an arrow head at the last one"""
    s = ''.join(L(*pts[i], *pts[i + 1], col, sw) for i in range(len(pts) - 2))
    return s + arrow(*pts[-2], *pts[-1], col, sw)

# ================================================================ cover
c0 = T0['steps'][0][0]
add(f'''
<section class="slide cover" data-sec="Cover">
  <div class="coverpat">
    <div><span class="h-dim">task    </span><span class="h-val">purchase rate, Platinum card?</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">model   </span><span class="h-bad">&lt;tool_call&gt;</span></div>
    <div><span class="h-dim">        {esc('{"name": "search",')}</span></div>
    <div><span class="h-dim">        {esc('"arguments": {"query": ...}}')}</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">code    </span><span class="h-val">search(...)</span></div>
    <div><span class="h-dim">result  [CC-02#1] ... 19.9% APR</span></div>
    <div>&#160;</div>
    <div><span class="h-dim">model   </span><span class="h-val">ANSWER: 19.9%</span></div>
  </div>
  <div class="kicker">Machine Learning for Finance</div>
  <h1>Agentic AI</h1>
  <div class="pt">A model, tools and a loop</div>
  <div class="rule"></div>
  <p class="sub">An agent is a language model that asks for functions to be run and reads what they return, in a loop, until the
  task is done. The session covers how the loop works, what it costs, where it fails and what a bank puts around it.</p>
  <div class="by"><span>Instructor</span>Dr. Arun B Ayyar</div>
  <table class="meta">
    <tr><td>Duration</td><td>20 minutes</td></tr>
    <tr><td>Follows</td><td>LLMs in Depth &#183; same model, same fictional bank</td></tr>
    <tr><td>Model</td><td>{TRC['model']}</td></tr>
  </table>
</section>
''')

# ================================================================ the loop: concept (build)
s = G_(1, box(10, 112, 150, 60, 'task', 'from a member of staff', INK) + arrow(160, 142, 226, 142) +
       box(230, 100, 210, 84, 'model', 'reads the whole conversation', BLUE))
s += G_(2, arrow(440, 142, 506, 142) + box(510, 100, 270, 84, 'tool call, as text', '{"name": "search", "arguments": ...}', PURP) +
        T(473, 132, 'writes', 10.5, SOFT, 'middle', MONO))
s += G_(3, arrow(780, 142, 846, 142) + box(850, 100, 260, 84, 'code runs the function', 'search( ) · calculator( ) · ...', GREEN) +
        T(813, 132, 'parsed', 10.5, SOFT, 'middle', MONO))
s += G_(4, path([(980, 184), (980, 240), (335, 240), (335, 186)]) + T(657, 232, 'result appended to the conversation · the model is called again', 11.5, INK, 'middle', MONO, 600))
s += G_(5, path([(335, 100), (335, 36), (506, 36)], GREEN, 2) + box(510, 8, 270, 56, 'answer', 'a reply with no tool call ends the loop', GREEN))
s += G_(6, Rect(850, 8, 260, 56, '#FDF3F4', RED, 1.2) + T(980, 31, 'step limit', 13.5, RED, 'middle', BODY, 700) +
        T(980, 50, 'the code stops the loop after n calls', 10.5, SOFT, 'middle', MONO))
add(slide('The loop', 'The loop &#183; Concept', 'Concept',
    'An agent is a model called in a loop. Each turn it either asks for a function to be run or gives the answer.',
    fig(svg(1120, 252, s, 'The agent loop: model, tool call, code, result, model'), 'one task &#183; every arrow into the model is a new model call') +
    '<div class="cols3">'
    '<div class="card sm" data-at="2"><h3>The model writes text</h3><p>A tool call is a few lines of JSON in the reply. The model has no access to the function itself.</p></div>'
    '<div class="card sm g" data-at="3"><h3>The code holds the permissions</h3><p>The loop parses the text and chooses whether to run it. Access rights, limits and logging live here.</p></div>'
    '<div class="card sm o" data-at="6"><h3>The loop needs a stop</h3><p>A model can keep asking for tools. A step limit and a spending limit are set in code.</p></div>'
    '</div>' + bld(6), 'build'))

# ================================================================ the loop: mechanism
tool_json = ('{"name": "search",\n "description": "Search the bank\'s\n   product guides and policies.\n   Returns the two sections that\n   best match the query. ...",\n'
             ' "parameters": {\n   "type": "object",\n   "required": ["query"],\n   "properties": {"query": {\n     "type": "string",\n     "description": "what to\n       look for ..."}}}}')
obs0 = c0['obs'].split('\n')[0]
add(slide('The loop', 'The loop &#183; Mechanism', 'Mechanism',
    f'One recorded run of {TRC["model"]}. The tool is described in the system message, the model replies with a call, the code runs it.',
    '<div class="cols3">'
    '<div class="colstack" data-at="1">' + out(tool_json, 'xs', '1 · the tool, as the model reads it (abridged)') +
    '<div class="callout sm">A name, a description and a typed list of arguments. The description is the only thing the model knows about the tool.</div></div>'
    '<div class="colstack" data-at="2">' + out('task\n' + T0['q'] + '\n\nmodel\n<tool_call>\n' + call_txt(c0) + '\n</tool_call>', 'xs', '2 · the model writes a call') +
    '<div class="callout sm">The chat template tells the model to wrap a call in <code>&lt;tool_call&gt;</code> tags. The loop looks for those tags in the reply.</div></div>'
    '<div class="colstack" data-at="3">' + out('search("' + c0['args']['query'] + '")\n\n' + obs0 + '\n\nmodel\n' + T0['final'], 'xs', '3 · the code runs it, the model reads the result') +
    '<div class="callout good sm">The second reply has no tool call, so the loop ends. Two model calls for one figure.</div></div>'
    '</div>' + bld(3), 'build'))

# ================================================================ the loop: recorded runs (widget)
wd = [{'q': t['q'], 'level': t['level'], 'answer': f"{t['answer']:g}", 'expr': t['expr'], 'grade': t['grade'], 'cause': t['cause'], 'final': t['final'],
       'steps': [[{'name': c['name'], 'args': c['args'], 'obs': c['obs'].split('\n')[0][:230] + (' ...' if len(c['obs'].split('\n')[0]) > 230 else ''), 'err': c['err']} for c in st]
                 for st in t['steps']]} for t in TR]
add(slide('The loop', 'The loop &#183; Example', 'Example',
    f'Five recorded runs of {TRC["model"]} with two tools, search and calculator, on tasks that need one to three sections. {sum(t["grade"] == "correct" for t in TR)} of the 5 end correct.',
    f'<div class="wctl"><button class="wbtn" id="tx-prev">&#8249; task</button><button class="wbtn" id="tx-next">task &#8250;</button>'
    f'<button class="wbtn" id="tx-step" style="background:#B02F3B;border-color:#B02F3B;min-width:110px">next call &#8250;</button>'
    f'<div class="out sm" id="tx-q" style="flex:1;white-space:normal"></div>{JSB}</div>'
    '<div id="tx-tab"></div><div class="out sm" id="tx-v" style="white-space:normal"></div>' + jdata('tx-data', wd)))

# ================================================================ the loop: drawbacks
W1, W3, W4, W5 = TR[4], TR[3], TR[2], TR[1]
bad_expr = W4['steps'][0][-1]['args']['expression']
add(slide('The loop', 'The loop &#183; Drawbacks', 'Drawbacks',
    'Four failures from the five recorded runs, each with the evidence and the change to the loop or the tools that addresses it.',
    table(['failure', 'in the recorded run', 'change'],
          [['<b>Answers from memory</b>', f'Asked for the Premier cash machine limit, the model called no tool and replied <code>{esc(W1["final"])}</code>. The document says {W1["answer"]:g}.',
            'Accept a figure only if it appears in a tool result of that run.'],
           ['<b>Searches for the wrong thing</b>', f'The task is about the foreign currency fee. Both queries asked for the <code>monthly fee</code>.',
            'Narrow tools with typed arguments, such as <code>get_fee(product, fee_type)</code>.'],
           ['<b>Uses results it has not seen</b>', f'Searches and the calculator were requested in one reply, so the expression was <code>{esc(bad_expr[:58])}...</code>',
            'Run one call per reply. Return the error text to the model.'],
           ['<b>Does the arithmetic itself</b>', f'After the calculator refused, the model replied <code>{esc(W4["final"])}</code>. The figures it had retrieved give {W4["expr"]} = {W4["answer"]:,.0f}.',
            'Check in code that the final number is one the calculator returned.']], 'd free') +
    '<div class="cols nat"><div class="callout warn sm"><b>A small model.</b> These runs use a 1.5-billion-weight model, chosen because it runs on a laptop. The four checks apply to a model of any size.</div>'
    '<div class="callout sm"><b>Every fix is code.</b> None of the four changes asks the model to behave better. Each one is a check or a narrower interface outside it.</div></div>'))

# ================================================================ cost (widget)
rg = lambda id_, lab, lo, hi, val, step=1: (f'<label class="rg nar">{lab} <b id="{id_}l"></b><input type="range" id="{id_}" min="{lo}" max="{hi}" value="{val}" step="{step}"/></label>')
add(slide('Cost', 'The loop &#183; Cost', 'Cost',
    'Two pieces of arithmetic. Steps multiply their error rates, and every model call rereads the conversation so far.',
    f'<div class="wctl">{rg("cl-p", "CHANCE ONE STEP IS RIGHT", 70, 99, 95)}{rg("cl-n", "STEPS IN THE TASK", 1, 20, 8)}'
    f'{rg("cl-b", "TOKENS IN THE FIRST CALL", 200, 4000, 600, 100)}{rg("cl-a", "TOKENS ADDED PER STEP", 50, 2000, 300, 50)}{JSB}</div>'
    '<div class="cols"><div class="colstack" style="flex:1.25">' +
    fig('<svg id="cl-fig" viewBox="0 0 560 250" width="100%" role="img" aria-label="Share of tasks with every step right against number of steps"></svg>',
        'grey curves: 0.99, 0.95, 0.90 and 0.80 per step &#183; arithmetic, with steps assumed independent') +
    '</div><div class="colstack"><div class="outlbl">for the sliders above</div><div class="out sm" id="cl-out"></div>'
    '<div class="callout sm"><b>Tokens read grow with the square of the steps.</b> Call k reads the first call plus k &#8722; 1 results. Latency and the bill follow the tokens read.</div>'
    '</div></div>'))

# ================================================================ chain or agent: concept (build)
COLS = [('one model call', 'prompt in, text out', '1 call', SLATE),
        ('retrieve, then answer', 'search, then one call', '1 call', SLATE),
        ('fixed chain', 'call, code, call, code', 'k calls, fixed', BLUE),
        ('router', 'a call picks the branch', 'k calls per branch', BLUE),
        ('agent', 'tools, in a loop', 'calls vary by task', PURP),
        ('several agents', 'a lead hands out work', 'calls vary by task', PURP)]
cw = 176; s = ''
for i, (name, how, calls, col) in enumerate(COLS):
    x = 10 + i * (cw + 8)
    s += G_(i + 1, box(x, 10, cw, 62, name, '', col, tsize=14) + T(x + cw / 2, 96, how, 10.2, INK, 'middle', MONO) + T(x + cw / 2, 116, calls, 10.2, SOFT, 'middle', MONO))
s += L(10, 142, 1108, 142, INK, 1.6) + arrow(1090, 142, 1112, 142, INK, 1.6) + arrow(30, 142, 8, 142, INK, 1.6)
s += T(12, 164, 'the programmer writes the order of the steps', 12, INK, 'start', BODY, 700) + T(1108, 164, 'the model chooses the next step', 12, INK, 'end', BODY, 700)
s += T(12, 182, 'each step can be tested alone · cost is known before the run', 10.5, SOFT, 'start', MONO) + T(1108, 182, 'handles tasks nobody listed · cost is known after the run', 10.5, SOFT, 'end', MONO)
add(slide('Chain or agent', 'Chain or agent &#183; Concept', 'Concept',
    'Six designs, ordered by how much of the control flow is left to the model.',
    fig(svg(1120, 190, s, 'Six designs from one model call to several agents'), 'who decides what happens next') +
    '<div class="cols nat">'
    '<div class="card sm" data-at="3"><h3>Workflow</h3><p>The first four. Model calls sit inside a program whose branches a person wrote. Retrieval-augmented generation from the last session is one.</p></div>'
    '<div class="card sm o" data-at="5"><h3>Agent</h3><p>The last two. The program is the loop, and the path through the tools is different for each task.</p></div>'
    '</div>' + bld(6), 'build'))

# ================================================================ chain or agent: example (measured in the LLM session)
T3 = MU['top 3']
items = [('model compares, top 3 sections', T3['correct'], AMBER, f"{pct(T3['covered'], 0)} of questions had every section"),
         ('model compares, every section', FO['correct'], AMBER, f"{pct(FO['covered'], 0)} had every section"),
         ('model reads, code compares', MC['correct'], GREEN, f"{MC['calls']} model calls, one per section")]
fx_ = ON['figs_ex'] if isinstance(ON.get('figs_ex'), list) else None
add(slide('Chain or agent', 'Chain or agent &#183; Example', 'Example',
    f'From the last session: {NM} questions that compare the products of a class. A fixed chain answers {pct(MC["correct"], 0)}; the same model, given every section and asked to compare, answers {pct(FO["correct"], 0)}.',
    '<div class="cols"><div class="colstack" style="flex:1.2">' +
    fig(hbars(items, w=680, label_w=250, bar_h=30, gap=14, maxv=1.9, valfmt=lambda v: pct(v, 0), label='Correct answers on comparison questions by design'),
        f'correct answers on {NM} comparison questions &#183; measured in the LLMs in Depth notebook') +
    out(f"question   {ON['multi_ex']['q']}\n\nstep 1     the ontology lists the products of the class      (code)\nstep 2     one current section is fetched for each           (code)\n"
        f"step 3     the model reads one figure from each section      ({MC['calls']} calls)\nstep 4     max() or min() names the product                  (code)", 'xs', 'the fixed chain') +
    '</div><div class="colstack">'
    '<div class="card sm"><h3>Use a fixed chain when</h3><p>the steps are the same for every task of the kind, as they are here. Each step has one small job and its own test.</p></div>'
    '<div class="card sm o"><h3>Use an agent when</h3><p>the steps depend on what earlier steps return: an investigation, a support case, a codebase. The path cannot be written down in advance.</p></div>'
    '<div class="callout good sm"><b>Start with the chain.</b> Move a step into the model&#8217;s hands only when the tasks need it.</div>'
    '</div></div>'))

# ================================================================ designs (build)
def mini(x, title, nodes, links, note, k):
    """a small diagram: nodes (label, dx, dy, colour), links between node indices"""
    s = Rect(x, 6, 262, 232, '#FFFFFF', HAIR, 1) + T(x + 12, 28, title, 14, INK, 'start', BODY, 700)
    P = [(x + dx, 44 + dy) for _, dx, dy, _ in nodes]
    for (lab, dx, dy, col), (px, py) in zip(nodes, P):
        s += Rect(px, py, 84, 30, '#FFFFFF', col, 1.4) + T(px + 42, py + 19, lab, 10.5, INK, 'middle', MONO, 600)
    for a, b in links:
        (ax, ay), (bx, by) = P[a], P[b]
        if abs(ay - by) < 4:
            s += arrow(ax + 84, ay + 15, bx, by + 15) if bx > ax else arrow(ax, ay + 15, bx + 84, by + 15)
        else:
            s += arrow(ax + 42, ay + (30 if by > ay else 0), bx + 42, by + (0 if by > ay else 30))
    for i, ln in enumerate(note):
        s += T(x + 12, 208 + i * 15, ln, 10.2, SOFT, 'start', MONO)
    return G_(k, s)
s = mini(8, 'Tool loop (ReAct)', [('model', 12, 20, BLUE), ('tool', 160, 20, GREEN), ('answer', 12, 100, INK)], [(0, 1), (1, 0), (0, 2)],
         ['one model, one tool at a time,', 'until it answers'], 1)
s += mini(286, 'Plan, then execute', [('plan', 12, 6, PURP), ('step 1', 12, 56, BLUE), ('step 2', 160, 56, BLUE), ('answer', 160, 106, INK)], [(0, 1), (1, 2), (2, 3)],
          ['the plan is written first and can', 'be shown to a person for approval'], 2)
s += mini(564, 'Worker and checker', [('worker', 12, 20, BLUE), ('checker', 160, 20, RED), ('answer', 160, 100, INK)], [(0, 1), (1, 0), (1, 2)],
          ['a second call reviews the draft', 'against a rubric and sends it back'], 3)
s += mini(842, 'Lead and workers', [('lead', 86, 6, PURP), ('worker', 12, 62, BLUE), ('worker', 160, 62, BLUE), ('merge', 86, 116, INK)], [(0, 1), (0, 2), (1, 3), (2, 3)],
          ['sub-tasks run in separate', 'conversations, in parallel'], 4)
add(slide('Designs', 'Designs &#183; Concept', 'Concept',
    'Four arrangements of the same parts. Each one is the loop with a different rule for who is called next.',
    fig(svg(1120, 244, s, 'Four agent designs'), 'boxes are model calls or tools &#183; arrows are text passed between them') +
    table(['part', 'what it is', 'in a bank'],
          [['<b>Tools</b>', 'functions with a description and typed arguments; the Model Context Protocol is a common way to publish them', 'policy search, customer look-up, case system, calculator'],
           ['<b>Memory</b>', 'the conversation itself, plus a store the agent reads and writes through tools', 'case notes carried from one contact to the next'],
           ['<b>State</b>', 'a saved record of every call, so a run can be paused, resumed and replayed', 'the audit trail of a decision']], 'd sm free') + bld(4), 'build'))

# ================================================================ guardrails (build)
TIER = [('Read', 'search a policy · look up a balance · calculator', 'runs without asking', 'a wrong call wastes a step', GREEN),
        ('Draft', 'prepare a refund · write a reply to a customer', 'goes to a queue for a member of staff', 'a wrong call is caught at review', AMBER),
        ('Write', 'move money · change a limit · close a case', 'runs only after a named person approves', 'a wrong call is a loss or a breach', RED)]
s = ''
for i, (name, ex, rule, risk, col) in enumerate(TIER):
    x = 10 + i * 372
    g = Rect(x, 6, 356, 132, '#FFFFFF', HAIR, 1) + Rect(x, 6, 356, 6, col) + T(x + 14, 38, name, 18, col, 'start', BODY, 700)
    g += T(x + 14, 62, ex, 10.5, INK, 'start', MONO) + T(x + 14, 92, rule, 13, INK, 'start', BODY, 700) + T(x + 14, 116, risk, 10.5, SOFT, 'start', MONO)
    s += G_(i + 1, g)
g = box(10, 30, 170, 56, 'tool call', 'text from the model', PURP) + arrow(180, 58, 226, 58)
g += box(230, 30, 210, 56, 'allowed for this task?', 'list of tools, set in code', INK) + arrow(440, 58, 486, 58)
g += box(490, 30, 180, 56, 'which tier?', 'read · draft · write', INK)
for j, (lab, sub, col) in enumerate((('run it', 'result goes back to the model', GREEN), ('put it in a queue', 'a member of staff reviews', AMBER), ('wait for approval', 'a named person says yes or no', RED))):
    y = 4 + j * 38
    g += arrow(670, 58, 716, y + 15, col) + Rect(720, y, 390, 30, '#FFFFFF', col, 1.4) + T(734, y + 19, lab, 12.5, INK, 'start', BODY, 700) + T(1098, y + 19, sub, 10.2, SOFT, 'end', MONO)
g += path([(335, 86), (335, 124), (486, 124)], RED) + T(494, 128, 'no: an error message goes back to the model, and the call is logged', 10.5, SOFT, 'start', MONO)
GATE = G_(4, g)
add(slide('Guardrails', 'Guardrails &#183; Concept', 'Concept',
    'Tools are sorted by what a wrong call costs. The rule for each tier is enforced by the loop, outside the model.',
    fig(svg(1120, 144, s, 'Three tiers of tools: read, draft, write'), 'three tiers of tools') +
    fig(svg(1120, 136, GATE, 'The check the loop makes before running a tool call'), 'the check the loop makes on every tool call') +
    '<div class="cols3">'
    f'<div class="callout warn sm" data-at="5"><b>Instructions inside tool results.</b> In the last session a sentence planted in a retrieved section was obeyed in {pct(SP["obeyed"], 0)} of {N_INJ} answers. In a chat that is one wrong reply. With a write tool it is a payment.</div>'
    '<div class="callout sm" data-at="6"><b>Least privilege.</b> Each task gets the tools it needs and the agent acts with the access rights of the member of staff who started it.</div>'
    '<div class="callout sm" data-at="7"><b>Limits and a log.</b> A cap on steps, on tokens and on the amount per action. Every call and result is stored, which gives the audit trail.</div>'
    '</div>' + bld(7), 'build'))

# ================================================================ where it fits
add(slide('Guardrails', 'Guardrails &#183; Example', 'Example',
    'Four tasks in a bank, with the tools an agent would need and the step that stays with a member of staff.',
    table(['task', 'tools the agent calls', 'tier', 'stays with staff'],
          [['<b>Complaint triage</b>', 'case system (read), policy search, draft a reply', 'read, draft', 'sending the final response; any goodwill payment'],
           ['<b>Periodic customer review</b>', 'customer record, company registry, sanctions screening, write the review file', 'read, draft', 'the risk rating and the sign-off'],
           ['<b>Payment investigation</b>', 'ledger look-up, message search, counterparty look-up, propose a correction', 'read, draft', 'posting the correcting entry'],
           ['<b>Incident runbook</b>', 'monitoring queries, log search, restart a service', 'read, write', 'approval of each restart; the post-incident review']], 'd free') +
    '<div class="cols nat"><div class="callout sm"><b>What the four share.</b> Many look-ups across systems, a path that depends on what the look-ups return, and a written result that a person reviews.</div>'
    '<div class="callout good sm"><b>Where to begin.</b> Read and draft tools only. The agent prepares the file and a member of staff takes the action.</div></div>'))

# ================================================================ evaluation: concept (build)
import re
bx = [(10, 120, 'task', INK), (170, 150, 'search call', PURP), (360, 150, 'result', GREEN), (550, 150, 'calculator call', PURP), (740, 150, 'result', GREEN), (930, 180, 'answer', BLUE)]
s = ''
for i, (x, w, lab, col) in enumerate(bx):
    s += box(x, 78, w, 46, lab, '', col)
    if i:
        px, pw = bx[i - 1][0], bx[i - 1][1]
        s += arrow(px + pw, 101, x - 4, 101)
def brace(x1, x2, y, col, up=False):
    d = -8 if up else 8
    return L(x1, y, x2, y, col, 2.4) + L(x1, y, x1, y - d, col, 2.4) + L(x2, y, x2, y - d, col, 2.4)
s += G_(1, brace(930, 1110, 140, BLUE) + T(1020, 162, 'Outcome', 14, BLUE, 'middle', BODY, 700) + T(1020, 180, 'right answer, or right', 10.2, SOFT, 'middle', MONO) +
        T(1020, 194, 'final state of the system', 10.2, SOFT, 'middle', MONO))
s += G_(2, brace(170, 890, 140, PURP) + T(530, 162, 'Path', 14, PURP, 'middle', BODY, 700) +
        T(530, 180, 'the right tools · valid arguments · every needed call made · no wasted steps', 10.2, SOFT, 'middle', MONO) +
        T(530, 194, 'the final figure traced to a tool result', 10.2, SOFT, 'middle', MONO))
s += G_(3, brace(360, 510, 66, GREEN, True) + brace(740, 890, 66, GREEN, True) + T(625, 50, 'Parts', 14, GREEN, 'middle', BODY, 700) +
        T(625, 66, 'each tool tested alone', 10.2, SOFT, 'middle', MONO))
s += G_(4, brace(10, 1110, 22, RED, True) + T(12, 14, 'Whole run', 14, RED, 'start', BODY, 700) +
        T(1108, 14, 'model calls · tokens · seconds · any write action taken without cause', 10.2, SOFT, 'end', MONO))
R2 = TR[1]
add(slide('Evaluation', 'Evaluation &#183; Concept', 'Concept',
    'A chat answer is scored once, at the end. An agent leaves a log of calls, and each level of that log has its own score.',
    fig(svg(1120, 202, s, 'Four levels at which a run of an agent is scored'), 'one run of an agent, from the task to the answer') +
    '<div class="cols3">'
    f'<div class="callout warn sm" data-at="2"><b>Right for the wrong reason.</b> Recorded run 2 ends on the right number, {R2["answer"]:g}, after the calculator refused its expression. An outcome score passes it. A path score fails it.</div>'
    '<div class="callout sm" data-at="2"><b>Reference path.</b> For each test task, write down the tool calls a careful member of staff would make. The agent&#8217;s calls are compared with that list.</div>'
    '<div class="callout sm" data-at="4"><b>Repeats.</b> Run each task k times and report the share solved all k times. A task solved 4 times in 5 fails a customer on the fifth.</div>'
    '</div>' + bld(4), 'build'))

# ================================================================ evaluation: the recorded runs, scored
nums_ = lambda t: set(float(x) for x in re.findall(r'\d+(?:\.\d+)?', t.replace(',', '')))
def score(t):
    calls = [c for st in t['steps'] for c in st]; ok = [c for c in calls if not c['err']]
    seen = set(re.findall(r'\[(\S+?#\d+)\]', ' '.join(c['obs'] for c in ok if c['name'] == 'search')))
    fin = nums_(t['final']); res = nums_(' '.join(c['obs'] for c in ok))
    return {'ok': t['grade'] == 'correct', 'valid': (len(ok), len(calls)), 'sect': (len(set(t['gold']) & seen), len(t['gold'])),
            'ground': bool(fin) and fin <= res, 'calls': (len(calls), len(t['gold']) + (t['level'] > 1))}
SC = [score(t) for t in TR]
yes = lambda v: f'<b style="color:{GREEN}">yes</b>' if v else f'<b style="color:{RED}">no</b>'
frac = lambda a, b: f'<span style="color:{GREEN if a == b and b else RED};font-weight:700">{a} of {b}</span>'
SHORTQ = ['Platinum card purchase rate', 'gap between cash machine limits, three accounts', 'sum of overdraft buffers, three accounts',
          'difference in foreign currency fee, two accounts', 'Premier cash machine limit']
rows = [[f'{i + 1}', q, yes(x['ok']), frac(*x['valid']), frac(*x['sect']), yes(x['ground']), f"{x['calls'][0]} / {x['calls'][1]}"] for i, (q, x) in enumerate(zip(SHORTQ, SC))]
tot = lambda k: (sum(x[k][0] for x in SC), sum(x[k][1] for x in SC))
rows.append(['', '<b>all five</b>', f'<b>{sum(x["ok"] for x in SC)} of 5</b>', '<b>%d of %d</b>' % tot('valid'), '<b>%d of %d</b>' % tot('sect'),
             f'<b>{sum(x["ground"] for x in SC)} of 5</b>', '<b>%d / %d</b>' % tot('calls')])
add(slide('Evaluation', 'Evaluation &#183; Example', 'Example',
    f'The five recorded runs, scored on the outcome and on four properties of the path. The outcome passes {sum(x["ok"] for x in SC)}; the path passes {sum(x["ok"] and x["ground"] and x["valid"][0] == x["valid"][1] for x in SC)}.',
    table(['run', 'task', 'outcome right', 'tool calls that ran', 'needed sections retrieved', 'final figure in a tool result', 'tool calls made / needed'], rows, 'd free') +
    '<div class="cols3">'
    f'<div class="callout warn sm"><b>Run 2.</b> Right outcome, every section retrieved, and a final figure that no tool returned. The model subtracted in its head, which is the step that failed in run 3.</div>'
    f'<div class="callout sm"><b>Run 4.</b> Every call it made is valid JSON and two of three ran. The score that catches it is the sections: {SC[3]["sect"][0]} of {SC[3]["sect"][1]} needed.</div>'
    '<div class="callout good sm"><b>All five columns are code.</b> Each one is a few lines over the log of calls. No second model is needed to score them.</div>'
    '</div>'))

# ================================================================ evaluation: test
add(slide('Evaluation', 'Evaluation &#183; Test', 'Test',
    'Nine scores computed from the log of calls, and the routine that keeps them current.',
    '<div class="cols"><div class="colstack" style="flex:1.45">' +
    table(['score', 'how it is computed', 'level'],
          [['task success', 'tasks with the right answer or final state &#247; tasks', 'outcome'],
           ['solved every time', 'tasks solved in all k repeats &#247; tasks', 'outcome'],
           ['tool-call validity', 'calls that parse and run &#247; calls', 'path'],
           ['tool selection', 'calls that are in the reference path &#247; calls', 'path'],
           ['needed calls made', 'reference calls the agent made &#247; reference calls', 'path'],
           ['step efficiency', 'fewest calls needed &#247; calls made', 'path'],
           ['groundedness', 'final figures found in a tool result &#247; final figures', 'path'],
           ['unsafe actions', 'write calls made without cause &#247; tasks', 'whole run'],
           ['cost per solved task', 'tokens read and written &#247; tasks solved', 'whole run']], 'd sm free') +
    '</div><div class="colstack"><ol class="n">'
    '<li><b>Write tasks with known answers</b> and a reference path. Include one where the right action is to refuse.</li>'
    '<li><b>Score from the log</b> with code. Use a model with a rubric only for tone and policy, and have staff sample its verdicts.</li>'
    '<li><b>Rerun the set</b> when the prompt, a tool description or the model changes.</li>'
    '<li><b>Add every production failure</b> to the set as a new task.</li>'
    '</ol></div></div>'))

# ================================================================ summary
add(slide('Summary', 'Summary', 'Summary',
    'The loop, its cost and its controls.',
    '<div class="kv">' +
    tile('model calls for one figure', '2', 'b') +
    tile(f'recorded runs correct', f'{sum(t["grade"] == "correct" for t in TR)} of {len(TR)}', 'bad') +
    tile('8 steps at 0.95 each', f'{0.95 ** 8:.0%}', 'bad') +
    tile(f'fixed chain, {NM} comparisons', pct(MC['correct'], 0), 'ok') +
    '</div>'
    '<div class="cols3">'
    '<div class="callout"><b>The loop.</b> The model writes a tool call as text, the code runs it and appends the result, and the model is called again until it answers.</div>'
    '<div class="callout"><b>The cost.</b> Error rates multiply across steps and every call rereads the conversation. Where the steps are known in advance, a fixed chain is cheaper and easier to test.</div>'
    '<div class="callout"><b>The controls.</b> Tools in tiers, approval for anything that writes, limits on steps and amounts, and a log of every call. The log is scored on outcome and on path, on a task set rerun at every change.</div>'
    '</div>'))

# ================================================================ references
add(slide('References', 'References', 'References',
    'Papers and guides behind the session.',
    '<div class="cols"><div class="colstack" style="flex:1.3"><h2 class="s">Reading</h2><div class="refs">'
    '<p>Yao et al. (2023). ReAct: synergizing reasoning and acting in language models. <i>ICLR</i>.</p>'
    '<p>Schick et al. (2023). Toolformer: language models can teach themselves to use tools. <i>NeurIPS</i>.</p>'
    '<p>Wang et al. (2023). Plan-and-solve prompting. <i>ACL</i>.</p>'
    '<p>Shinn et al. (2023). Reflexion: language agents with verbal reinforcement learning. <i>NeurIPS</i>.</p>'
    '<p>Schluntz and Zhang (2024). Building effective agents. Anthropic.</p>'
    '<p>Yao et al. (2024). &#964;-bench: a benchmark for tool-agent-user interaction in real-world domains. <i>arXiv</i> 2406.12045.</p>'
    '<p>Liu et al. (2023). AgentBench: evaluating LLMs as agents. <i>arXiv</i> 2308.03688.</p>'
    '<p>Greshake et al. (2023). Not what you&#8217;ve signed up for: compromising real-world LLM-integrated applications with indirect prompt injection. <i>AISec</i>.</p>'
    '<p>Model Context Protocol specification, modelcontextprotocol.io.</p>'
    '</div></div><div class="colstack"><h2 class="s">On the slides</h2>'
    f'<div class="callout sm">The recorded runs use {TRC["model"]} (Apache 2.0) on the knowledge base of the fictional Northmere Bank from the LLMs in Depth session.</div>'
    '<div class="callout sm">The comparison-question and hidden-instruction numbers are measured in <code>llm/llms_in_depth.ipynb</code>.</div>'
    '<div class="callout sm">The cost slide is arithmetic on the sliders, with steps assumed independent.</div>'
    '<div class="callout good sm"><b>Questions?</b></div>'
    '</div></div>'))
