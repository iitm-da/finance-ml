"""Run only the four added evaluation blocks of llm_part.py (claim-level faithfulness, token probabilities, judge
validation, the RAG triad), put the executed cells into the notebook and their numbers into build/m_llm.json.

    python llm/build/apply_eval_cells.py        (run with the GPU environment)

A full `make_session_notebook.py llm llms_in_depth` run gives the same cells. This one regenerates the test-bed answers
and the two detectors the new cells build on, and skips every other experiment.
"""
import json, os, re, time
import nbformat
from nbclient import NotebookClient

LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(LLM_DIR)
NB = 'llms_in_depth.ipynb'
src = open('build/llm_part.py', encoding='utf-8').read()
heads = re.findall(r'^# %%(.*)$', src, flags=re.M)
bodies = [b.strip('\n') for b in re.split(r'^# %%.*$', src, flags=re.M)[1:]]

def cell(h, b):
    if 'markdown' in h:
        return nbformat.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in b.split('\n')))
    c = nbformat.v4.new_code_cell(b)
    if 'build' in h:
        c.metadata['tags'] = ['build']
    return c

first = lambda text: next(i for i, b in enumerate(bodies) if b.startswith(text))
code = lambda text: cell('', bodies[first(text)])
block = lambda a, b: [cell(heads[i], bodies[i]) for i in range(first(a), first(b))]
more = block('# ## Faithfulness, claim by claim', '# ## Mitigation, one step at a time')
triad = block('# ## One score per stage: the RAG triad', '# # 3. Prompt engineering, measured')
stub1 = nbformat.v4.new_code_cell("SYS_PLAIN = 'You answer questions from staff of Northmere Bank. Answer in one short sentence.'")
stub2 = nbformat.v4.new_code_cell(                                       # the parts of earlier cells that the new ones read
    "N = len(qs)\nA = {'RAG, dense, top 3': chat([rag_prompt(i, 'dense', 3) for i in range(N)], SYS_RAG)}")
stub3 = nbformat.v4.new_code_cell(
    "from sklearn.metrics import roc_auc_score\n"
    "OLD = json.load(open('build/m_llm.json'))['eval']\n"
    "assert {k: G.count(k) for k in ('correct', 'abstained', 'wrong')} == OLD['pool'], 'the test-bed answers differ from the full run'\n"
    "D = {}\n"
    "nli = CrossEncoder('cross-encoder/nli-deberta-v3-small', device=DEV)\n"
    "lab = [nli.model.config.id2label[k].lower() for k in range(3)]\n"
    "pr = torch.tensor(nli.predict([(CTX[i], POOL[i]) for i in range(N)], batch_size=16)).softmax(-1).numpy()\n"
    "D['entailment model'] = 1 - pr[:, lab.index('entailment')]\n"
    "del nli\n"
    "J = chat([f\"Context:\\n{CTX[i]}\\n\\nQuestion: {qs[i]['question']}\\nAnswer: {POOL[i]}\\n\\nIs every fact in the answer stated in the context? Reply YES or NO.\"\n"
    "          for i in range(N)], max_new_tokens=3)\n"
    "D['model as judge'] = np.array([0.0 if j.strip().upper().startswith('YES') else 1.0 for j in J])\n"
    "print('judge flags', int(D['model as judge'].sum()), 'was', OLD['det']['model as judge']['flagged'],\n"
    "      '| entailment flags', int((D['entailment model'] > 0.5).sum()), 'was', OLD['det']['entailment model']['flagged'])")
dump = nbformat.v4.new_code_cell("json.dump(RES['eval_more'], open('build/_evalx.json', 'w'))")
setup = [code('import os, sys, urllib.request'), code('import json, re, time, warnings'), code('def load_llm(name)'), code('kb, qs = load('),
         code('nrm = lambda s'), stub1, code('from rank_bm25 import'), code('def rag_prompt('), stub2, code("POOL = A['RAG, dense, top 3']"), stub3]
for c in more + triad:                                                   # the triad's build cell writes m_llm.json in a full run only
    if 'build' in c.metadata.get('tags', []):
        c.source = c.source.replace("json.dump(RES, open('build/m_llm.json', 'w'))", '')
tmp = nbformat.v4.new_notebook(cells=setup + more + triad + [dump])
tmp.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
t0 = time.time()
try:
    NotebookClient(tmp, timeout=None, kernel_name='python3', resources={'metadata': {'path': LLM_DIR}}).execute()
finally:
    nbformat.write(tmp, 'build/_evalx_run.ipynb')           # kept for debugging; not part of the repo
print('executed in %.0f s' % (time.time() - t0))

def done(cells):
    out = [c for c in cells if 'build' not in c.metadata.get('tags', [])]
    for c in out:
        if c.cell_type == 'code':
            c.outputs = [o for o in c.outputs if o.get('name') != 'stderr']
    return out

nb = nbformat.read(NB, as_version=4)
at = lambda text: next((i for i, c in enumerate(nb.cells) if c.source.startswith(text)), None)
for start, end, new in (('## Faithfulness, claim by claim', '## Mitigation, one step at a time', done(more)),
                        ('## One score per stage: the RAG triad', '# 3. Prompt engineering, measured', done(triad))):
    if at(start) is not None:                               # replace an earlier insertion
        del nb.cells[at(start):at(end)]
    nb.cells[at(end):at(end)] = new
for k, c in enumerate(c for c in nb.cells if c.cell_type == 'code'):
    c.execution_count = k + 1
    for o in c.get('outputs', []):
        if 'execution_count' in o:
            o['execution_count'] = k + 1
nbformat.write(nb, NB)
M = json.load(open('build/m_llm.json'))
M['eval_more'] = json.load(open('build/_evalx.json'))
json.dump(M, open('build/m_llm.json', 'w'))
os.remove('build/_evalx.json')
print('notebook now has', len(nb.cells), 'cells; m_llm.json eval_more has', list(M['eval_more']))
