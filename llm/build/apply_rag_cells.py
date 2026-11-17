"""Run only the chunking, query expansion and ontology cells of llm_part.py, put the executed cells into the notebook
and their numbers into build/m_llm.json.

    python llm/build/apply_rag_cells.py        (run with the GPU environment)

A full `make_session_notebook.py llm llms_in_depth` run gives the same cells. This one loads the models, rebuilds the
retrieval index (seconds) and skips every other experiment, so it takes minutes where the full run takes hours.
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
chunk = block('# ## Chunking strategies', '# ## Retrieval and generation together')
qexp = block('# ## Query expansion', 'ex_i = ')
stub = nbformat.v4.new_code_cell("SYS_PLAIN = 'You answer questions from staff of Northmere Bank. Answer in one short sentence.'\n"
                                 "GEN = json.load(open('build/m_llm.json'))['rag']['gen']")
dump = nbformat.v4.new_code_cell("json.dump({'strat': STRAT, 'late': LATE, 'figc': FIGC, 'qe': QEX, 'onto': ONTO}, open('build/_ragx.json', 'w'))")
setup = [code('import os, sys, urllib.request'), code('import json, re, time, warnings'), code('def load_llm(name)'), code('kb, qs = load('),
         code('nrm = lambda s'), stub, code('from rank_bm25 import'), code('full = {}')]
tmp = nbformat.v4.new_notebook(cells=setup + chunk + [code('def rag_prompt(')] + qexp + [dump])
tmp.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
t0 = time.time()
try:
    NotebookClient(tmp, timeout=None, kernel_name='python3', resources={'metadata': {'path': LLM_DIR}}).execute()
finally:
    nbformat.write(tmp, 'build/_ragx_run.ipynb')            # kept for debugging; not part of the repo
print('executed in %.0f s' % (time.time() - t0))

def done(cells):
    out = [c for c in cells if 'build' not in c.metadata.get('tags', [])]
    for c in out:
        if c.cell_type == 'code':
            c.outputs = [o for o in c.outputs if o.get('name') != 'stderr']
    return out

nb = nbformat.read(NB, as_version=4)
at = lambda text: next((i for i, c in enumerate(nb.cells) if c.source.startswith(text)), None)
for start, end, new in (('## Chunking strategies', '## Retrieval and generation together', done(chunk)),
                        ('## Query expansion', '# 2. Evaluating answers', done(qexp))):
    if at(start) is not None:                               # replace an earlier insertion
        del nb.cells[at(start):at(end)]
    nb.cells[at(end):at(end)] = new
nb.cells[at('import os, sys, urllib.request')].source = bodies[first('import os, sys, urllib.request')]
for k, c in enumerate(c for c in nb.cells if c.cell_type == 'code'):
    c.execution_count = k + 1
    for o in c.get('outputs', []):
        if 'execution_count' in o:
            o['execution_count'] = k + 1
nbformat.write(nb, NB)
M = json.load(open('build/m_llm.json'))
M['rag'].update(json.load(open('build/_ragx.json')))
json.dump(M, open('build/m_llm.json', 'w'))
os.remove('build/_ragx.json')
print('notebook now has', len(nb.cells), 'cells; m_llm.json rag has', [k for k in ('strat', 'late', 'figc', 'qe', 'onto') if k in M['rag']])
