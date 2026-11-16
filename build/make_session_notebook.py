"""Build a day 2 notebook from <session>/build/<session>_part.py, execute it, save it without the deck cells.

    python build/make_session_notebook.py nlp advanced_nlp_for_finance        (run with the GPU environment)
"""
import os, re, sys, time
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS, NAME = sys.argv[1], sys.argv[2]
os.chdir(os.path.join(ROOT, SESS))
INTRO = open('build/intro.md', encoding='utf-8').read().strip()

def cells_of(path):
    src = open(path, encoding='utf-8').read()
    heads = re.findall(r'^# %%(.*)$', src, flags=re.M)
    bodies = re.split(r'^# %%.*$', src, flags=re.M)[1:]
    out = []
    for h, b in zip(heads, bodies):
        b = b.strip('\n')
        if 'markdown' in h:
            c = nbformat.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in b.split('\n')))
        else:
            c = nbformat.v4.new_code_cell(b)
            if 'build' in h:
                c.metadata['tags'] = ['build']
        out.append(c)
    return out

nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_markdown_cell(INTRO)] + cells_of(f'build/{SESS}_part.py')
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
t0 = time.time()
NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': os.getcwd()}}).execute()
print('executed in %.0f s' % (time.time() - t0))
nb.cells = [c for c in nb.cells if 'build' not in c.metadata.get('tags', [])]
for c in nb.cells:                     # library warnings carry local paths and are not part of the lesson
    if c.cell_type == 'code':
        c.outputs = [o for o in c.outputs if o.get('name') != 'stderr']
nbformat.write(nb, NAME + '.ipynb')
print('wrote', NAME + '.ipynb', 'with', len(nb.cells), 'cells')
