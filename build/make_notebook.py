"""Build the session notebook from ts_part.py and gan_part.py, execute it, and save it without the deck cells.

    python build/make_notebook.py        (needs torch, prophet, statsmodels, pytorch-forecasting)
"""
import os, re, time
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
INTRO = """# Forecasting and generative models for finance

Machine Learning for Finance, Dr. Arun B Ayyar. This notebook produces every number on the slides.

All data is simulated by `build/make_data.py`. The two CSV files it writes are in `data/`.

Packages: `pip install torch statsmodels prophet scikit-learn pytorch-forecasting lightning`

The whole notebook takes about 25 minutes on a laptop CPU. The Temporal Fusion Transformer and the GAN are the slow cells."""

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
nb.cells = [nbformat.v4.new_markdown_cell(INTRO)] + cells_of('build/ts_part.py') + cells_of('build/gan_part.py')
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
t0 = time.time()
NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': ROOT}}).execute()
print('executed in %.0f s' % (time.time() - t0))
nb.cells = [c for c in nb.cells if 'build' not in c.metadata.get('tags', [])]
nbformat.write(nb, 'forecasting_and_generative_models.ipynb')
print('wrote notebook with', len(nb.cells), 'cells')
