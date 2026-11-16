"""Run only the 'Applications' cells of ts_part.py on the forecasts saved by the last full run (build/ts_fc.npz),
put the executed cells into the notebook and their numbers into build/m_ts.json.

    python build/apply_app_cells.py        (the same environment as make_notebook.py)

A full `make_notebook.py` run gives the same cells; this avoids retraining the models for a change that only
reads their forecasts.
"""
import json, os, re
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
NB = 'forecasting_and_generative_models.ipynb'
src = open('build/ts_part.py', encoding='utf-8').read()
part = src[src.index('# %% [markdown]\n# ## Applications'):src.index('# %% [build]')]
heads = re.findall(r'^# %%(.*)$', part, flags=re.M)
bodies = [b.strip('\n') for b in re.split(r'^# %%.*$', part, flags=re.M)[1:]]
cells = []
for h, b in zip(heads, bodies):
    if 'markdown' in h:
        cells.append(nbformat.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in b.split('\n'))))
    else:
        cells.append(nbformat.v4.new_code_cell(b))
setup = nbformat.v4.new_code_cell('''import json, time, numpy as np, pandas as pd
_m = json.load(open('build/m_ts.json')); _z = np.load('build/ts_fc.npz')
SEGS, HEAD, H, ORIGINS = _m['segs'], _m['head'], _m['H'], _m['ORIGINS']
ACT, Q = _z['ACT'], _z['Q']
FC = {k: _z[k] for k in ('naive', 'arima', 'arimax', 'prophet', 'lstm', 'tft')}
_Y = np.array([_m['y'][s] for s in SEGS]).T
SCALE = np.abs(_Y[7:_m['TEST0']] - _Y[:_m['TEST0'] - 7]).mean(axis=0)
NAMES = {'naive': 'Seasonal naive', 'arima': 'ARIMA', 'prophet': 'Prophet', 'arimax': 'ARIMA + calendar',
         'lstm': 'LSTM', 'tft': 'TFT (median)'}''')
dump = nbformat.v4.new_code_cell("json.dump(APP, open('build/_app.json', 'w'))")
tmp = nbformat.v4.new_notebook(cells=[setup] + cells + [dump])
NotebookClient(tmp, timeout=3600, kernel_name='python3', resources={'metadata': {'path': ROOT}}).execute()
done = tmp.cells[1:-1]

nb = nbformat.read(NB, as_version=4)
src_of = lambda c: c.source.split('\n')[0]
a = next((i for i, c in enumerate(nb.cells) if src_of(c).startswith('## Applications')), None)
if a is not None:                                   # replace an earlier insertion
    b = next(i for i, c in enumerate(nb.cells) if src_of(c).startswith('# Generative models'))
    del nb.cells[a:b]
b = next(i for i, c in enumerate(nb.cells) if src_of(c).startswith('# Generative models'))
nb.cells[b:b] = done
for k, c in enumerate(c for c in nb.cells if c.cell_type == 'code'):
    c.execution_count = k + 1
    for o in c.get('outputs', []):
        if 'execution_count' in o:
            o['execution_count'] = k + 1
nbformat.write(nb, NB)
M = json.load(open('build/m_ts.json'))
M['app'] = json.load(open('build/_app.json'))
json.dump(M, open('build/m_ts.json', 'w'))
os.remove('build/_app.json')
print('notebook now has', len(nb.cells), 'cells; m_ts.json has app:', list(M['app']))
