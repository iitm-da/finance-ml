"""Run a '# %%' script as plain Python from a session folder (nlp/ or llm/).

    python build/run_cells.py nlp build/nlp_part.py [first_cell last_cell]

Markdown cells are skipped, a bare last expression in a cell is printed. The optional cell range (code cells,
counted from 0) is for testing one part of a long script.
"""
import ast, os, re, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', sys.argv[1]))
src = open(sys.argv[2], encoding='utf-8').read()
if os.environ.get('QUICK') and os.path.exists('build/quick.json'):      # debugging only: shrink the experiments
    import json
    for a, b in json.load(open('build/quick.json')):
        assert a in src, a
        src = src.replace(a, b)
cells = re.split(r'^# %%.*$', src, flags=re.M)
heads = re.findall(r'^# %%(.*)$', src, flags=re.M)
code = [(h, b) for h, b in zip(heads, cells[1:]) if 'markdown' not in h]
lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(code) - 1)
g = {'__name__': '__main__'}
for k, (head, body) in enumerate(code):
    if not lo <= k <= hi:
        continue
    print(f'--- cell {k}{head}', flush=True)
    tree = ast.parse(body)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, '<cell>', 'exec'), g)
        v = eval(compile(last, '<cell>', 'eval'), g)
        if v is not None:
            print(v)
    else:
        exec(compile(tree, '<cell>', 'exec'), g)
