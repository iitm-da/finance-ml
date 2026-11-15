"""Run a '# %%' script as plain Python from the lloyds/ folder.

    python build/run_part.py build/ts_part.py

Markdown cells are skipped, a bare last expression in a cell is printed.
"""
import ast, os, re, sys

os.chdir('/mnt/e/iitm course/lloyds')
src = open(sys.argv[1], encoding='utf-8').read()
cells = re.split(r'^# %%.*$', src, flags=re.M)
heads = re.findall(r'^# %%(.*)$', src, flags=re.M)
g = {'__name__': '__main__'}
for head, body in zip(heads, cells[1:]):
    if 'markdown' in head:
        continue
    tree = ast.parse(body)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, '<cell>', 'exec'), g)
        v = eval(compile(last, '<cell>', 'eval'), g)
        if v is not None:
            print(v)
    else:
        exec(compile(tree, '<cell>', 'exec'), g)
