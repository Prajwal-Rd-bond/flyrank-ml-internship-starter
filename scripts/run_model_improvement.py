"""Execute this text-output-only experiment notebook in order without a kernel server."""
import contextlib
import io
import os
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
os.chdir(root)
p=root/'work/notebooks/model_improvement.ipynb'
n=nbformat.read(p,as_version=4)
ns={'__name__':'__main__'}
count=0
for cell in n.cells:
    if cell.cell_type!='code':continue
    count+=1
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(cell.source,str(p),'exec'),ns)
    cell.execution_count=count
    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=output.getvalue())]
nbformat.write(n,p)
print(output.getvalue())
print('Executed every code cell and saved notebook outputs.')
