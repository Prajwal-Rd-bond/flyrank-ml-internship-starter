"""Execute and save the capstone top to bottom in the active Python environment."""
import os
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parents[1]
path = root / 'work/notebooks/capstone.ipynb'
with tempfile.TemporaryDirectory(prefix='flyrank-kernel-') as directory:
    import json
    kernel = Path(directory) / 'kernels' / 'capstone'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Capstone pinned environment', 'language': 'python'}))
    manager = KernelSpecManager(kernel_dirs=[str(kernel.parent)])
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=1200, kernel_name='capstone', resources={'metadata': {'path': str(root)}})
    client.create_kernel_manager()
    client.km.kernel_spec_manager = manager
    client.execute()
    nbformat.write(notebook, path)
print('Executed and saved capstone; reports regenerated from computed metrics.')
