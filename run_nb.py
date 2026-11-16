import nbformat, time, sys
from nbclient import NotebookClient
nb = nbformat.read("Glioma_Grading_R3.ipynb", as_version=4)
t=time.time()
c = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}})
try:
    c.execute()
    print("OK", round(time.time()-t), "s")
finally:
    nbformat.write(nb, "Glioma_Grading_R3_executed.ipynb")
