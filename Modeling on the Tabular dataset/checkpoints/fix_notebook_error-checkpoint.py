import json, pathlib

nb_path = "ADNI_Survival_Pipeline.ipynb"  # replace with your notebook filename
nb = json.loads(pathlib.Path(nb_path).read_text())
nb["metadata"].pop("widgets", None)
pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1))
