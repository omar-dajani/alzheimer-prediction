import json, pathlib
import sys
nb_path = sys.argv[1] if len(sys.argv) > 1 else "ADNI_Survival_Pipeline.ipynb"
nb = json.loads(pathlib.Path(nb_path).read_text())
nb["metadata"].pop("widgets", None)
pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1))
