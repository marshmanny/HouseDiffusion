"""isort:skip_file"""

import importlib
import os

import hdif.utils.trainer_utils as trainer_utils

METRICS_REGISTRY = trainer_utils.ClassRegistry()

def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        if file == "__pycache__":
            continue
        
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)

# automatically import any Python files in the losses/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "hdif.logging")
