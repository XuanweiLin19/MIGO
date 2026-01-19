import importlib
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")


def _config_path(name):
    return os.path.join(CONFIG_DIR, name)

TASKS = {
    "scRNA_scHiC": {
        "train": "migo.tasks.scRNA_scHiC.train",
        "infer": "migo.tasks.scRNA_scHiC.infer",
        "default_configs": {
            "1M": "scRNA_scHiC_1M.json",
            "50K": "scRNA_scHiC_50K.json",
        },
    },
    "scRNA_scATAC": {
        "train": "migo.tasks.scRNA_scATAC.train",
        "infer": "migo.tasks.scRNA_scATAC.infer",
        "default_config": "scRNA_scATAC.json",
    },
    "scRNA_ADT": {
        "train": "migo.tasks.scRNA_ADT.train",
        "infer": "migo.tasks.scRNA_ADT.infer",
        "default_config": "scRNA_ADT.json",
    },
    "RNA_Ribo_seq": {
        "train": "migo.tasks.RNA_Ribo_seq.train",
        "infer": "migo.tasks.RNA_Ribo_seq.infer",
        "default_config": "RNA_Ribo_seq.json",
    },
}


def resolve_config(task_name, mode, config_path=None, resolution=None):
    task = TASKS[task_name]
    if task_name == "scRNA_scHiC":
        if config_path:
            return config_path
        resolved = resolution or "1M"
        return _config_path(task["default_configs"][resolved])
    if config_path:
        return config_path
    default_config = task.get("default_config")
    return _config_path(default_config) if default_config else None


def run_task(task_name, mode, config_path=None, resolution=None, overrides=None):
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    if mode not in ("train", "infer"):
        raise ValueError(f"Unsupported mode: {mode}")
    os.environ["MIGO_TASK"] = task_name
    os.environ.setdefault("MIGO_ISOLATE_TASK", "1")
    module_path = TASKS[task_name][mode]
    config_path = resolve_config(task_name, mode, config_path, resolution)
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"{module_path} does not define main().")
    module.main(config_path, overrides=overrides)
