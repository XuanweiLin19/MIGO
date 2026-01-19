"""Task-specific encoder/decoder layers."""

import importlib
import os

_SYMBOLS = {
    "scRNA_ADT_Encoder": (".scRNA_ADT", "RA_VQVAE_Encoder"),
    "scRNA_ADT_Decoder": (".scRNA_ADT", "RA_VQVAE_Decoder"),
    "scRNA_scATAC_Encoder": (".scRNA_scATAC", "RA_VQVAE_Encoder"),
    "scRNA_scATAC_Decoder": (".scRNA_scATAC", "RA_VQVAE_Decoder"),
    "scRNA_scATAC_CrossVQ": (".scRNA_scATAC", "Cross_VQ_RA"),
    "scRNA_scHiC_Encoder": (".scRNA_scHiC", "RA_VQVAE_Encoder"),
    "scRNA_scHiC_Decoder": (".scRNA_scHiC", "RA_VQVAE_Decoder"),
    "RNA_Ribo_seq_Encoder": (".RNA_Ribo_seq", "RA_VQVAE_Encoder"),
    "RNA_Ribo_seq_Decoder": (".RNA_Ribo_seq", "RA_VQVAE_Decoder"),
}

_TASK_SYMBOLS = {
    "scRNA_scATAC": {"scRNA_scATAC_Encoder", "scRNA_scATAC_Decoder", "scRNA_scATAC_CrossVQ"},
    "scRNA_scHiC": {"scRNA_scHiC_Encoder", "scRNA_scHiC_Decoder"},
    "scRNA_ADT": {"scRNA_ADT_Encoder", "scRNA_ADT_Decoder"},
    "RNA_Ribo_seq": {"RNA_Ribo_seq_Encoder", "RNA_Ribo_seq_Decoder"},
}


def __getattr__(name):
    if name in _SYMBOLS:
        if os.environ.get("MIGO_ISOLATE_TASK") == "1":
            task_name = os.environ.get("MIGO_TASK")
            allowed = _TASK_SYMBOLS.get(task_name)
            if allowed is not None and name not in allowed:
                raise ImportError(
                    f"Isolation enabled for {task_name!r}; refusing to import {name!r}."
                )
        module_path, attr = _SYMBOLS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_SYMBOLS.keys())
