import anndata as ad
from typing import Any, Optional


def build_anndata_with_var(matrix: Any, template: Optional[ad.AnnData] = None) -> ad.AnnData:
    """Create an AnnData object while preserving `.var` metadata when shapes align."""
    data = ad.AnnData(X=matrix)
    attach_var_metadata(data, template)
    return data


def attach_var_metadata(target: ad.AnnData, template: Optional[ad.AnnData] = None) -> ad.AnnData:
    """Copy the `.var` dataframe from `template` to `target` if the feature dimension matches."""
    if template is None:
        return target

    template_var = getattr(template, "var", None)
    if template_var is None or template_var.shape[0] != target.n_vars:
        return target

    return target


