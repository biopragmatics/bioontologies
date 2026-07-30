"""Tools for grounding relations."""

import warnings

from .api import get_normalized_label, ground_relation, label_norm

__all__ = [
    "get_normalized_label",
    "ground_relation",
    "label_norm",
]


warnings.warn(
    "bioontologies.relations is deprecated and will be removed in v0.1.0, "
    "use curies-processing instead",
    DeprecationWarning,
    stacklevel=2,
)
