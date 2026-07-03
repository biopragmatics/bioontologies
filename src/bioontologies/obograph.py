"""Data structures for representing OBO Graphs.

.. seealso:: https://github.com/geneontology/obographs
"""

from __future__ import annotations

import warnings

from obographs import Definition, Edge, Graph, GraphDocument, Meta, Node, Property, Synonym, Xref

__all__ = [
    "Definition",
    "Edge",
    "Graph",
    "GraphDocument",
    "Meta",
    "Node",
    "Property",
    "Synonym",
    "Xref",
]

warnings.warn(
    "The bioontologies.obograph module is deprecated. Use the `obographs` package instead.",
    DeprecationWarning,
    stacklevel=2,
)
