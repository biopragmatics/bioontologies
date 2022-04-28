"""Data structures for representing OBO Graphs.

.. seealso:: https://github.com/geneontology/obographs
"""

from typing import Any, List, Optional, TypedDict

__all__ = [
    "Edge",
    "Property",
    "Meta",
    "Graph",
    "Graphs",
]


class Edge(TypedDict):
    """Represents an edge in an OBO Graph."""

    sub: str
    pred: str
    obj: str


class Property(TypedDict):
    """Represent a property inside a metadata element."""

    pred: str
    val: str


class Meta(TypedDict):
    """Represents the metadata about a node or ontology."""

    definition: str
    subsets: List
    xrefs: List
    synonyms: List
    comments: List
    version: Optional[str]
    basicPropertyValues: List[Property]  # noqa:N815


class Graph(TypedDict):
    """A graph corresponds to an ontology."""

    id: str
    meta: Meta
    nodes: List
    edges: List[Edge]
    equivalentNodesSets: Any  # noqa:N815
    logicalDefinitionAxioms: Any  # noqa:N815
    domainRangeAxioms: Any  # noqa:N815
    propertyChainAxioms: Any  # noqa:N815


class Graphs(TypedDict):
    """Represents a list of OBO graphs."""

    graphs: List[Graph]
