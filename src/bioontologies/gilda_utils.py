"""Bioontologies' Gilda utilities."""

import logging
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .ner import literal_mappings_from_graph
from .obograph import Graph
from .robot import get_obograph_by_prefix

if TYPE_CHECKING:
    import gilda.term

__all__ = [
    "get_gilda_terms",
    "gilda_from_graph",
]

logger = logging.getLogger(__name__)


def get_gilda_terms(prefix: str, **kwargs: Any) -> Iterable["gilda.term.Term"]:
    """Get gilda terms for the given namespace.

    :param prefix:
        The prefix of the ontology to load. Will look up the "best" resource
        via the :mod:`bioregistry` and convert with ROBOT.
    :param kwargs:
        Keyword arguments to pass to :func:`bioontologies.get_obograph_by_prefix`
    :yields: Term objects for Gilda

    Example usage:

    .. code-block::

        import bioontologies
        import gilda

        terms = bioontologies.get_gilda_terms("go")
        grounder = gilda.make_grounder(terms)
        scored_matches = grounder.ground("apoptosis")

    Some ontologies don't parse nicely with ROBOT because they have malformed
    entries. To disregard these, you can use the ``check=False`` argument:

    .. code-block::

        import bioontologies
        import gilda

        terms = bioontologies.get_gilda_terms("vo", check=False)
        grounder = gilda.make_grounder(terms)
        scored_matches = grounder.ground("comirna")
    """
    warnings.warn(
        "prefer to use bioontologies.get_literal_mappings() directly and convert to gilda yourself",
        stacklevel=2,
    )
    parse_results = get_obograph_by_prefix(prefix, **kwargs)
    if parse_results.graph_document is None:
        return
    for graph in parse_results.graph_document.graphs:
        graph.standardize(prefix=prefix, nodes=True, edges=False)
        yield from gilda_from_graph(prefix, graph)


def _get_species(graph: Graph, prefix: str) -> dict[str, str]:
    species = {}
    for edge in graph.edges:
        if not edge.standardized:
            edge.standardize()
        if (
            edge.subject
            and edge.subject.prefix == prefix
            and edge.predicate
            and edge.predicate.pair == ("ro", "0002162")
            and edge.object
            and edge.object.prefix == "ncbitaxon"
        ):
            species[edge.subject.identifier] = edge.object.identifier
    return species


def gilda_from_graph(prefix: str, graph: Graph) -> Iterable["gilda.term.Term"]:
    """Get Gilda terms from a given graph."""
    id_to_species = _get_species(graph=graph, prefix=prefix)
    for term in literal_mappings_from_graph(graph=graph, prefix=prefix):
        yield term.to_gilda(id_to_species.get(term.reference.identifier))
