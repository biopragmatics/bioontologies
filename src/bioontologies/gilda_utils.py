# -*- coding: utf-8 -*-

"""Bioontologies' Gilda utilities."""

import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional

from tqdm.auto import tqdm

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
    parse_results = get_obograph_by_prefix(prefix, **kwargs)
    if parse_results.graph_document is None:
        return
    for graph in parse_results.graph_document.graphs:
        graph.standardize(prefix=prefix, nodes=True, edges=False)
        yield from gilda_from_graph(prefix, graph)


def _fast_term(
    text, prefix, identifier, name, status, source, organism
) -> Optional["gilda.term.Term"]:
    import gilda.term
    from gilda.process import normalize

    try:
        term = gilda.term.Term(
            norm_text=normalize(text),
            text=text,
            db=prefix,
            id=identifier,
            entry_name=name,
            status=status,
            source=source,
            organism=organism,
        )
    except ValueError:
        return None
    return term


def gilda_from_graph(prefix: str, graph: Graph) -> Iterable["gilda.term.Term"]:
    """Get Gilda terms from a given graph."""
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
    for node in tqdm(graph.nodes, leave=False, unit_scale=True, desc=f"{prefix} to Gilda"):
        if not node.name or node.reference is None:
            continue
        if node.reference.prefix != prefix:
            # Don't add references from other namespaces
            continue
        organism = species.get(node.reference.identifier)
        term = _fast_term(
            text=node.name,
            prefix=prefix,
            identifier=node.reference.identifier,
            name=node.name,
            status="name",
            source=prefix,
            organism=organism,
        )
        if term is not None:
            yield term
        for synonym in node.synonyms:
            term = _fast_term(
                text=synonym.value,
                prefix=prefix,
                identifier=node.reference.identifier,
                name=node.name,
                status="synonym",
                source=prefix,
                organism=organism,
            )
            if term is not None:
                yield term
