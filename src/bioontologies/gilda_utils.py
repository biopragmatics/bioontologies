# -*- coding: utf-8 -*-

"""Bioontologies' Gilda utilities."""

import logging
from typing import TYPE_CHECKING, Any, Iterable

from tqdm.auto import tqdm

from .obograph import Graph
from .robot import get_obograph_by_prefix

if TYPE_CHECKING:
    import gilda.term

__all__ = [
    "get_gilda_terms",
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
        graph.standardize(prefix=prefix)
        yield from _gilda_from_graph(prefix, graph)


def _gilda_from_graph(prefix: str, graph: Graph) -> Iterable["gilda.term.Term"]:
    import gilda.term
    from gilda.process import normalize

    species = {}
    for edge in graph.edges:
        s, p, o = edge.parse_curies()
        if s[0] == prefix and p == ("ro", "0002162") and o[0] == "ncbitaxon":
            species[s[1]] = o[1]
    for node in tqdm(graph.nodes, leave=False):
        name = node.lbl
        if not name:
            continue
        organism = species.get(node.luid)
        yield gilda.term.Term(
            norm_text=normalize(name),
            text=name,
            db=prefix,
            id=node.luid,
            entry_name=name,
            status="name",
            source=prefix,
            organism=organism,
        )
        for synonym in node.synonyms:
            yield gilda.term.Term(
                norm_text=normalize(synonym.val),
                text=synonym.val,
                db=prefix,
                id=node.luid,
                entry_name=name,
                status="synonym",
                source=prefix,
                organism=organism,
            )
