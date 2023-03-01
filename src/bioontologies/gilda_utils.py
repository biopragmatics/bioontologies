# -*- coding: utf-8 -*-

"""Bioontologies' Gilda utilities."""

import logging
from typing import Iterable

import gilda.api
import gilda.term
from gilda.process import normalize
from tqdm.auto import tqdm

from bioontologies import get_obograph_by_prefix
from bioontologies.obograph import Graph

__all__ = [
    "get_gilda_terms",
]

logger = logging.getLogger(__name__)


def get_gilda_terms(prefix: str) -> Iterable[gilda.term.Term]:
    """Get gilda terms for the given namespace."""
    parse_results = get_obograph_by_prefix(prefix)
    if parse_results.graph_document is None:
        return
    for graph in parse_results.graph_document.graphs:
        graph.standardize(prefix=prefix)
        yield from _gilda_from_graph(prefix, graph)


def _gilda_from_graph(prefix: str, graph: Graph) -> Iterable[gilda.term.Term]:
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
