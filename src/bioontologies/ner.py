"""NER utilities."""

import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import curies
import ssslm
from curies import vocabulary as v
from obographs import StandardizedGraph, StandardizedNode
from tqdm import tqdm

from .robot import get_obograph_by_prefix

__all__ = [
    "get_literal_mappings",
    "get_literal_mappings_subset",
    "literal_mappings_from_graph",
]

# TODO upstream into obographs, then remove


warnings.warn(
    "bioontologies.ner is deprecated and will be removed in v0.1.0, use pyobo instead",
    DeprecationWarning,
    stacklevel=2,
)


def get_literal_mappings(
    prefix: str, *, converter: curies.Converter, strict: bool = False, **kwargs: Any
) -> Iterable[ssslm.LiteralMapping]:
    """Get literal mappings for the given namespace.

    :param prefix:
        The prefix of the ontology to load. Will look up the "best" resource
        via the :mod:`bioregistry` and convert with ROBOT.
    :param kwargs:
        Keyword arguments to pass to :func:`bioontologies.get_obograph_by_prefix`
    :yields: literal mappings

    Example usage:

    .. code-block::

        import bioontologies
        import ssslm

        literal_mappings = bioontologies.get_literal_mappings("go")
        grounder = ssslm.make_grounder(literal_mappings)
        scored_matches = grounder.ground("apoptosis")

    Some ontologies don't parse nicely with ROBOT because they have malformed
    entries. To disregard these, you can use the ``check=False`` argument:

    .. code-block::

        import bioontologies
        import ssslm

        literal_mappings = bioontologies.get_literal_mappings("vo", check=False)
        grounder = ssslm.make_grounder(literal_mappings)
        scored_matches = grounder.ground("comirna")
    """
    parse_results = get_obograph_by_prefix(prefix, **kwargs)
    if parse_results.graph_document is None:
        return
    for graph in parse_results.graph_document.graphs:
        for node in graph.nodes:
            st_node = StandardizedNode.from_obograph_raw(node, converter, strict=strict)
            if st_node is None:
                continue
            yield from _lm_from_node(st_node, prefix)


def literal_mappings_from_graph(
    prefix: str, graph: StandardizedGraph
) -> Iterable[ssslm.LiteralMapping]:
    """Get literal mappings from a given graph."""
    for node in tqdm(graph.nodes, leave=False, unit_scale=True, desc=f"{prefix} get synonyms"):
        yield from _lm_from_node(node, prefix)


def _lm_from_node(node: StandardizedNode, prefix: str) -> Iterable[ssslm.LiteralMapping]:
    if node.reference is None:
        return
    if node.reference.prefix != prefix:
        # Don't add references from other namespaces
        return

    reference = curies.NamableReference(
        prefix=prefix,
        identifier=node.reference.identifier,
        name=node.label,
    )

    if node.label is not None:
        yield ssslm.LiteralMapping(
            reference=reference,
            predicate=v.has_label,
            text=node.label,
            source=prefix,
        )
    if node.meta is not None and node.meta.synonyms:
        for synonym in node.meta.synonyms:
            yield ssslm.LiteralMapping(
                reference=reference,
                predicate=synonym.predicate,
                text=synonym.text,
                source=prefix,
                provenance=synonym.xrefs or [],
            )


def get_literal_mappings_subset(
    prefix: str,
    ancestors: curies.Reference | Sequence[curies.Reference],
    *,
    check: bool = False,
    **kwargs: Any,
) -> list[ssslm.LiteralMapping]:
    """Get a subset of literal mappings for terms under the ancestors."""
    if isinstance(ancestors, curies.Reference):
        ancestors = [ancestors]

    import networkx as nx

    parse_results = get_obograph_by_prefix(prefix, check=check, **kwargs)
    obograph = parse_results.squeeze(standardize=True, prefix=prefix)
    graph: nx.DiGraph[curies.Reference] = nx.DiGraph()
    for edge in obograph.edges:
        if (
            edge.subject
            and edge.predicate
            and edge.object
            and edge.predicate.curie == "rdfs:subClassOf"
        ):
            graph.add_edge(edge.subject, edge.object)

    descendants: set[curies.Reference] = {
        descendant for ancestor in ancestors for descendant in nx.ancestors(graph, ancestor)
    }

    return [lm for lm in get_literal_mappings(prefix, **kwargs) if lm.reference in descendants]
