"""NER utilities."""

from collections.abc import Iterable, Sequence
from typing import Any

import curies
import ssslm
from curies import vocabulary as v
from tqdm import tqdm

from bioontologies.obograph import Graph

from .robot import get_obograph_by_prefix

__all__ = [
    "get_literal_mappings",
    "get_literal_mappings_subset",
]


def get_literal_mappings(prefix: str, **kwargs: Any) -> Iterable[ssslm.LiteralMapping]:
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
        graph.standardize(prefix=prefix, nodes=True, edges=False)
        yield from literal_mappings_from_graph(prefix, graph)


def literal_mappings_from_graph(
    prefix: str,
    graph: Graph,
    *,
    reference_cls: type[curies.NamableReference] = curies.NamableReference,
) -> Iterable[ssslm.LiteralMapping]:
    """Get literal mappings from a given graph."""
    label_predicate = reference_cls(
        prefix=v.has_label.prefix, identifier=v.has_label.identifier, name=v.has_label.name
    )
    for node in tqdm(graph.nodes, leave=False, unit_scale=True, desc=f"{prefix} get synonyms"):
        if node.reference is None:
            continue
        if node.reference.prefix != prefix:
            # Don't add references from other namespaces
            continue

        if node.name:
            node_name = node.name.strip() or None
        else:
            node_name = None

        reference = reference_cls(
            prefix=prefix,
            identifier=node.reference.identifier,
            name=node_name,
        )

        if node_name:
            yield ssslm.LiteralMapping(
                reference=reference, predicate=label_predicate, text=node_name, source=prefix
            )
        for synonym in node.synonyms:
            if text := synonym.value.strip():
                yield ssslm.LiteralMapping(
                    reference=reference,
                    predicate=reference_cls(prefix="oboInOwl", identifier=synonym.predicate_raw),
                    text=text,
                    source=prefix,
                )


def get_literal_mappings_subset(
    prefix: str,
    ancestors: curies.Reference | Sequence[curies.Reference],
    *,
    check: bool = False,
    **kwargs,
) -> list[ssslm.LiteralMapping]:
    """Get a subset of literal mappings for terms under the ancestors."""
    if isinstance(ancestors, curies.Reference):
        ancestors = [ancestors]

    import networkx as nx

    parse_results = get_obograph_by_prefix(prefix, check=check, **kwargs)
    obograph = parse_results.squeeze().standardize(prefix=prefix)
    graph = nx.DiGraph()
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
