"""NER utilities."""

from typing import Any, Iterable, Sequence

from tqdm import tqdm

import biosynonyms
import curies
from bioontologies import get_obograph_by_prefix
from bioontologies.obograph import Graph
from biosynonyms.model import DEFAULT_PREDICATE
from curies import vocabulary as v

__all__ = [
    "get_literal_mappings",
    "get_literal_mappings_subset",
]


def get_literal_mappings(prefix: str, **kwargs: Any) -> Iterable[biosynonyms.LiteralMapping]:
    """Get literal mappings for the given namespace.

    :param prefix:
        The prefix of the ontology to load. Will look up the "best" resource
        via the :mod:`bioregistry` and convert with ROBOT.
    :param kwargs:
        Keyword arguments to pass to :func:`bioontologies.get_obograph_by_prefix`
    :yields: Term objects for Gilda

    Example usage:

    .. code-block::

        import bioontologies
        import biosynonyms

        literal_mappings = bioontologies.get_literal_mappings("go")
        grounder = biosynonyms.grounder_from_literal_mappings(literal_mappings)
        scored_matches = grounder.ground("apoptosis")

    Some ontologies don't parse nicely with ROBOT because they have malformed
    entries. To disregard these, you can use the ``check=False`` argument:

    .. code-block::

        import bioontologies
        import gilda

        literal_mappings = bioontologies.get_literal_mappings("vo", check=False)
        grounder = biosynonyms.grounder_from_literal_mappings(literal_mappings)
        scored_matches = grounder.ground("comirna")
    """
    parse_results = get_obograph_by_prefix(prefix, **kwargs)
    if parse_results.graph_document is None:
        return
    for graph in parse_results.graph_document.graphs:
        graph.standardize(prefix=prefix, nodes=True, edges=False)
        yield from literal_mappings_from_graph(prefix, graph)


def literal_mappings_from_graph(prefix: str, graph: Graph) -> Iterable[biosynonyms.LiteralMapping]:
    """Get literal mappings from a given graph."""
    for node in tqdm(graph.nodes, leave=False, unit_scale=True, desc=f"{prefix} to Gilda"):
        if node.reference is None:
            continue
        if node.reference.prefix != prefix:
            # Don't add references from other namespaces
            continue

        reference = curies.NamableReference(
            prefix=prefix,
            identifier=node.reference.identifier,
            name=node.name,
        )

        yield biosynonyms.LiteralMapping(
            reference=reference,
            predicate=v.has_label,
            text=node.name,
            source=prefix,
        )
        for synonym in node.synonyms:
            yield biosynonyms.LiteralMapping(
                reference=reference,
                predicate=curies.Reference(prefix="oboInOwl", identifier=synonym.predicate_raw)
                if synonym.predicate_raw
                else DEFAULT_PREDICATE,
                text=synonym.value,
                source=prefix,
            )


def get_literal_mappings_subset(
    prefix: str, ancestors: curies.Reference | Sequence[curies.Reference], *, check: bool = False, **kwargs
) -> Iterable[biosynonyms.LiteralMapping]:
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
        descendant
        for ancestor in ancestors
        for descendant in nx.ancestors(graph, ancestor)
    }

    return [
        lm
        for lm in get_literal_mappings(prefix, **kwargs)
        if lm.reference in descendants
    ]
