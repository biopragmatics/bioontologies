"""Constants."""

from __future__ import annotations

from typing import overload

import bioregistry
import obographs

CANONICAL = {
    "mamo": "http://identifiers.org/mamo",
    "swo": "http://www.ebi.ac.uk/swo/swo.json",
    "ito": "https://identifiers.org/ito:ontology",
    "apollosv": "http://purl.obolibrary.org/obo/apollo_sv.owl",
    "cheminf": "http://semanticchemistry.github.io/semanticchemistry/ontology/cheminf.owl",
    "dideo": "http://purl.obolibrary.org/obo/dideo/release/2022-06-14/dideo.owl",
    "micro": "http://purl.obolibrary.org/obo/MicrO.owl",
    "ogsf": "http://purl.obolibrary.org/obo/ogsf-merged.owl",
    "mfomd": "http://purl.obolibrary.org/obo/MF.owl",
    "one": "http://purl.obolibrary.org/obo/ONE",
    "ons": "https://raw.githubusercontent.com/enpadasi/Ontology-for-Nutritional-Studies/master/ons.owl",
    "ontie": "https://ontology.iedb.org/ontology/ontie.owl",
}

IRI_TO_PREFIX = {v: k for k, v in CANONICAL.items()}
for resource in bioregistry.resources():
    owl_iri = resource.get_download_owl()
    if owl_iri:
        IRI_TO_PREFIX[owl_iri] = resource.prefix


@overload
def guess(self: obographs.GraphDocument, prefix: str) -> obographs.Graph: ...


@overload
def guess(
    self: obographs.StandardizedGraphDocument, prefix: str
) -> obographs.StandardizedGraph: ...


def guess(
    graph_document: obographs.GraphDocument | obographs.StandardizedGraphDocument,
    prefix: str,
) -> obographs.Graph | obographs.StandardizedGraph:
    """Guess the primary graph."""
    if 1 == len(graph_document.graphs):
        return graph_document.graphs[0]
    id_to_graph = {graph.id: graph for graph in graph_document.graphs if graph.id}
    for suffix in ["owl", "obo", "json"]:
        standard_id = f"http://purl.obolibrary.org/obo/{prefix.lower()}.{suffix}"
        if standard_id in id_to_graph:
            return id_to_graph[standard_id]
    if prefix in CANONICAL and CANONICAL[prefix] in id_to_graph:
        return id_to_graph[CANONICAL[prefix]]
    raise ValueError(f"Several graphs in {prefix}: {sorted(id_to_graph)}")
