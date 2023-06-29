"""API for grounding relations."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import requests
from tqdm import tqdm

__all__ = [
    "ground_relation",
    "get_normalized_label",
]

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data.json")
URLS = [
    ("ro", "http://purl.obolibrary.org/obo/ro.json"),
    (
        "debio",
        "https://raw.githubusercontent.com/biopragmatics/debio/main/releases/current/debio.json",
    ),
    ("bfo", None),
    ("oboinowl", None),
    ("owl", None),
    ("rdf", None),
    ("rdfs", None),
    ("bspo", None),
    ("iao", None),
    ("omo", None),
]
PREFIX_OBO = "http://purl.obolibrary.org/obo/"
PREFIX_OIO = "http://www.geneontology.org/formats/oboInOwl#"

LABELS = {
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy": "is_defined_by",
    "rdf:type": "type",
    "owl:inverseOf": "inverse_of",
    "skos:exactMatch": "exact_match",
    "rdfs:subClassOf": "is_a",
    "rdfs:subPropertyOf": "subproperty",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "type",
    # FIXME deal with these relations
    "http://purl.obolibrary.org/obo/uberon/core#proximally_connected_to": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#extends_fibers_into": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#channel_for": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#distally_connected_to": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#channels_into": "channels_into",
    "http://purl.obolibrary.org/obo/uberon/core#channels_from": "channels_from",
    "http://purl.obolibrary.org/obo/uberon/core#subdivision_of": "subdivision_of",
    "http://purl.obolibrary.org/obo/uberon/core#protects": "protects",
    "http://purl.obolibrary.org/obo/uberon/core#posteriorly_connected_to": "posteriorly_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#evolved_from": "evolved_from",
    "http://purl.obolibrary.org/obo/uberon/core#anteriorly_connected_to": "anteriorly_connected_to",
}


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("_", "").replace(":", "").lower()


def ground_relation(s: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Ground a string to a RO property."""
    return get_lookups().get(_norm(s), (None, None))


def get_normalized_label(curie_or_uri: str) -> Optional[str]:
    """Get a normalized label."""
    rv = LABELS.get(curie_or_uri)
    if rv:
        return rv
    rv = get_curie_to_norm_name().get(curie_or_uri)
    if rv:
        return rv
    return None


@lru_cache(1)
def get_lookups() -> Mapping[str, Tuple[str, str]]:
    """Get lookups for relation ontology properties."""
    d = {}
    for record in json.loads(PATH.read_text()):
        prefix, identifier, label = record["prefix"], record["identifier"], record["label"]
        d[_norm(label)] = prefix, identifier
        for s in record.get("synonyms", []):
            d[_norm(s)] = prefix, identifier
    return d


def label_norm(s: str) -> str:
    """Normalize a label string."""
    return s.lower().replace(" ", "_")


@lru_cache(1)
def get_curie_to_norm_name() -> Mapping[str, str]:
    """Get a dictionary mapping CURIEs to their normalized names."""
    curie_to_norm_name = {}
    for record in json.loads(PATH.read_text()):
        prefix, identifier, label = record["prefix"], record["identifier"], record["label"]
        curie_to_norm_name[f"{prefix}:{identifier}"] = label_norm(label)
    return curie_to_norm_name


HEADER = ["prefix", "identifier", "label", "synonyms"]


def main():
    """Download and process the relation ontology data."""
    from bioontologies import get_obograph_by_prefix
    from bioontologies.obograph import GraphDocument
    from bioontologies.robot import correct_raw_json

    rows = []
    for source, url in URLS:
        if url is not None:
            res = requests.get(url)
            res.raise_for_status()
            res_json = res.json()
            correct_raw_json(res_json)
            graph_document = GraphDocument.parse_obj(res_json)
            graph = graph_document.guess(source)
        else:
            try:
                results = get_obograph_by_prefix(source)
                graph = results.guess(source)
            except ValueError as e:
                tqdm.write(f"[{source}] error: {e}")
                continue
        for node in tqdm(graph.nodes, desc=source, unit="node"):
            if node.type != "PROPERTY" or not node.name:
                continue
            node.standardize()
            if not node.prefix:
                tqdm.write(f"[{source}] could not parse {node.id}")
                continue
            rows.append(
                (
                    node.prefix,
                    node.identifier,
                    node.name,
                    tuple(sorted(synonym.value for synonym in node.synonyms)),
                )
            )
    rows = sorted(set(rows))
    row_dicts = [{k: v for k, v in zip(HEADER, row) if v} for row in rows]
    PATH.write_text(json.dumps(row_dicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
