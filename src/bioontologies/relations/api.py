"""API for grounding relations."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import requests
from tqdm import tqdm

__all__ = [
    "ground_relation",
]

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data.json")
URLS = [
    # ("bfo", "http://purl.obolibrary.org/obo/bfo.json"),
    ("ro", "http://purl.obolibrary.org/obo/ro.json"),
]
PREFIX_OBO = "http://purl.obolibrary.org/obo/"
PREFIX_OIO = "http://www.geneontology.org/formats/oboInOwl#"


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("_", "").replace(":", "").lower()


def ground_relation(s: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Ground a string to a RO property."""
    return get_lookups().get(_norm(s), (None, None))


@lru_cache(1)
def get_lookups():
    """Get lookups for relation ontology properties."""
    d = {}
    for record in json.loads(PATH.read_text()):
        prefix, identifier, label = record["prefix"], record["identifier"], record["label"]
        d[_norm(label)] = prefix, identifier
        for s in record.get("synonyms", []):
            d[_norm(s)] = prefix, identifier
    return d


HEADER = ["prefix", "identifier", "label", "synonyms", "source"]


def main():
    """Download and process the relation ontology data."""
    rows = []
    for source, url in URLS:
        res = requests.get(url)
        res.raise_for_status()
        nodes = res.json()["graphs"][0]["nodes"]
        for node in tqdm(nodes, desc=source, unit="node", unit_scale=True):
            if node.get("type") != "PROPERTY":
                continue
            label = node.get("lbl")
            if not label:
                continue
            iri = node["id"]
            if iri.startswith(PREFIX_OIO):
                prefix, identifier = "oboinowl", iri[len(PREFIX_OIO) :]
            else:
                try:
                    prefix, identifier = iri[len(PREFIX_OBO) :].split("_", 1)
                except ValueError:
                    tqdm.write(f"error in {iri} - {label}")
                    continue
            if prefix in {"valid"}:
                continue
            rows.append(
                (
                    prefix.lower(),
                    identifier,
                    label,
                    tuple(sorted(x["val"] for x in node.get("meta", {}).get("synonyms", []))),
                    source,
                )
            )

    row_dicts = [dict(zip(HEADER, row)) for row in rows]
    PATH.write_text(json.dumps(row_dicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
