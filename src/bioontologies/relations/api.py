import requests
from pathlib import Path
import csv
import json
from typing import Union, Tuple, Iterable
from collections import defaultdict
from functools import lru_cache

__all__ = [
    "ground_relation",
]

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data.json")
URL = "http://purl.obolibrary.org/obo/ro.json"
PREFIX_OBO = "http://purl.obolibrary.org/obo/"
PREFIX_OIO = "http://www.geneontology.org/formats/oboInOwl#"


def _norm(s):
    return s.replace(" ", "").replace("_", "").replace(":", "").lower()


def ground_relation(s) -> Union[Tuple[str, str], Tuple[None, None]]:
    return get_lookups().get(_norm(s), (None, None))


@lru_cache(1)
def get_lookups():
    d = {}
    for record in json.loads(PATH.read_text()):
        p, i, l = record["prefix"], record["identifier"], record["label"]
        d[_norm(l)] = p, i
        for s in record.get("synonyms", []):
            d[_norm(s)] = p, i
    return d


def main():
    nodes = requests.get(URL).json()["graphs"][0]["nodes"]
    rows = []
    for node in nodes:
        if node.get("type") != "PROPERTY":
            continue
        label = node.get("lbl")
        if not label:
            continue
        iri = node["id"]
        if iri.startswith(PREFIX_OIO):
            prefix, identifier = "oboinowl", iri[len(PREFIX_OIO):]
        else:
            try:
                prefix, identifier = iri[len(PREFIX_OBO):].split("_", 1)
            except ValueError:
                print(iri, label)
                continue
        if prefix in {"valid"}:
            continue
        rows.append(dict(
            prefix=prefix.lower(), identifier=identifier, label=label,
            synonyms=[x["val"] for x in node.get("meta", {}).get("synonyms", [])],
        ))
    PATH.write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
