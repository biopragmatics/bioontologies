"""A curated database of upgrades for outdated strings and IRIs appearing in ontologies."""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from curies import ReferenceTuple

__all__ = [
    "upgrade",
    "load",
    "write",
    "insert",
    "PATH",
    "Terms",
]

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data.tsv")

Terms = Mapping[str, ReferenceTuple]


def upgrade(s: str) -> Optional[ReferenceTuple]:
    """Upgrade a string, which is potentially an IRI to a curated CURIE pair."""
    return load().get(s)


@lru_cache(1)
def load() -> Terms:
    """Load the upgrade terms."""
    with PATH.open() as file:
        reader = csv.reader(file, delimiter="\t")
        return {term: ReferenceTuple(prefix, identifier) for term, prefix, identifier in reader}


def write(terms: Terms) -> None:
    """Write the upgrade terms."""
    with PATH.open("w") as file:
        writer = csv.writer(file, delimiter="\t")
        for term, (prefix, identifier) in sorted(terms.items()):
            writer.writerow((term, prefix, identifier))


def insert(term: str, prefix: str, identifier: str) -> None:
    """Insert a new upgrade term."""
    terms = dict(load())
    existing = terms.get(term)
    reference_tuple = ReferenceTuple(prefix, identifier)
    if existing:
        if existing == reference_tuple:
            return None
        else:
            raise KeyError
    terms[term] = reference_tuple
    write(terms)
    load.cache_clear()


if __name__ == "__main__":
    write(load())  # lints and sorts
