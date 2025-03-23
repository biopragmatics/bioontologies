"""A curated database of upgrades for outdated strings and IRIs appearing in ontologies."""

import csv
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from bioregistry import NormalizedNamableReference, NormalizedNamedReference
from tqdm import tqdm

__all__ = [
    "PATH",
    "Terms",
    "insert",
    "load",
    "upgrade",
    "write",
]

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data.tsv")

Terms = Mapping[str, NormalizedNamableReference]


def upgrade(s: str) -> NormalizedNamedReference | None:
    """Upgrade a string, which is potentially an IRI to a curated CURIE pair."""
    return load().get(s)


@lru_cache(1)
def load() -> Terms:
    """Load the upgrade terms."""
    with PATH.open() as file:
        reader = csv.reader(file, delimiter="\t")
        return {
            term: NormalizedNamableReference(prefix=prefix, identifier=identifier)
            for term, prefix, identifier in reader
        }


def write(terms: Terms) -> None:
    """Write the upgrade terms."""
    with PATH.open("w") as file:
        writer = csv.writer(file, delimiter="\t")
        for term, reference in sorted(terms.items()):
            writer.writerow((term, reference.prefix, reference.identifier))


def insert(term: str, prefix: str, identifier: str, *, name: str | None = None) -> None:
    """Insert a new upgrade term."""
    terms = dict(load())
    existing = terms.get(term)
    reference = NormalizedNamableReference(prefix=prefix, identifier=identifier, name=name)
    if existing:
        if existing != reference:
            tqdm.write(
                f"Conflict for inserting {term} between existing {existing} "
                f"and reference {reference}. Skipping."
            )
        return None
    terms[term] = reference
    write(terms)
    load.cache_clear()


if __name__ == "__main__":
    write(load())  # lints and sorts
