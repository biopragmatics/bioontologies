"""A curated database of upgrades for outdated strings and IRIs appearing in ontologies."""

import csv
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from bioregistry import NormalizedNamableReference

__all__ = [
    "PATH",
    "Terms",
    "insert",
    "load",
]

warnings.warn(
    "bioontologies.upgrade is deprecated and will be removed in v0.1.0, "
    "use curies-processing instead",
    DeprecationWarning,
    stacklevel=2,
)

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data.tsv")

Terms = Mapping[str, NormalizedNamableReference]


def upgrade(s: str) -> NormalizedNamableReference | None:
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
