"""Data structures for representing OBO Graphs.

.. seealso:: https://github.com/geneontology/obographs
"""

from collections import defaultdict
from operator import attrgetter
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union

from bioregistry import normalize_curie
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Literal

__all__ = [
    "Property",
    "Definition",
    "Xref",
    "Synonym",
    "Meta",
    "Edge",
    "Node",
    "Graph",
    "GraphDocument",
]

OBO_URI_PREFIX = "http://purl.obolibrary.org/obo/"
IDENTIFIERS_HTTP_PREFIX = "http://identifiers.org/"
IDENTIFIERS_HTTPS_PREFIX = "https://identifiers.org/"


class Property(BaseModel):
    """Represent a property inside a metadata element."""

    pred: str
    val: str


class Definition(BaseModel):
    """Represents a definition for a node."""

    val: str
    # Just a list of CURIEs/IRIs
    xrefs: Optional[List[str]]


class Xref(BaseModel):
    """Represents a cross-reference."""

    val: str


class Synonym(BaseModel):
    """Represents a synonym inside an object meta."""

    pred: str
    val: str
    # Just a list of CURIEs/IRIs
    xrefs: List[str]
    synonymType: Optional[str]  # noqa:N815


class Meta(BaseModel):
    """Represents the metadata about a node or ontology."""

    definition: Optional[Definition]
    subsets: Optional[List[str]]
    xrefs: Optional[List[Xref]]
    synonyms: Optional[List[Synonym]]
    comments: Optional[List]
    version: Optional[str]
    basicPropertyValues: Optional[List[Property]]  # noqa:N815
    deprecated: bool = False


class Edge(BaseModel):
    """Represents an edge in an OBO Graph."""

    sub: str
    pred: str
    obj: str
    meta: Optional[Meta]

    def as_tuple(self) -> Tuple[str, str, str]:
        """Get the edge as a tuple."""
        return self.sub, self.pred, self.obj


class Node(BaseModel):
    """Represents a node in an OBO Graph."""

    id: str
    lbl: Optional[str]
    meta: Optional[Meta]
    type: Optional[Literal["CLASS", "PROPERTY", "INDIVIDUAL"]]
    alternative_ids: Optional[List[str]]

    @property
    def deprecated(self) -> bool:
        """Get if the node is deprecated."""
        if self.meta is None:
            return False
        return self.meta.deprecated

    @property
    def synonyms(self) -> List[Synonym]:
        """Get the synonyms for the node."""
        if self.meta and self.meta.synonyms:
            return self.meta.synonyms
        return []

    @property
    def xrefs(self) -> List[Xref]:
        """Get the xrefs for the node."""
        if self.meta and self.meta.xrefs:
            return self.meta.xrefs
        return []

    @property
    def replaced_by(self) -> Optional[str]:
        """Get the identifier that this node was replaced by."""
        if not self.meta:
            return None
        preds = ["http://purl.obolibrary.org/obo/IAO_0100001", "IAO:0100001", "iao:0100001"]
        for prop in self.meta.basicPropertyValues or []:
            if any(prop.pred == pred for pred in preds):
                return prop.val
        return None


class Graph(BaseModel):
    """A graph corresponds to an ontology."""

    id: str
    meta: Meta
    nodes: List[Node]
    edges: List[Edge]
    equivalentNodesSets: Any  # noqa:N815
    logicalDefinitionAxioms: Any  # noqa:N815
    domainRangeAxioms: Any  # noqa:N815
    propertyChainAxioms: Any  # noqa:N815

    @property
    def roots(self) -> List[str]:
        """Get the ontology root terms."""
        return self._get_properties(
            [
                "http://purl.obolibrary.org/obo/IAO_0000700",
                "IAO:0000700",
            ]
        )

    @property
    def license(self) -> Optional[str]:
        """Get the license of the ontology."""
        return self._get_property("http://purl.org/dc/terms/license")

    @property
    def title(self) -> Optional[str]:
        """Get the title of the ontology."""
        return self._get_property("http://purl.org/dc/elements/1.1/title")

    @property
    def description(self) -> Optional[str]:
        """Get the license of the ontology."""
        return self._get_property("http://purl.org/dc/elements/1.1/description")

    @property
    def version_iri(self) -> Optional[str]:
        """Get the version of the ontology."""
        return self.meta.version

    @property
    def version(self) -> Optional[str]:
        """Get the version of the ontology."""
        return self._get_property("http://www.w3.org/2002/07/owl#versionInfo")

    @property
    def default_namespace(self) -> Optional[str]:
        """Get the version of the ontology."""
        return self._get_property("http://www.geneontology.org/formats/oboInOwl#default-namespace")

    def _get_property(self, pred: Union[str, List[str]]) -> Optional[str]:
        p = self._get_properties(pred)
        return p[0] if p else None

    def _get_properties(self, pred: Union[str, List[str]]) -> List[str]:
        if isinstance(pred, str):
            pred = [pred]
        return [
            prop.val
            for prop in self.meta.basicPropertyValues or []
            if any(prop.pred == p for p in pred)
        ]

    def standardize(self, keep_invalid: bool = False, use_tqdm: bool = True) -> "Graph":
        """Standardize the OBO graph.

        :param keep_invalid: Should CURIEs/IRIs that aren't handled
            by the Bioregistry be kept? Defaults to false.
        :param use_tqdm:
            Should a progress bar be used?
        :returns: This OBO graph, modified in place as follows:

            1. Convert IRIs to CURIEs (in many places) using :mod:`bioregistry`
            2. Add alternative identifiers to :class:`Node` objects
        """
        # Convert URIs to CURIEs
        for node in tqdm(
            self.nodes, desc="standardizing nodes", unit_scale=True, disable=not use_tqdm
        ):
            if node.id.startswith(OBO_URI_PREFIX):
                node.id = _clean_uri(node.id, keep_invalid=True)  # type:ignore
            if node.meta:
                for prop in node.meta.basicPropertyValues or []:
                    prop.pred = _clean_uri(prop.pred, keep_invalid=True)  # type:ignore
                    prop.val = _clean_uri(prop.val, keep_invalid=True)  # type:ignore

                for synonym in node.meta.synonyms or []:
                    synonym.pred = _clean_uri(synonym.pred, keep_invalid=True)  # type:ignore
                    if synonym.synonymType:
                        synonym.synonymType = _clean_uri(
                            synonym.synonymType, keep_invalid=True
                        )  # type:ignore

                # Remove self-xrefs, duplicate xrefs
                xrefs: List[Xref] = []
                xrefs_vals: Set[str] = set()
                for xref in node.meta.xrefs or []:
                    if xref.val == node.id:
                        continue
                    new_xref_val = _clean_uri(xref.val, keep_invalid=keep_invalid)
                    if new_xref_val is None:
                        continue
                    xref.val = new_xref_val
                    if xref.val == node.id or xref.val in xrefs_vals:
                        continue
                    xrefs_vals.add(xref.val)
                    xrefs.append(xref)
                node.meta.xrefs = sorted(xrefs, key=attrgetter("val"))

        for edge in tqdm(
            self.edges, desc="standardizing edges", unit_scale=True, disable=not use_tqdm
        ):
            edge.sub = _clean_uri(edge.sub, keep_invalid=True)
            edge.pred = _clean_uri(edge.pred, keep_invalid=True)
            edge.obj = _clean_uri(edge.obj, keep_invalid=True)

        # TODO add xrefs from definition into node if the are "real" CURIEs

        # Add alt ids
        alt_ids = self.get_alternative_ids()
        for node in self.nodes:
            node.alternative_ids = alt_ids.get(node.id, [])

        return self

    def get_alternative_ids(self) -> Mapping[str, List[str]]:
        """Get a mapping of primary identifiers to secondary identifiers."""
        rv = defaultdict(set)
        for node in self.nodes:
            if node.replaced_by:
                rv[node.replaced_by].add(node.id)
        return {k: sorted(v) for k, v in rv.items()}

    def nodes_from(self, prefix: str) -> Iterable[Node]:
        """Iterate non-deprecated nodes whose identifiers start with the given prefix."""
        for node in self.nodes:
            if node.deprecated:
                continue
            if not node.id.startswith(prefix):
                continue
            yield node


def _clean_uri(s: str, *, keep_invalid: bool) -> Optional[str]:
    s = _compress_uri(s)
    n = normalize_curie(s)
    if n:
        return n
    elif keep_invalid:
        return s
    else:
        return None


IS_A_STRINGS = {
    "is_a",
    "isa",
    "type",  # used for instance to class
}


def _compress_uri(s: str) -> str:
    if s in IS_A_STRINGS:
        return "rdfs:subClassOf"
    if s == "subPropertyOf":
        return "rdfs:subPropertyOf"
    if s.startswith(OBO_URI_PREFIX):
        s = s[len(OBO_URI_PREFIX) :]
        if "_" in s and s.split("_")[1].isnumeric():  # best guess that it's an identifier
            s = s.replace("_", ":", 1)
    for identifiers_prefix in (IDENTIFIERS_HTTP_PREFIX, IDENTIFIERS_HTTPS_PREFIX):
        if s.startswith(identifiers_prefix):
            s = s[len(identifiers_prefix) :]
            if ":" in s:
                return s
            else:
                return s.replace("/", ":", 1)
    for uri_prefix, prefix in [
        ("http://www.geneontology.org/formats/oboInOwl#", "oboinowl"),
        ("http://www.w3.org/2002/07/owl#", "owl"),
    ]:
        if s.startswith(uri_prefix):
            s = s[len(uri_prefix) :]
            s = f"{prefix}:{s}"
    return s


class GraphDocument(BaseModel):
    """Represents a list of OBO graphs."""

    graphs: List[Graph]
