"""Data structures for representing OBO Graphs.

.. seealso:: https://github.com/geneontology/obographs
"""

import logging
from collections import defaultdict
from operator import attrgetter
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union, cast

import bioregistry
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

logger = logging.getLogger(__name__)

OBO_URI_PREFIX = "http://purl.obolibrary.org/obo/"
OBO_URI_PREFIX_LEN = len(OBO_URI_PREFIX)
IDENTIFIERS_HTTP_PREFIX = "http://identifiers.org/"
IDENTIFIERS_HTTPS_PREFIX = "https://identifiers.org/"

MaybeCURIE = Union[Tuple[str, str], Tuple[None, None]]


class StandardizeMixin:
    """A mixin for classes representing standardizable data."""

    def standardize(self):
        """Standardize the data in this class."""
        raise NotImplementedError

    def raise_on_unstandardized(self):
        """Raise an exception if standarization has not occurred."""
        if not self.standardized:
            raise ValueError


class Property(BaseModel, StandardizeMixin):
    """Represent a property inside a metadata element."""

    pred: str
    val: str
    standardized: bool = False

    # Standardizable
    pred_prefix: Optional[str]
    pred_identifier: Optional[str]
    val_prefix: Optional[str]
    val_identifier: Optional[str]

    @property
    def pred_curie(self) -> str:
        """Get the predicate's CURIE or error if unparsable."""
        if self.pred_prefix is None or self.pred_identifier is None:
            raise
        return bioregistry.curie_to_str(self.pred_prefix, self.pred_identifier)

    @property
    def val_curie(self) -> str:
        """Get the value's CURIE or error if unparsable."""
        if self.val_prefix is None or self.val_identifier is None:
            raise
        return bioregistry.curie_to_str(self.val_prefix, self.val_identifier)

    def standardize(self):
        """Standardize this property."""
        self.pred_prefix, self.pred_identifier = _help_ground(self.pred)
        self.val_prefix, self.val_identifier = _help_ground(self.val)
        self.standardized = True


class Definition(BaseModel):
    """Represents a definition for a node."""

    val: str
    # Just a list of CURIEs/IRIs
    xrefs: Optional[List[str]]


class Xref(BaseModel, StandardizeMixin):
    """Represents a cross-reference."""

    val: str
    prefix: Optional[str]
    identifier: Optional[str]
    standardized: bool = False

    @property
    def curie(self) -> str:
        """Get the xref's CURIE."""
        if self.prefix is None or self.identifier is None:
            raise ValueError(f"can't parse xref: {self.val}")
        return bioregistry.curie_to_str(self.prefix, self.identifier)

    def standardize(self) -> None:
        """Standardize the xref."""
        if self.val.startswith("http") or self.val.startswith("https"):
            self.prefix, self.identifier = _help_ground(self.val)
        else:
            self.prefix, self.identifier = bioregistry.parse_curie(self.val)
        self.standardized = True


class Synonym(BaseModel, StandardizeMixin):
    """Represents a synonym inside an object meta."""

    pred: str
    val: str
    # Just a list of CURIEs/IRIs
    xrefs: List[str]
    synonymType: Optional[str]  # noqa:N815
    standardized: bool = False

    def standardize(self):
        """Standardize the synoynm."""
        self.pred = _clean_uri(self.pred, keep_invalid=True)  # type:ignore
        if self.synonymType:
            self.synonymType = _clean_uri(self.synonymType, keep_invalid=True)  # type:ignore
        self.standardized = True


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

    def parse_curies(self) -> Tuple[MaybeCURIE, MaybeCURIE, MaybeCURIE]:
        """Get parsed CURIEs for this relationship."""
        return (
            bioregistry.parse_curie(self.sub),
            bioregistry.parse_curie(self.pred),
            bioregistry.parse_curie(self.obj),
        )

    def standardize(self):
        """Standardize the edge."""
        self.sub = _clean_uri(self.sub, keep_invalid=True)
        self.pred = _clean_uri(self.pred, keep_invalid=True)
        self.obj = _clean_uri(self.obj, keep_invalid=True)


def _help_ground(uri: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Ground the node to a standard prefix and luid based on its id (URI)."""
    prefix, identifier = _compress_uri(uri)
    if prefix is None:
        return None, None
    resource = bioregistry.get_resource(prefix)
    if resource is not None:
        return resource.prefix, identifier
    return None, None


def _help_get_properties(self, pred: Union[str, List[str]]) -> List[str]:
    if not self.meta:
        return []
    if isinstance(pred, str):
        pred = [pred]
    # print(self.meta.basicPropertyValues, pred)
    return [
        bioregistry.normalize_curie(prop.val_curie) if prop.val_prefix else prop.val
        for prop in self.meta.basicPropertyValues or []
        if any(prop.pred == p for p in pred)
    ]


class Node(BaseModel, StandardizeMixin):
    """Represents a node in an OBO Graph."""

    id: str
    lbl: Optional[str]
    meta: Optional[Meta]
    type: Optional[Literal["CLASS", "PROPERTY", "INDIVIDUAL"]]
    prefix: Optional[str]
    luid: Optional[str]
    standardized: bool = False

    def standardize(self) -> None:
        """Ground the node to a standard prefix and luid based on its id (URI)."""
        self.prefix, self.luid = _help_ground(self.id)

        if self.meta:
            for prop in self.meta.basicPropertyValues or []:
                prop.standardize()

            for synonym in self.meta.synonyms or []:
                synonym.standardize()

            if self.meta.xrefs:
                xrefs: List[Xref] = []
                seen: Set[Tuple[str, str]] = set()
                for xref in self.meta.xrefs:
                    xref.standardize()
                    if xref.prefix is None or xref.identifier is None:
                        continue
                    if xref.prefix == self.prefix and xref.identifier == self.luid:
                        continue
                    pair = xref.prefix, xref.identifier
                    if pair in seen:
                        continue
                    seen.add(pair)
                    xrefs.append(xref)
                self.meta.xrefs = sorted(xrefs, key=attrgetter("prefix"))
        # tqdm.write("\t".join((self.curie, *(x.curie for x in self.xrefs))))
        # TODO add xrefs from definition into node if the are "real" CURIEs
        self.standardized = True

    @property
    def curie(self) -> str:
        """Get the CURIE string representing this node or error if not normalized."""
        if self.prefix is None or self.luid is None:
            raise ValueError("can not give curie for node")
        return bioregistry.curie_to_str(self.prefix, self.luid)

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
        preds = ["http://purl.obolibrary.org/obo/IAO_0100001", "IAO:0100001", "iao:0100001"]
        rv = self._get_property(preds)
        if not rv:
            return None
        return bioregistry.normalize_curie(rv)

    @property
    def alternative_ids(self) -> List[str]:
        """Get the alernative identifiers for this node."""
        preds = [
            "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId",
            "oboinowl:hasAlternativeId",
            "oboInOwl:hasAlternativeId",
        ]
        return [bioregistry.normalize_curie(curie) for curie in self._get_properties(preds)]

    @property
    def namespace(self) -> Optional[str]:
        """Get the OBO namespace."""
        preds = [
            "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
            "oboinowl:hasOBONamespace",
        ]
        return self._get_property(preds)

    @property
    def created_by(self) -> Optional[str]:
        """Get the creator of the node."""
        preds = ["http://www.geneontology.org/formats/oboInOwl#created_by", "oboinowl:created_by"]
        return self._get_property(preds)

    @property
    def creation_date(self) -> Optional[str]:
        """Get the creation date of the node."""
        preds = [
            "http://www.geneontology.org/formats/oboInOwl#creation_date",
            "oboinowl:creation_date",
        ]
        return self._get_property(preds)

    @property
    def definition(self) -> Optional[str]:
        """Get the definition of the node."""
        if self.meta and self.meta.definition:
            return self.meta.definition.val
        return None

    def _get_property(self, pred: Union[str, List[str]]) -> Optional[str]:
        p = self._get_properties(pred)
        return p[0] if p else None

    def _get_properties(self, pred: Union[str, List[str]]) -> List[str]:
        return _help_get_properties(self, pred)

    def parse_curie(self) -> MaybeCURIE:
        """Parse the identifier into a pair, assuming it's a CURIE."""
        return bioregistry.parse_curie(self.id)


class Graph(BaseModel, StandardizeMixin):
    """A graph corresponds to an ontology."""

    id: str
    meta: Meta
    nodes: List[Node]
    edges: List[Edge]
    equivalentNodesSets: Any  # noqa:N815
    logicalDefinitionAxioms: Any  # noqa:N815
    domainRangeAxioms: Any  # noqa:N815
    propertyChainAxioms: Any  # noqa:N815
    standardized: bool = False

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
        rv = self._get_property("http://www.w3.org/2002/07/owl#versionInfo")
        if rv:
            return rv
        version_iri = self.version_iri
        if not version_iri:
            return None
        "http://purl.obolibrary.org/obo/mondo/releases/2022-08-01/mondo.owl"
        if version_iri.startswith(OBO_URI_PREFIX):
            # the last part is prefix.owl, the penultimate part should be th version
            return version_iri.split("/")[-2]
        return None

    @property
    def default_namespace(self) -> Optional[str]:
        """Get the version of the ontology."""
        return self._get_property("http://www.geneontology.org/formats/oboInOwl#default-namespace")

    def _get_property(self, pred: Union[str, List[str]]) -> Optional[str]:
        p = self._get_properties(pred)
        return p[0] if p else None

    def _get_properties(self, pred: Union[str, List[str]]) -> List[str]:
        return _help_get_properties(self, pred)

    def standardize(
        self,
        keep_invalid: bool = False,
        use_tqdm: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        prefix: Optional[str] = None,
    ) -> "Graph":
        """Standardize the OBO graph.

        :param keep_invalid: Should CURIEs/IRIs that aren't handled
            by the Bioregistry be kept? Defaults to false.
        :param use_tqdm:
            Should a progress bar be used?
        :param tqdm_kwargs:
            Arguments to pass to tqdm if used
        :param prefix:
            The prefix this graph came from (used for logging purposes)
        :returns: This OBO graph, modified in place as follows:

            1. Convert IRIs to CURIEs (in many places) using :mod:`bioregistry`
            2. Add alternative identifiers to :class:`Node` objects
        """
        self.standardized = True

        _node_tqdm_kwargs = dict(
            desc="standardizing nodes" if not prefix else f"[{prefix}] standardizing nodes",
            unit_scale=True,
            disable=not use_tqdm,
        )
        if tqdm_kwargs:
            _node_tqdm_kwargs.update(tqdm_kwargs)
        for node in tqdm(self.nodes, **_node_tqdm_kwargs):
            node.standardize()

        _edge_tqdm_kwargs = dict(
            desc="standardizing edges" if not prefix else f"[{prefix}] standardizing edges",
            unit_scale=True,
            disable=not use_tqdm,
        )
        if tqdm_kwargs:
            _edge_tqdm_kwargs.update(tqdm_kwargs)
        for edge in tqdm(self.edges, **_edge_tqdm_kwargs):
            edge.standardize()

        return self

    def get_alternative_ids(self) -> Mapping[str, List[str]]:
        """Get a mapping of primary identifiers to secondary identifiers."""
        rv = defaultdict(set)
        for node in self.nodes:
            if node.replaced_by:
                rv[node.replaced_by].add(node.id)
            for x in node.alternative_ids:
                rv[x].add(node.id)
        return {k: sorted(v) for k, v in rv.items()}

    def nodes_from(self, prefix: str) -> Iterable[Node]:
        """Iterate non-deprecated nodes whose identifiers start with the given prefix."""
        self.raise_on_unstandardized()
        for node in self.nodes:
            if node.deprecated:
                continue
            if not node.prefix == prefix:
                continue
            yield node

    def get_incoming_xrefs(self, prefix: str) -> Mapping[str, str]:
        """Get incoming xrefs.

        :param prefix: An external prefix.
        :returns:
            A dictionary of external local unique identifiers
            to local unique identifiers in this ontology
        """
        xrefs = {}
        for node in self.nodes:
            for xref in node.xrefs:
                xref_prefix, xref_identifier = bioregistry.parse_curie(xref.val)
                if xref_prefix != prefix:
                    continue
                xrefs[xref_identifier] = node.id
        return xrefs


def _clean_uri(s: str, *, keep_invalid: bool, use_preferred: bool = False) -> Optional[str]:
    prefix, identifier = _compress_uri(s)
    if prefix is None:
        if keep_invalid:
            return s
        else:
            return None

    resource = bioregistry.get_resource(prefix)
    if resource is None:
        if keep_invalid:
            return s
        else:
            return None

    return resource.get_curie(identifier, use_preferred=use_preferred)


IS_A_STRINGS = {
    "is_a",
    "isa",
}


def _compress_uri(s: str) -> Union[Tuple[str, str], Tuple[None, str]]:
    if s.startswith(OBO_URI_PREFIX):
        s = s[OBO_URI_PREFIX_LEN:]
        if s.startswith("APOLLO_SV"):  # those monsters put an underscore in their prefix...
            return "apollosv", s[10:]  # hard-coded length of APOLLO_SV_
        for delimiter in [
            "#",  # local property like in chebi#... This needs to be first priority!
            "_",  # best guess that it's an identifier
            "/",  # local property like in chebi/charge
        ]:
            if delimiter in s:
                return cast(Tuple[str, str], s.split(delimiter, 1))
        return None, s
    if s in IS_A_STRINGS:
        return "rdfs", "subClassOf"
    if s == "subPropertyOf":
        return "rdfs", "subPropertyOf"
    if s == "type":  # instance of
        return "rdf", "type"
    for identifiers_prefix in (IDENTIFIERS_HTTP_PREFIX, IDENTIFIERS_HTTPS_PREFIX):
        if s.startswith(identifiers_prefix):
            s = s[len(identifiers_prefix) :]
            if ":" in s:
                return cast(Tuple[str, str], s.split(":", 1))
            else:
                return cast(Tuple[str, str], s.split("/", 1))
    for uri_prefix, prefix in [
        ("http://www.geneontology.org/formats/oboInOwl#", "oboinowl"),
        ("http://www.w3.org/2002/07/owl#", "owl"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs"),
    ]:
        if s.startswith(uri_prefix):
            return prefix, s[len(uri_prefix) :]

    # couldn't parse anything...
    return None, s


class GraphDocument(BaseModel):
    """Represents a list of OBO graphs."""

    graphs: List[Graph]
