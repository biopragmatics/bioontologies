"""Data structures for representing OBO Graphs.

.. seealso:: https://github.com/geneontology/obographs
"""

import itertools as itt
import logging
from collections import defaultdict
from operator import attrgetter
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union

from bioregistry import curie_to_str, get_default_converter, manager
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from typing_extensions import Literal

from .relations import ground_relation

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
PROVENANCE_PREFIXES = {"pubmed", "pmc", "doi", "arxiv", "biorxiv"}

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
        return curie_to_str(self.pred_prefix, self.pred_identifier)

    @property
    def val_curie(self) -> str:
        """Get the value's CURIE or error if unparsable."""
        if self.val_prefix is None or self.val_identifier is None:
            raise
        return curie_to_str(self.val_prefix, self.val_identifier)

    def standardize(self):
        """Standardize this property."""
        self.val = self.val.replace("\n", " ")
        self.pred_prefix, self.pred_identifier = _parse_uri_or_curie_or_str(self.pred)
        self.val_prefix, self.val_identifier = _parse_uri_or_curie_or_str(self.val)
        self.standardized = True


class Definition(BaseModel):
    """Represents a definition for a node."""

    val: str
    # Just a list of CURIEs/IRIs
    xrefs: Optional[List[str]]
    standardized: bool = False

    def standardize(self) -> None:
        """Standardize the xref."""
        if self.xrefs:
            curies = [_clean_uri_or_curie_or_str(xref, keep_invalid=False) for xref in self.xrefs]
            self.xrefs = [curie for curie in curies if curie]
        self.standardized = True


class Xref(BaseModel, StandardizeMixin):
    """Represents a cross-reference."""

    val: str
    # TODO ask the obo graph people to update the data model and include xref types
    pred: str = Field(default="oboinowl:hasDbXref")
    prefix: Optional[str]
    identifier: Optional[str]
    standardized: bool = False

    @property
    def curie(self) -> str:
        """Get the xref's CURIE."""
        if self.prefix is None or self.identifier is None:
            raise ValueError(f"can't parse xref: {self.val}")
        return curie_to_str(self.prefix, self.identifier)

    def standardize(self) -> None:
        """Standardize the xref."""
        self.prefix, self.identifier = _parse_uri_or_curie_or_str(self.val)
        self.standardized = True


class Synonym(BaseModel, StandardizeMixin):
    """Represents a synonym inside an object meta."""

    pred: str
    val: str
    synonymType: Optional[str]  # noqa:N815
    standardized: bool = False
    # Just a list of CURIEs/IRIs
    xrefs: List[str] = Field(default_factory=list)

    def standardize(self):
        """Standardize the synoynm."""
        self.pred = _clean_uri_or_curie_or_str(self.pred, keep_invalid=True)  # type:ignore
        if self.synonymType:
            self.synonymType = _clean_uri_or_curie_or_str(
                self.synonymType, keep_invalid=True
            )  # type:ignore
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
            _parse_uri_or_curie_or_str(self.sub),
            _parse_uri_or_curie_or_str(self.pred),
            _parse_uri_or_curie_or_str(self.obj),
        )

    def standardize(self):
        """Standardize the edge."""
        self.sub = _clean_uri_or_curie_or_str(self.sub, keep_invalid=True)
        self.pred = _clean_uri_or_curie_or_str(self.pred, keep_invalid=True, debug=True)
        self.obj = _clean_uri_or_curie_or_str(self.obj, keep_invalid=True)


def _help_get_properties(self, pred: Union[str, List[str]]) -> List[str]:
    if not self.meta:
        return []
    if isinstance(pred, str):
        pred = [pred]
    # print(self.meta.basicPropertyValues, pred)
    return [
        manager.normalize_curie(prop.val_curie) if prop.val_prefix else prop.val
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
        self.prefix, self.luid = _parse_uri_or_curie_or_str(self.id)

        if self.meta:
            for prop in self.meta.basicPropertyValues or []:
                prop.standardize()

            for synonym in self.meta.synonyms or []:
                synonym.standardize()

            if self.meta.definition:
                self.meta.definition.standardize()

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

        self.standardized = True

    @property
    def curie(self) -> str:
        """Get the CURIE string representing this node or error if not normalized."""
        if self.prefix is None or self.luid is None:
            raise ValueError(f"can not give curie for node {self.id}")
        return curie_to_str(self.prefix, self.luid)

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
    def properties(self) -> List[Property]:
        """Get the properties for this node."""
        if not self.meta or self.meta.basicPropertyValues is None:
            return []
        # TODO filter out ones grabbed by other getters
        return self.meta.basicPropertyValues

    @property
    def replaced_by(self) -> Optional[str]:
        """Get the identifier that this node was replaced by."""
        preds = ["http://purl.obolibrary.org/obo/IAO_0100001", "IAO:0100001", "iao:0100001"]
        rv = self._get_property(preds)
        if not rv:
            return None
        return manager.normalize_curie(rv)

    @property
    def alternative_ids(self) -> List[str]:
        """Get the alernative identifiers for this node."""
        preds = [
            "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId",
            "oboinowl:hasAlternativeId",
            "oboInOwl:hasAlternativeId",
        ]
        rv = []
        for curie in self._get_properties(preds):
            norm_curie = manager.normalize_curie(curie)
            if norm_curie:
                rv.append(norm_curie)
            else:
                logger.warning("could not parse CURIE: %s", curie)
        return rv

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
        return _parse_uri_or_curie_or_str(self.id)

    @property
    def definition_provenance(self) -> List[str]:
        """Get the provenance CURIEs for the definition."""
        if self.meta and self.meta.definition and self.meta.definition.xrefs:
            return self.meta.definition.xrefs
        return []

    def get_provenance(self) -> List[str]:
        """Get provenance CURIEs from definition and xrefs."""
        return list(
            itt.chain(
                (
                    curie
                    for curie in self.definition_provenance
                    if curie.split(":")[0] in PROVENANCE_PREFIXES
                ),
                (xref.curie for xref in self.xrefs if xref.prefix in PROVENANCE_PREFIXES),
            )
        )


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
                xref_prefix, xref_identifier = _parse_uri_or_curie_or_str(xref.val)
                if xref_prefix is None or xref_identifier is None:
                    continue
                if xref_prefix != prefix:
                    continue
                if " " in xref_identifier:
                    tqdm.write(f"node {node.id} with space in xref {xref.val}")
                xrefs[xref_identifier] = node.id
        return xrefs

    def get_curie_to_name(self) -> Mapping[str, str]:
        """Get a mapping from CURIEs to names."""
        return {node.curie: node.lbl for node in self.nodes if node.lbl and node.prefix}


def _parse_uri_or_curie_or_str(
    s: str, *, debug: bool = False
) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Ground the node to a standard prefix and luid based on its id (URI)."""
    prefix, identifier = _compress_uri_or_curie_or_str(s, debug=debug)
    if prefix is None:
        return None, None
    resource = manager.get_resource(prefix)
    if resource is None:
        return None, None
    return resource.prefix, resource.standardize_identifier(identifier)


def _clean_uri_or_curie_or_str(s: str, *, keep_invalid: bool, debug: bool = False) -> Optional[str]:
    prefix, identifier = _parse_uri_or_curie_or_str(s=s, debug=debug)
    if prefix is not None and identifier is not None:
        return curie_to_str(prefix, identifier)
    elif keep_invalid:
        return s
    else:
        return None


WARNED = set()
YEARS = {f"{n}-" for n in range(1000, 2030)}


def _parse_obo_rel(s: str, identifier: str) -> Union[Tuple[str, str], Tuple[None, str]]:
    _, inner_identifier = identifier.split("#", 1)
    _p, _i = ground_relation(inner_identifier)
    if _p and _i:
        return _p, _i
    if s not in WARNED:
        tqdm.write(f"could not parse OBO internal relation: {s}")
        WARNED.add(s)
    return None, s


def _compress_uri_or_curie_or_str(
    s: str, *, debug: bool = False
) -> Union[Tuple[str, str], Tuple[None, str]]:
    from .upgrade import insert, upgrade

    s = s.replace(" ", "")

    cv = upgrade(s)
    if cv:
        return cv

    prefix, identifier = get_default_converter().parse_uri(s)
    if prefix and identifier:
        if prefix == "obo" and "#" in identifier:
            return _parse_obo_rel(s, identifier)
        return prefix, identifier

    if "upload.wikimedia.org" in s:
        return None, s

    for x in [
        "http://www.obofoundry.org/ro/#OBO_REL:",
        "http://www.obofoundry.org/ro/ro.owl#",
    ]:
        if s.startswith(x):
            prefix, identifier = ground_relation(s[len(x) :])
            if prefix and identifier:
                insert(s, prefix, identifier)
                return prefix, identifier
            elif s not in WARNED:
                tqdm.write(f"could not parse legacy RO: {s}")

    prefix, identifier = ground_relation(s)
    if prefix and identifier:
        return prefix, identifier

    # couldn't parse anything...
    if debug and (
        not s.startswith("_:")
        and " " not in s
        and "upload.wikimedia.org" not in s
        and "violinID:" not in s
        and s not in WARNED
        and s[:5] not in YEARS
        and not s.isnumeric()
    ):
        tqdm.write(f"could not parse {s}")
        WARNED.add(s)
    return None, s


class GraphDocument(BaseModel):
    """Represents a list of OBO graphs."""

    graphs: List[Graph]
