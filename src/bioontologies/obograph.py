"""Data structures for representing OBO Graphs.

.. seealso:: https://github.com/geneontology/obographs
"""

import itertools as itt
import logging
import typing
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union

import bioregistry
import curies
import pandas as pd
from bioregistry import manager
from curies import Reference, ReferenceTuple
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from typing_extensions import Literal, Self

from .constants import CANONICAL, IRI_TO_PREFIX
from .relations import get_normalized_label, ground_relation, label_norm

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
    "OBO_SYNONYM_TO_OIO",
    "OIO_TO_REFERENCE",
]

logger = logging.getLogger(__name__)

OBO_URI_PREFIX = "http://purl.obolibrary.org/obo/"
OBO_URI_PREFIX_LEN = len(OBO_URI_PREFIX)
IDENTIFIERS_HTTP_PREFIX = "http://identifiers.org/"
IDENTIFIERS_HTTPS_PREFIX = "https://identifiers.org/"
PROVENANCE_PREFIXES = {"pubmed", "pmc", "doi", "arxiv", "biorxiv", "medrxiv", "agricola"}

MISSING_PREDICATE_LABELS = set()

MaybeCURIE = Union[Tuple[str, str], Tuple[None, None]]


class StandardizeMixin:
    """A mixin for classes representing standardizable data."""

    def standardize(self) -> Self:
        """Standardize the data in this class."""
        raise NotImplementedError

    def raise_on_unstandardized(self):
        """Raise an exception if standarization has not occurred."""
        if not self.standardized:
            raise ValueError


class Property(BaseModel, StandardizeMixin):
    """Represent a property inside a metadata element."""

    predicate_raw: str = Field(..., alias="pred")
    value_raw: str = Field(..., alias="val")

    # Extras beyond the OBO Graph spec
    standardized: bool = Field(False, exclude=True)
    predicate: Optional[Reference] = None
    value: Optional[Reference] = None

    def standardize(self) -> Self:
        """Standardize this property."""
        self.value_raw = self.value_raw.replace("\n", " ")
        self.predicate = _get_reference(self.predicate_raw)
        self.value = _get_reference(self.value_raw)
        self.standardized = True
        return self


class Definition(BaseModel):
    """Represents a definition for a node."""

    value: Optional[str] = Field(default=None, alias="val")
    xrefs_raw: Optional[List[str]] = Field(
        default=None, alias="xrefs"
    )  # Just a list of CURIEs/IRIs

    # Extras beyond the OBO Graph spec
    references: Optional[List[Reference]] = None
    standardized: bool = Field(False, exclude=True)

    def standardize(self) -> Self:
        """Standardize the xref."""
        if self.xrefs_raw:
            self.references = _get_references(self.xrefs_raw)
        if self.value:
            self.value = self.value.strip().replace("  ", " ").replace("\n", " ")
        self.standardized = True
        return self

    @classmethod
    def from_parsed(cls, value: str, references: Optional[List[Reference]] = None) -> "Definition":
        """Construct a definition object from pre-standardized content."""
        if not references:
            references = []
        return cls(
            value=value,
            xrefs_raw=[r.curie for r in references],
            references=references,
            standardize=True,
        )


class Xref(BaseModel, StandardizeMixin):
    """Represents a cross-reference."""

    value_raw: str = Field(..., alias="val")
    predicate_raw: str = Field(
        default="oboinowl:hasDbXref"
    )  # note this is not part of the OBO Graph spec

    # Extras beyond the OBO Graph spec
    predicate: Optional[Reference] = Field(
        default=None, description="The reference for the predicate"
    )
    value: Optional[Reference] = Field(default=None, description="The reference for the value")
    standardized: bool = Field(default=False, exclude=True)

    def standardize(self) -> Self:
        """Standardize the xref."""
        self.value = _get_reference(self.value_raw)
        self.predicate = _get_reference(self.predicate_raw)
        self.standardized = True
        return self

    @classmethod
    def from_parsed(cls, predicate: Reference, value: Reference) -> "Xref":
        """Construct an xref object from pre-standardized content."""
        return Xref(
            val=value.curie,
            value=value,
            predicate_raw=predicate.curie,
            predicate=predicate,
            standardized=True,
        )


#: Mapping from shorthand for predicates to qualified references
OIO_TO_REFERENCE: Mapping[str, Reference] = {
    "hasExactSynonym": Reference(prefix="oboInOwl", identifier="hasExactSynonym"),
    "hasBroadSynonym": Reference(prefix="oboInOwl", identifier="hasBroadSynonym"),
    "hasNarrowSynonym": Reference(prefix="oboInOwl", identifier="hasNarrowSynonym"),
    "hasRelatedSynonym": Reference(prefix="oboInOwl", identifier="hasRelatedSynonym"),
}

#: A mapping from OBO flat file format internal synonym types to OBO in OWL vocabulary
#: identifiers. See https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_4.html
OBO_SYNONYM_TO_OIO = {
    "EXACT": "hasExactSynonym",
    "BROAD": "hasBroadSynonym",
    "NARROW": "hasNarrowSynonym",
    "RELATED": "hasRelatedSynonym",
}


class Synonym(BaseModel, StandardizeMixin):
    """Represents a synonym inside an object meta."""

    value: Optional[str] = Field(default=None, alias="val")
    predicate_raw: str = Field(default="hasExactSynonym", alias="pred")
    synonym_type_raw: str = Field(
        alias="synonymType", default="oboInOwl:SynonymType", example="OMO:0003000"
    )  # noqa:N815
    xrefs_raw: List[str] = Field(
        default_factory=list,
        alias="xrefs",
        description="A list of CURIEs/IRIs for provenance for the synonym",
    )

    # Added
    predicate: Optional[Reference] = Field(
        default=None, example=Reference(prefix="", identifier="hasExactSynonym")
    )
    synonym_type: Optional[Reference] = Field(
        default=None, example=Reference(prefix="OMO", identifier="0003000")
    )
    references: Optional[List[Reference]] = None
    standardized: bool = Field(False, exclude=True)

    def standardize(self) -> Self:
        """Standardize the synoynm."""
        self.predicate = _get_reference(self.predicate_raw)
        self.synonym_type = self.synonym_type_raw and _get_reference(self.synonym_type_raw)
        if self.value:
            self.value = self.value.strip().replace("\n", " ").replace("  ", " ")
        if self.xrefs_raw:
            self.references = _get_references(self.xrefs_raw)
        self.standardized = True
        return self

    @classmethod
    def from_parsed(
        cls,
        name: str,
        predicate: Reference,
        synonym_type: Optional[Reference] = None,
        references: Optional[List[Reference]] = None,
    ) -> "Synonym":
        """Construct a synonym object from pre-standardized content."""
        if not references:
            references = []
        if synonym_type is None:
            synonym_type = Reference(prefix="oboInOwl", identifier="SynonymType")
        return Synonym(
            val=name,
            predicate_raw=predicate.curie,
            predicate=predicate,
            synonym_type_raw=synonym_type.curie,
            synonym_type=synonym_type,
            standardized=True,
            xrefs_raw=[x.curie for x in references],
            references=references,
        )


class Meta(BaseModel, StandardizeMixin):
    """Represents the metadata about a node or ontology."""

    definition: Optional[Definition] = None
    subsets: Optional[List[str]] = None
    xrefs: Optional[List[Xref]] = None
    synonyms: Optional[List[Synonym]] = None
    comments: Optional[List] = None
    version: Optional[str] = None
    properties: Optional[List[Property]] = Field(None, alias="basicPropertyValues")
    deprecated: bool = False

    #
    standardized: bool = Field(False, exclude=True)

    def standardize(self) -> Self:
        """Standardize the metadata."""
        for prop in self.properties or []:
            prop.standardize()
        for synonym in self.synonyms or []:
            synonym.standardize()
        if self.definition:
            self.definition.standardize()
        if self.xrefs:
            xrefs: List[Xref] = []
            seen: Set[Tuple[str, str]] = set()
            for xref in self.xrefs:
                xref.standardize()
                if xref.predicate is None or xref.value is None:
                    continue
                # if xref.value.prefix == self.prefix and xref.value.identifier == self.luid:
                # this is a reference to itself, weird!
                #    continue
                if xref.value.pair in seen:
                    continue
                seen.add(xref.value.pair)
                xrefs.append(xref)
            # we ignore type checking since the loop for construting the xrefs lis
            # checks that the predicate and value are both non-none
            self.xrefs = sorted(
                xrefs, key=lambda x: (x.predicate.curie, x.value.curie)  # type:ignore
            )
        return self


class Edge(BaseModel):
    """Represents an edge in an OBO Graph."""

    sub: str = Field(..., alias="sub", example="http://purl.obolibrary.org/obo/CHEBI_99998")
    pred: str = Field(..., alias="pred", example="is_a")
    obj: str = Field(..., alias="obj", example="http://purl.obolibrary.org/obo/CHEBI_24995")
    meta: Optional[Meta] = None

    standardized: bool = Field(False, exclude=True)
    subject: Optional[Reference] = Field(
        default=None, example=Reference(prefix="chebi", identifier="99998")
    )
    predicate: Optional[Reference] = Field(
        default=None, example=Reference(prefix="rdfs", identifier="subClassOf")
    )
    object: Optional[Reference] = Field(
        default=None, example=Reference(prefix="chebi", identifier="24995")
    )

    def as_tuple(self) -> Tuple[str, str, str]:
        """Get the edge as a tuple."""
        if self.subject is None or self.predicate is None or self.object is None:
            raise ValueError
        return self.subject.curie, self.predicate.curie, self.object.curie

    def standardize(self) -> Self:
        """Standardize the edge."""
        if self.meta:
            self.meta.standardize()
        self.subject = _get_reference(self.sub)
        self.predicate = _get_reference(self.pred)
        self.object = _get_reference(self.obj)
        self.standardized = True
        return self

    @classmethod
    def from_parsed(
        cls, s: Reference, p: Reference, o: Reference, meta: Optional[Meta] = None
    ) -> "Edge":
        """Construct an edge object from pre-standardized content."""
        return Edge(
            sub=s.curie,
            pred=p.curie,
            obj=o.curie,
            standardized=True,
            subject=s,
            predicate=p,
            object=o,
            meta=meta,
        )


def _help_get_properties(self, predicate_iris: Union[str, List[str]]) -> List[str]:
    if not self.meta:
        return []
    if isinstance(predicate_iris, str):
        predicate_iris = [predicate_iris]
    return [
        prop.value.curie if prop.value else prop.value_raw
        for prop in self.meta.properties or []
        if any(prop.predicate_raw == predicate_iri for predicate_iri in predicate_iris)
    ]


class Node(BaseModel, StandardizeMixin):
    """Represents a node in an OBO Graph."""

    id: str = Field(..., description="The IRI for the node")
    name: Optional[str] = Field(None, alias="lbl", description="The name of the node")
    meta: Optional[Meta] = None
    type: Literal["CLASS", "PROPERTY", "INDIVIDUAL"] = Field(..., description="Type of node")

    # Extras beyond OBO Graph spec
    reference: Optional[Reference] = None
    standardized: bool = Field(False, exclude=True)

    @property
    def prefix(self) -> Optional[str]:
        """Get the prefix for the node if it has been standardized."""
        return self.reference and self.reference.prefix

    @property
    def identifier(self) -> Optional[str]:
        """Get the identifier for the node if it has been standardized."""
        return self.reference and self.reference.identifier

    def standardize(self) -> Self:
        """Ground the node to a standard prefix and luid based on its id (URI)."""
        prefix, identifier = _parse_uri_or_curie_or_str(self.id)
        if prefix and identifier:
            self.reference = Reference(prefix=prefix, identifier=identifier)
        if self.name:
            self.name = self.name.strip().replace("\n", " ").replace("  ", " ")
        if self.meta:
            self.meta.standardize()
        self.standardized = True
        return self

    @property
    def curie(self) -> str:
        """Get the CURIE string representing this node or error if not normalized."""
        if not self.reference:
            raise ValueError(f"can not give curie for node {self.id}")
        return self.reference.curie

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
        rv = []
        skip_skos = {
            "definition",
            "altLabel",
            "example",
            "prefLabel",
            "note",
            "scopeNote",
            "changeNote",
            "editorialNote",
            "hasTopConcept",
            "notation",
            "historyNote",
            "inScheme",
        }
        if self.meta:
            for xref in self.meta.xrefs or []:
                if not xref.predicate or not xref.value or xref.value.prefix in PROVENANCE_PREFIXES:
                    continue
                rv.append(xref)
            for prop in self.meta.properties or []:
                if prop.predicate is None:
                    continue
                if prop.predicate.prefix == "skos" and prop.predicate.identifier not in skip_skos:
                    if prop.value is None:
                        WARNED[prop.value_raw] += 1
                        continue
                    rv.append(
                        Xref(
                            val=prop.value.curie,
                            predicate_raw=prop.predicate.curie,
                            value=prop.value,
                            predicate=prop.predicate,
                            standardized=True,
                        )
                    )
        return rv

    @property
    def properties(self) -> List[Property]:
        """Get the properties for this node."""
        if not self.meta or self.meta.properties is None:
            return []
        # TODO filter out ones grabbed by other getters
        return self.meta.properties

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
            return self.meta.definition.value
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
    def definition_provenance(self) -> List[Reference]:
        """Get the provenance CURIEs for the definition."""
        if self.meta and self.meta.definition and self.meta.definition.references:
            return self.meta.definition.references
        return []

    def get_provenance(self) -> List[Reference]:
        """Get provenance CURIEs from definition and xrefs."""
        return list(
            itt.chain(
                (
                    reference
                    for reference in self.definition_provenance
                    if reference.prefix in PROVENANCE_PREFIXES
                ),
                (
                    xref.value
                    for xref in self.xrefs
                    if xref.value and xref.value.prefix in PROVENANCE_PREFIXES
                ),
            )
        )


class Graph(BaseModel, StandardizeMixin):
    """A graph corresponds to an ontology."""

    id: Optional[str] = None
    meta: Optional[Meta] = None
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    equivalentNodesSets: Any  # noqa:N815
    logicalDefinitionAxioms: Any  # noqa:N815
    domainRangeAxioms: Any  # noqa:N815
    propertyChainAxioms: Any  # noqa:N815

    # Extras beyond the OBO Graph spec
    prefix: Optional[str] = None
    standardized: bool = Field(False, exclude=True)

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
        return self.meta and self.meta.version

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
    ) -> Self:
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

        if self.meta:
            self.meta.standardize()

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

        if self.prefix is None:
            self._standardize_prefix()

        return self

    def _standardize_prefix(self):
        if not self.id:
            return
        if self.id in IRI_TO_PREFIX:
            self.prefix = IRI_TO_PREFIX[self.id]
        elif self.id.startswith("http://purl.obolibrary.org/obo/"):
            for suffix in [".owl", ".obo", ".json"]:
                if not self.id.endswith(suffix):
                    continue
                prefix = (
                    self.id.removeprefix("http://purl.obolibrary.org/obo/")
                    .removesuffix(suffix)
                    .removesuffix("_import")
                )
                if prefix != bioregistry.normalize_prefix(prefix):
                    tqdm.write(f"could not guess prefix from {self.id}")
                    return
                self.prefix = prefix
                return

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

    def get_xrefs(self) -> List[Tuple[Reference, Reference, Reference]]:
        """Get all database cross-references from the ontology."""
        rv = []
        for node in self.nodes:
            if node.reference is None:
                continue
            for xref in node.xrefs:
                if xref.value is None or " " in xref.value.identifier:
                    tqdm.write(f"node {node.id} with space in xref {xref.value_raw}")
                    continue
                rv.append((node.reference, xref.predicate, xref.value))
        return rv

    def _get_edge_predicate_label(self, edge: Edge, ctn) -> str:
        if edge.predicate:
            label = get_normalized_label(edge.predicate.curie)
            if label:
                return label

            label = ctn.get(edge.predicate.curie)
            if label:
                return label_norm(label)

            label = get_normalized_label(edge.pred)
            if label:
                return label

            if edge.predicate.curie not in MISSING_PREDICATE_LABELS:
                MISSING_PREDICATE_LABELS.add(edge.predicate.curie)
                tqdm.write(f"No label for CURIE {edge.predicate.curie}")
            return edge.predicate.curie

        label = get_normalized_label(edge.pred)
        if label:
            return label

        if edge.pred not in MISSING_PREDICATE_LABELS:
            MISSING_PREDICATE_LABELS.add(edge.pred)
            tqdm.write(f"No CURIE/label for {edge.pred}")
        return edge.pred

    def get_edges_df(self) -> pd.DataFrame:
        """Get all triples as a dataframe."""
        self.raise_on_unstandardized()
        if self.prefix is None:
            raise ValueError(f"Could not parse prefix in {self.id}")
        columns = [":START_ID", ":TYPE", ":END_ID", "curie"]
        ctn = self.get_curie_to_name()
        rows = sorted(
            (
                edge.subject.curie,
                self._get_edge_predicate_label(edge, ctn=ctn),
                edge.object.curie,
                edge.predicate.curie,
            )
            for edge in self.edges
            if edge.subject
            and edge.predicate
            and edge.object
            and edge.subject.prefix == self.prefix
        )
        return pd.DataFrame(rows, columns=columns).drop_duplicates()

    def get_sssom_df(self) -> pd.DataFrame:
        """Get a SSSOM dataframe of mappings."""
        self.raise_on_unstandardized()
        if self.prefix is None:
            raise ValueError(f"Could not parse prefix in {self.id}")
        columns = [
            "source_id",
            "source_label",
            "predicate_id",
            "object_id",
        ]
        # TODO add justification?
        rows = [
            (
                node.curie,
                node.name,
                xref.predicate.curie,
                xref.value.curie,
            )
            for node in self.nodes
            if node.prefix == self.prefix
            for xref in node.xrefs
            if xref.predicate and xref.value
        ]
        return pd.DataFrame(sorted(rows), columns=columns)

    def get_nodes_df(self, sep: str = ";") -> pd.DataFrame:
        """Get a nodes dataframe appropriate for serialization."""
        self.raise_on_unstandardized()
        if self.prefix is None:
            raise ValueError(f"Could not parse prefix in {self.id}")
        columns = [
            "curie:ID",
            "name:string",
            "synonyms:string[]",
            "synonym_predicates:string[]",
            "synonym_types:string[]",
            "definition:string",
            "deprecated:boolean",
            "type:string",
            "provenance:string[]",
            "alts:string[]",
            "replaced_by:string",
            "xrefs:string[]",
            "xref_types:string[]",
            "version:string",
        ]
        version = self.version
        rows = []
        for node in self.nodes:
            if node.prefix != self.prefix:
                continue
            synonym_predicates, synonym_types, synonym_values = [], [], []
            for synonym in node.synonyms:
                if synonym.predicate and synonym.synonym_type and synonym.value:
                    synonym_predicates.append(synonym.predicate.curie)
                    synonym_types.append(synonym.synonym_type.curie)
                    synonym_values.append(synonym.value)
            xref_types, xref_values = [], []
            for xref in node.xrefs:
                if xref.predicate and xref.value:
                    xref_types.append(xref.predicate.curie)
                    xref_values.append(xref.value.curie)
            # prop_types, prop_values = [], []
            rows.append(
                (
                    node.curie,
                    node.name,
                    sep.join(synonym_values),
                    sep.join(synonym_predicates),
                    sep.join(synonym_types),
                    node.definition,
                    node.deprecated,
                    node.type,
                    sep.join(reference.curie for reference in node.get_provenance()),
                    sep.join(node.alternative_ids),
                    node.replaced_by,
                    sep.join(xref_values),
                    sep.join(xref_types),
                    version,
                )
            )
        return pd.DataFrame(sorted(rows), columns=columns)

    def get_incoming_xrefs(self, prefix: str) -> Mapping[str, str]:
        """Get incoming xrefs.

        :param prefix: An external prefix.
        :returns:
            A dictionary of external local unique identifiers
            to local unique identifiers in this ontology
        """
        ontology_prefix = self.prefix or self.default_namespace
        return {
            xref.identifier: node.identifier
            for node, _predicate, xref in self.get_xrefs()
            if xref.prefix == prefix and node.prefix == ontology_prefix
        }

    def get_curie_to_name(self) -> Mapping[str, str]:
        """Get a mapping from CURIEs to names."""
        return {
            node.curie: node.name for node in self.nodes if node.name and node.reference is not None
        }


def _parse_uri_or_curie_or_str(
    s: str, *, debug: bool = False
) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Ground the node to a standard prefix and luid based on its id (URI)."""
    reference_tuple = omni_parse(s, debug=debug)
    if reference_tuple is None:
        return None, None
    resource = manager.get_resource(reference_tuple.prefix)
    if resource is None:
        return None, None
    return resource.prefix, resource.standardize_identifier(reference_tuple.identifier)


def _get_reference(s: str, *, debug: bool = False) -> Optional[Reference]:
    p, i = _parse_uri_or_curie_or_str(s, debug=debug)
    if p and i:
        return Reference(prefix=p, identifier=i)
    return None


def _get_references(strings: List[str]) -> List[Reference]:
    references = [_get_reference(s) for s in strings]
    rv = [reference for reference in references if reference is not None]
    return rv


WARNED: typing.Counter[str] = Counter()
YEARS = {f"{n}-" for n in range(1000, 2030)}


def write_warned(path: Union[str, Path]) -> None:
    """Write warned unparsable."""
    path = Path(path).resolve()
    path.write_text("\n".join(f"{k}\t{v}" for k, v in sorted(WARNED.items())))


def _parse_obo_rel(s: str, identifier: str) -> Optional[ReferenceTuple]:
    _, inner_identifier = identifier.split("#", 1)
    _p, _i = ground_relation(inner_identifier)
    if _p and _i:
        return ReferenceTuple(_p, _i)
    if s not in WARNED:
        tqdm.write(f"could not parse OBO internal relation: {s}")
    WARNED[s] += 1
    return None


@lru_cache(1)
def _get_converter():
    return curies.Converter(records=bioregistry.manager.get_curies_records(include_prefixes=True))


def omni_parse(s: str, *, debug: bool = False) -> Optional[ReferenceTuple]:
    """Parse a string, CURIE, or IRI into a proper refernce, if possible."""
    from .upgrade import insert, upgrade

    s = s.replace(" ", "")

    cv = upgrade(s)
    if cv is not None:
        return cv

    prefix, identifier = _get_converter().parse_uri(s)
    if prefix and identifier:
        if prefix == "obo" and "#" in identifier:
            return _parse_obo_rel(s, identifier)
        return ReferenceTuple(prefix, identifier)

    if "upload.wikimedia.org" in s:
        return None

    for x in [
        "http://www.obofoundry.org/ro/#OBO_REL:",
        "http://www.obofoundry.org/ro/ro.owl#",
    ]:
        if s.startswith(x):
            prefix, identifier = ground_relation(s[len(x) :])
            if prefix and identifier:
                insert(s, prefix, identifier)
                return ReferenceTuple(prefix, identifier)
            if s not in WARNED:
                tqdm.write(f"could not parse legacy RO: {s}")
            WARNED[s] += 1

    prefix, identifier = ground_relation(s)
    if prefix and identifier:
        return ReferenceTuple(prefix, identifier)

    # couldn't parse anything...
    if debug and (
        not s.startswith("_:")
        and " " not in s
        and "upload.wikimedia.org" not in s
        and "violinID:" not in s
        and s[:5] not in YEARS
        and not s.isnumeric()
    ):
        if s not in WARNED:
            tqdm.write(f"could not parse {s}")
        WARNED[s] += 1
    return None


class GraphDocument(BaseModel):
    """Represents a list of OBO graphs."""

    graphs: List[Graph]

    def standardize(self) -> Self:
        """Standardize all graphs in the document."""
        for graph in self.graphs:
            graph.standardize()
        return self

    def guess(self, prefix: str) -> Graph:
        """Guess the primary graph."""
        if 1 == len(self.graphs):
            return self.graphs[0]
        id_to_graph = {graph.id: graph for graph in self.graphs if graph.id}
        for suffix in ["owl", "obo", "json"]:
            standard_id = f"http://purl.obolibrary.org/obo/{prefix.lower()}.{suffix}"
            if standard_id in id_to_graph:
                return id_to_graph[standard_id]
        if prefix in CANONICAL and CANONICAL[prefix] in id_to_graph:
            return id_to_graph[CANONICAL[prefix]]
        raise ValueError(f"Several graphs in {prefix}: {sorted(id_to_graph)}")
