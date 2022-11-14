# -*- coding: utf-8 -*-

"""A wrapper around ROBOT functionality.

.. seealso:: https://robot.obolibrary.org
"""

import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import check_output
from typing import List, Optional, Union

import bioregistry
import requests
from pystow.utils import download, name_from_url
from typing_extensions import Literal

from .obograph import Graph, GraphDocument

__all__ = [
    "is_available",
    "ParseResults",
    # Conversions
    "convert",
    "convert_to_obograph_local",
    "convert_to_obograph_remote",
    "convert_to_obograph",
    # Processors
    "get_obograph_by_prefix",
    "get_obograph_by_iri",
    "get_obograph_by_path",
]

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if ROBOT is available."""
    # suggested in https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script
    from shutil import which

    return which("robot") is not None


CANONICAL = {
    "apollosv": "http://purl.obolibrary.org/obo/apollo_sv.owl",
    "cheminf": "http://semanticchemistry.github.io/semanticchemistry/ontology/cheminf.owl",
    "dideo": "http://purl.obolibrary.org/obo/dideo/release/2022-06-14/dideo.owl",
    "micro": "http://purl.obolibrary.org/obo/MicrO.owl",
    "ogsf": "http://purl.obolibrary.org/obo/ogsf-merged.owl",
    "mfomd": "http://purl.obolibrary.org/obo/MF.owl",
    "one": "http://purl.obolibrary.org/obo/ONE",
    "ons": "https://raw.githubusercontent.com/enpadasi/Ontology-for-Nutritional-Studies/master/ons.owl",
}


@dataclass
class ParseResults:
    """A dataclass containing an OBO Graph JSON and text output from ROBOT."""

    graph_document: Optional[GraphDocument]
    messages: List[str] = dataclasses.field(default_factory=list)
    iri: Optional[str] = None

    def squeeze(self, standardize: bool = False) -> Graph:
        """Get the first graph."""
        if self.graph_document is None:
            raise ValueError(f"graph document was not successfully parsed: {self.messages}")
        rv = self.graph_document.graphs[0]
        if standardize:
            rv = rv.standardize()
        return rv

    def guess(self, prefix: str) -> Graph:
        """Guess the right graph."""
        if self.graph_document is None:
            raise ValueError
        graphs = self.graph_document.graphs
        if 1 == len(graphs):
            return graphs[0]
        id_to_graph = {graph.id: graph for graph in graphs}
        standard_id = f"http://purl.obolibrary.org/obo/{prefix.lower()}.owl"
        if standard_id in id_to_graph:
            return id_to_graph[standard_id]
        if prefix in CANONICAL and CANONICAL[prefix] in id_to_graph:
            return id_to_graph[CANONICAL[prefix]]
        raise ValueError(f"Several graphs in {prefix}: {sorted(id_to_graph)}")

    def guess_version(self, prefix: str) -> Optional[str]:
        """Guess the version."""
        try:
            graph = self.guess(prefix)
        except ValueError:
            return None
        else:
            return graph.version or graph.version_iri


def get_obograph_by_iri(
    iri: str,
) -> ParseResults:
    """Get an ontology by its OBO Graph JSON iri."""
    res_json = requests.get(iri).json()
    graph_document = GraphDocument(**res_json)
    return ParseResults(graph_document=graph_document, iri=iri)


def get_obograph_by_path(path: Union[str, Path], *, iri: Optional[str] = None) -> ParseResults:
    """Get an ontology by its OBO Graph JSON file path."""
    res_json = json.loads(Path(path).resolve().read_text())
    graph_document = GraphDocument(**res_json)
    if iri is None:
        if graph_document.graphs and len(graph_document.graphs) == 1:
            iri = graph_document.graphs[0].id
    return ParseResults(graph_document=graph_document, iri=iri)


def get_obograph_by_prefix(
    prefix: str,
    *,
    json_path: Union[None, str, Path] = None,
    cache: bool = False,
) -> ParseResults:
    """Get an ontology by its Bioregistry prefix."""
    if prefix != bioregistry.normalize_prefix(prefix):
        raise ValueError("this function requires bioregistry canonical prefixes")

    messages = []
    json_iri = bioregistry.get_json_download(prefix)

    if json_iri is not None:
        try:
            parse_results = get_obograph_by_iri(json_iri)
        except (IOError, ValueError):
            msg = f"could not parse JSON for {prefix} from {json_iri}"
            messages.append(msg)
            logger.warning(msg)
        else:
            return parse_results

    owl_iri = bioregistry.get_owl_download(prefix)
    obo_iri = bioregistry.get_obo_download(prefix)

    for label, iri in [("OWL", owl_iri), ("OBO", obo_iri)]:
        if iri is None:
            continue

        try:
            if cache:
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, name_from_url(iri))
                    download(iri, path=path)
                    parse_results = convert_to_obograph_local(
                        path, json_path=json_path, from_iri=iri
                    )
            else:
                parse_results = convert_to_obograph_remote(iri, json_path=json_path)
        except subprocess.CalledProcessError:
            msg = f"could not parse {label} for {prefix} from {iri}"
            messages.append(msg)
            logger.warning(msg)
            continue
        else:
            # stick all messages before
            parse_results.messages = [*messages, *parse_results.messages]
            return parse_results

    return ParseResults(graph_document=None, messages=messages)


def convert_to_obograph_local(
    path: Union[str, Path],
    *,
    json_path: Union[None, str, Path] = None,
    from_iri: Optional[str] = None,
) -> ParseResults:
    """Convert a local OWL/OBO file to an OBO Graph JSON object.

    :param path: The path to a local OWL or OBO file
    :param json_path: The optional path to store the intermediate
        OBO Graph JSON file generated by ROBOT. If not given, the
        OBO Graph JSON file will be put in a temporary directory
        and deleted after the function finishes.
    :param from_iri: Use this parameter to say what IRI the graph came from
    :returns: An object with the parsed OBO Graph JSON and text
        output from the ROBOT conversion program
    """
    return convert_to_obograph(
        input_path=path, input_flag="-i", json_path=json_path, from_iri=from_iri
    )


def convert_to_obograph_remote(
    iri: str,
    *,
    json_path: Union[None, str, Path] = None,
) -> ParseResults:
    """Convert a remote OWL/OBO file to an OBO Graph JSON object.

    :param iri: The IRI for a remote OWL or OBO file
    :param json_path: The optional path to store the intermediate
        OBO Graph JSON file generated by ROBOT. If not given, the
        OBO Graph JSON file will be put in a temporary directory
        and deleted after the function finishes.
    :returns: An object with the parsed OBO Graph JSON and text
        output from the ROBOT conversion program
    """
    return convert_to_obograph(
        input_path=iri, input_flag="-I", json_path=json_path, input_is_iri=True
    )


def convert_to_obograph(
    input_path: Union[str, Path],
    *,
    input_flag: Optional[Literal["-i", "-I"]] = None,
    json_path: Union[None, str, Path] = None,
    input_is_iri: bool = False,
    extra_args: Optional[List[str]] = None,
    from_iri: Optional[str] = None,
    merge: bool = True,
) -> ParseResults:
    """Convert a local OWL file to a JSON file.

    :param input_path: Either a local file path or IRI. If a local file path
        is used, pass ``"-i"`` to ``flag``. If an IRI is used, pass ``"-I"``
        to ``flag``.
    :param input_flag: The flag to denote if the file is local or remote.
        Tries to infer from input string if none is given
    :param json_path: The optional path to store the intermediate
        OBO Graph JSON file generated by ROBOT. If not given, the
        OBO Graph JSON file will be put in a temporary directory
        and deleted after the function finishes.
    :param input_is_iri:
        Should the ``input_path`` varible be considered as an IRI that
        gets stored in the returned parse results?
    :param extra_args:
        Extra positional arguments to pass in the command line
    :param from_iri: Use this parameter to say what IRI the graph came from
    :param merge: Use ROBOT's merge command to squash all graphs together

    :returns: An object with the parsed OBO Graph JSON and text
        output from the ROBOT conversion program

    :raises ValueError: if a graph is missing an ID
    :raises TypeError: if ``input_as_iri`` is marked as true but a path
        object is given for the ``input_path``
    """
    if input_is_iri and not isinstance(input_path, str):
        raise TypeError
    if input_is_iri and from_iri is not None:
        raise ValueError("can't specifiy from_iri when input is IRI")

    with _path_context(json_path) as path:
        ret = convert(
            input_path=input_path,
            input_flag=input_flag,
            output_path=path,
            fmt="json",
            extra_args=extra_args,
            merge=merge,
        )
        messages = ret.strip().splitlines()
        graph_document_raw = json.loads(path.read_text())

        graphs_raw = graph_document_raw["graphs"]
        if len(graphs_raw) == 1 and "id" not in graphs_raw[0]:
            if input_is_iri:
                logger.warning(
                    f"{input_path} has a single graph, missing an ID. assigning with IRI"
                )
                graphs_raw[0]["id"] = input_path
            elif from_iri is not None:
                logger.warning(
                    f"{input_path} has a single graph, missing an ID. assigning with IRI: {from_iri}"
                )
                graphs_raw[0]["id"] = from_iri
            else:
                raise ValueError(f"{input_path} only graph is missing id")
        else:
            missing = [i for i, graph in enumerate(graphs_raw) if "id" not in graph]
            if missing:
                raise ValueError(f"{input_path} graphs missing IDs: {missing}")

        graph_document = GraphDocument(**graph_document_raw)
        return ParseResults(
            graph_document=graph_document,
            messages=messages,
            iri=input_path if input_is_iri else None,  # type:ignore
        )


#: Prefixes that denote remote resources
PROTOCOLS = {
    "https://",
    "http://",
    "ftp://",
    "ftps://",
}


def _is_remote(url: Union[str, Path]) -> bool:
    return isinstance(url, str) and any(url.startswith(protocol) for protocol in PROTOCOLS)


@contextmanager
def _path_context(path: Union[None, str, Path], name: str = "output.json"):
    if path is not None:
        yield Path(path).resolve()
    else:
        with tempfile.TemporaryDirectory() as directory:
            yield Path(directory).joinpath(name)


def convert(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    input_flag: Optional[Literal["-i", "-I"]] = None,
    *,
    merge: bool = True,
    fmt: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> str:
    """Convert an OBO file to an OWL file with ROBOT."""
    if input_flag is None:
        input_flag = "-I" if _is_remote(input_path) else "-i"
    if merge:
        args = [
            "robot",
            "merge",
            input_flag,
            str(input_path),
            "convert",
        ]
    else:
        args = [
            "robot",
            "convert",
            input_flag,
            str(input_path),
        ]
    args.extend(("-o", str(output_path)))
    if extra_args:
        args.extend(extra_args)
    if fmt:
        args.extend(("--format", fmt))
    logger.debug("Running shell command: %s", args)
    ret = check_output(  # noqa:S603
        args,
        cwd=os.path.dirname(__file__),
    )
    return ret.decode()
