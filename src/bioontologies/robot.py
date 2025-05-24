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
from typing import Any, Literal

import bioregistry
import pystow
import requests
from pystow.utils import download, name_from_url

from .obograph import Graph, GraphDocument

__all__ = [
    "ParseResults",
    "convert",
    "convert_to_obograph",
    "convert_to_obograph_local",
    "convert_to_obograph_remote",
    "get_obograph_by_iri",
    "get_obograph_by_path",
    "get_obograph_by_prefix",
    "is_available",
]

logger = logging.getLogger(__name__)

#: The default ROBOT version to download
VERSION = "1.9.7"
ROBOT_MODULE = pystow.module("robot")


def get_robot_jar_path(*, version: str | None = None) -> Path:
    """Ensure the robot jar is there."""
    if version is None:
        version = VERSION
    url = f"https://github.com/ontodev/robot/releases/download/v{version}/robot.jar"
    return ROBOT_MODULE.ensure(url=url, version=version)


def is_available() -> bool:
    """Check if ROBOT is available."""
    from shutil import which

    if which("java") is None:
        # suggested in https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script
        logger.error("java is not on the PATH")
        return False

    try:
        check_output(["java", "--help"])  # noqa:S607,S603
    except Exception:
        logger.error(
            "java --help failed - this means the java runtime environment (JRE) "
            "might not be configured properly"
        )
        return False

    robot_jar_path = get_robot_jar_path()
    if not robot_jar_path.is_file():
        logger.error("ROBOT was not successfully downloaded to %s", robot_jar_path)
        # ROBOT was unsuccessfully downloaded
        return False

    try:
        call_robot(["--help"])
    except Exception:
        logger.error("ROBOT was downloaded to %s but could not be run with --help", robot_jar_path)
        return False

    return True


def call_robot(args: list[str]) -> str:
    """Run a robot command and return the output as a string."""
    rr = ["java", "-jar", str(get_robot_jar_path()), *args]
    logger.debug("Running shell command: %s", args)
    ret = check_output(  # noqa:S603
        rr,
        cwd=os.path.dirname(__file__),
    )
    return ret.decode()


@dataclass
class ParseResults:
    """A dataclass containing an OBO Graph JSON and text output from ROBOT."""

    graph_document: GraphDocument | None
    messages: list[str] = dataclasses.field(default_factory=list)
    iri: str | None = None

    def squeeze(self, standardize: bool = False, **kwargs: Any) -> Graph:
        """Get the first graph."""
        if self.graph_document is None:
            raise ValueError(f"graph document was not successfully parsed: {self.messages}")
        rv = self.graph_document.graphs[0]
        if standardize:
            rv = rv.standardize(**kwargs)
        return rv

    def guess(self, prefix: str) -> Graph:
        """Guess the right graph."""
        if self.graph_document is None:
            raise ValueError("no graph document")
        return self.graph_document.guess(prefix)

    def guess_version(self, prefix: str) -> str | None:
        """Guess the version."""
        try:
            graph = self.guess(prefix)
        except ValueError:
            return None
        else:
            return graph.version or graph.version_iri

    def write(self, path: str | Path) -> None:
        """Write the graph document to a file in JSON."""
        if not self.graph_document:
            raise ValueError
        path = Path(path)
        path.write_text(
            self.graph_document.json(
                indent=2, sort_keys=True, exclude_unset=True, exclude_none=True
            )
        )


def get_obograph_by_iri(
    iri: str,
    timeout: int | None = 60,
) -> ParseResults:
    """Get an ontology by its OBO Graph JSON iri."""
    res_json = requests.get(iri, timeout=timeout).json()
    correct_raw_json(res_json)
    graph_document = GraphDocument.model_validate(res_json)
    return ParseResults(graph_document=graph_document, iri=iri)


def get_obograph_by_path(path: str | Path, *, iri: str | None = None) -> ParseResults:
    """Get an ontology by its OBO Graph JSON file path."""
    res_json = json.loads(Path(path).resolve().read_text())
    correct_raw_json(res_json)
    graph_document = GraphDocument.model_validate(res_json)
    if iri is None:
        if graph_document.graphs and len(graph_document.graphs) == 1:
            iri = graph_document.graphs[0].id
    return ParseResults(graph_document=graph_document, iri=iri)


GETTER_MESSAGES = []


def get_obograph_by_prefix(
    prefix: str,
    *,
    json_path: None | str | Path = None,
    cache: bool = False,
    check: bool = True,
    reason: bool = True,
) -> ParseResults:
    """Get an ontology by its Bioregistry prefix."""
    if prefix != bioregistry.normalize_prefix(prefix):
        raise ValueError(f"this function requires bioregistry canonical prefixes: {prefix}")

    messages = []
    json_iri = bioregistry.get_json_download(prefix)

    if json_iri is not None:
        try:
            parse_results = get_obograph_by_iri(json_iri)
        except (OSError, ValueError, TypeError) as e:
            msg = f"[{prefix}] could not parse JSON from {json_iri}: {e}"
            messages.append(msg)
            GETTER_MESSAGES.append(msg)
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
                        path, json_path=json_path, from_iri=iri, check=check
                    )
            else:
                parse_results = convert_to_obograph_remote(
                    iri, json_path=json_path, check=check, reason=reason
                )
        except (subprocess.CalledProcessError, KeyError):
            msg = f"[{prefix}] could not parse {label} from {iri}"
            messages.append(msg)
            GETTER_MESSAGES.append(msg)
            logger.warning(msg)
            continue
        else:
            # stick all messages before
            parse_results.messages = [*messages, *parse_results.messages]
            return parse_results

    return ParseResults(graph_document=None, messages=messages)


def convert_to_obograph_local(
    path: str | Path,
    *,
    json_path: None | str | Path = None,
    from_iri: str | None = None,
    check: bool = True,
) -> ParseResults:
    """Convert a local OWL/OBO file to an OBO Graph JSON object.

    :param path: The path to a local OWL or OBO file
    :param json_path: The optional path to store the intermediate
        OBO Graph JSON file generated by ROBOT. If not given, the
        OBO Graph JSON file will be put in a temporary directory
        and deleted after the function finishes.
    :param from_iri: Use this parameter to say what IRI the graph came from
    :param check:
        By default, the OBO writer strictly enforces
        `document structure rules <http://owlcollab.github.io/oboformat/doc/obo-syntax.html#4>`_.
        If an ontology violates these, the convert to OBO operation will fail.
        These checks can be ignored by setting this to false.
    :returns: An object with the parsed OBO Graph JSON and text
        output from the ROBOT conversion program
    """
    return convert_to_obograph(
        input_path=path, input_flag="-i", json_path=json_path, from_iri=from_iri, check=check
    )


def convert_to_obograph_remote(
    iri: str,
    *,
    json_path: None | str | Path = None,
    check: bool = True,
    reason: bool = True,
) -> ParseResults:
    """Convert a remote OWL/OBO file to an OBO Graph JSON object.

    :param iri: The IRI for a remote OWL or OBO file
    :param json_path: The optional path to store the intermediate
        OBO Graph JSON file generated by ROBOT. If not given, the
        OBO Graph JSON file will be put in a temporary directory
        and deleted after the function finishes.
    :param check:
        By default, the OBO writer strictly enforces
        `document structure rules <http://owlcollab.github.io/oboformat/doc/obo-syntax.html#4>`.
        If an ontology violates these, the convert to OBO operation will fail.
        These checks can be ignored by setting this to false.
    :param reason:
        Turn on ontology reasoning
    :returns: An object with the parsed OBO Graph JSON and text
        output from the ROBOT conversion program
    """
    return convert_to_obograph(
        input_path=iri,
        input_flag="-I",
        json_path=json_path,
        input_is_iri=True,
        check=check,
        reason=reason,
    )


def convert_to_obograph(
    input_path: str | Path,
    *,
    input_flag: Literal["-i", "-I"] | None = None,
    json_path: None | str | Path = None,
    input_is_iri: bool = False,
    extra_args: list[str] | None = None,
    from_iri: str | None = None,
    merge: bool = True,
    check: bool = True,
    reason: bool = True,
    debug: bool = False,
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
    :param check:
        By default, the OBO writer strictly enforces
        `document structure rules <http://owlcollab.github.io/oboformat/doc/obo-syntax.html#4>`.
        If an ontology violates these, the convert to OBO operation will fail.
        These checks can be ignored by setting this to false.
    :param reason:
        Turn on ontology reasoning
    :param debug:
        Turn on debugging (-vvv)

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
            check=check,
            reason=reason,
            debug=debug,
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
                    f"{input_path} has a single graph, missing an ID. "
                    f"assigning with IRI: {from_iri}"
                )
                graphs_raw[0]["id"] = from_iri
            else:
                raise ValueError(f"{input_path} only graph is missing id")
        else:
            missing = [i for i, graph in enumerate(graphs_raw) if "id" not in graph]
            if missing:
                raise ValueError(f"{input_path} graphs missing IDs: {missing}")

        correct_raw_json(graph_document_raw)
        graph_document = GraphDocument.model_validate(graph_document_raw)
        return ParseResults(
            graph_document=graph_document,
            messages=messages,
            iri=input_path if input_is_iri else None,  # type:ignore
        )


def correct_raw_json(graph_document_raw) -> None:
    """Correct issues in raw graph documents, in place."""
    for graph in graph_document_raw["graphs"]:
        _clean_raw_meta(graph)
        for node in graph["nodes"]:
            _clean_raw_meta(node)
        for edge in graph["edges"]:
            _clean_raw_meta(edge)
        graph["nodes"] = [node for node in graph["nodes"] if "type" in node]
    return graph_document_raw


def _clean_raw_meta(element):
    meta = element.get("meta")
    if not meta:
        return
    basic_property_values = meta.get("basicPropertyValues")
    if basic_property_values:
        meta["basicPropertyValues"] = [
            basic_property_value
            for basic_property_value in basic_property_values
            if basic_property_value.get("pred") and basic_property_value.get("val")
        ]

    definition = meta.get("definition")
    if definition is not None and not definition.get("val"):
        del meta["definition"]

    xrefs = meta.get("xrefs")
    if xrefs:
        meta["xrefs"] = [xref for xref in xrefs if xref.get("val")]

    # What's the point of a synonym with an empty value? Nothing!
    synonyms = meta.get("synonyms")
    if synonyms:
        meta["synonyms"] = [synonym for synonym in synonyms if synonym.get("val")]


#: Prefixes that denote remote resources
PROTOCOLS = {
    "https://",
    "http://",
    "ftp://",
    "ftps://",
}


def _is_remote(url: str | Path) -> bool:
    return isinstance(url, str) and any(url.startswith(protocol) for protocol in PROTOCOLS)


@contextmanager
def _path_context(path: None | str | Path, name: str = "output.json"):
    if path is not None:
        yield Path(path).resolve()
    else:
        with tempfile.TemporaryDirectory() as directory:
            yield Path(directory).joinpath(name)


def convert(
    input_path: str | Path,
    output_path: str | Path,
    input_flag: Literal["-i", "-I"] | None = None,
    *,
    merge: bool = True,
    fmt: str | None = None,
    check: bool = True,
    reason: bool = False,
    extra_args: list[str] | None = None,
    debug: bool = False,
) -> str:
    """Convert an OBO file to an OWL file with ROBOT.

    :param input_path: Either a local file path or IRI. If a local file path
        is used, pass ``"-i"`` to ``flag``. If an IRI is used, pass ``"-I"``
        to ``flag``.
    :param output_path: The local file path to save the converted ontology to.
        Will infer format from the extension, otherwise, use the ``fmt`` param.
    :param input_flag: The flag to denote if the file is local or remote.
        Tries to infer from input string if none is given
    :param merge: Use ROBOT's merge command to squash all graphs together
    :param fmt: Explicitly set the format
    :param check:
        By default, the OBO writer strictly enforces
        `document structure rules <http://owlcollab.github.io/oboformat/doc/obo-syntax.html#4>`.
        If an ontology violates these, the convert to OBO operation will fail.
        These checks can be ignored by setting this to false.
    :param reason:
        Turn on ontology reasoning
    :param extra_args:
        Extra positional arguments to pass in the command line
    :param debug:
        Turn on -vvv
    :return: Output from standard out from running ROBOT
    """
    if input_flag is None:
        input_flag = "-I" if _is_remote(input_path) else "-i"

    args: list[str] = []

    if merge and not reason:
        args.extend(["merge", str(input_flag), str(input_path), "convert"])
    elif merge and reason:
        args.extend(
            [
                "merge",
                str(input_flag),
                str(input_path),
                "reason",
                "convert",
            ]
        )
    elif not merge and reason:
        args.extend(
            [
                "reason",
                str(input_flag),
                str(input_path),
                "convert",
            ]
        )
    else:
        args.extend(
            [
                "convert",
                str(input_flag),
                str(input_path),
            ]
        )

    args.extend(("-o", str(output_path)))
    if extra_args:
        args.extend(extra_args)
    if not check:
        args.append("--check=false")
    if fmt:
        args.extend(("--format", fmt))
    if debug:
        args.append("-vvv")

    return call_robot(args)


def write_getter_warnings(path: str | Path) -> None:
    """Write warned unparsable."""
    path = Path(path).resolve()
    path.write_text("\n".join(GETTER_MESSAGES))
