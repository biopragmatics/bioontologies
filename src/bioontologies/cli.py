# -*- coding: utf-8 -*-

"""Command line interface for :mod:`bioontologies`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m bioontologies`` python will execute``__main__.py`` as a script.
  That means there won't be any ``bioontologies.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``bioontologies.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/latest/setuptools/#setuptools-integration
"""

import json
import logging
from operator import attrgetter
from pathlib import Path
from typing import Optional

import bioregistry
import click
import pystow

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """CLI for bioontologies."""


@main.command()
@click.argument("prefix")
@click.option("--directory", type=Path)
@click.option("--graph-id")
@click.option("--save-obograph", is_flag=True)
def index(prefix: str, graph_id: Optional[str], directory: Optional[Path], save_obograph: bool):
    """Generate a node index file."""
    from .robot import get_obograph_by_prefix

    prefix = bioregistry.normalize_prefix(prefix, strict=True)

    if directory is None:
        directory = pystow.join("bioontologies", "index", prefix)
    directory = Path(directory).expanduser().resolve()

    parse_results = get_obograph_by_prefix(
        prefix, json_path=directory.joinpath(prefix).with_suffix(".json") if save_obograph else None
    )
    if parse_results.graph_document is None:
        raise ValueError("missing graph document")

    elif len(parse_results.graph_document.graphs) == 1:
        graph_id = parse_results.graph_document.graphs[0].id
    elif len(parse_results.graph_document.graphs) > 1 and graph_id is None:
        guess = f"http://purl.obolibrary.org/obo/{prefix}.owl"
        if any(graph.id == guess for graph in parse_results.graph_document.graphs):
            graph_id = guess
        else:
            x = "\n".join(sorted(f"  {graph.id}" for graph in parse_results.graph_document.graphs))
            raise ValueError(f"Need to use --graph-id to specify one of:\n{x}")

    graph = next(
        graph for graph in parse_results.graph_document.graphs if graph.id == graph_id
    ).standardize(prefix=prefix)

    nodes_tsv_path = directory.joinpath("nodes.tsv")
    nodes_json_path = directory.joinpath("nodes.json")
    summary_path = directory.joinpath("summary.json")

    jv = {}
    header = (
        "curie",
        "label",
        "synonyms",
        "xrefs",
        "deprecated",
        "replaced_by",
        "alternative_ids",
        "namespace",
        "created_by",
        "creation_date",
    )
    with nodes_tsv_path.open("w") as file:
        print(  # noqa:T201
            *header,
            sep="\t",
            file=file,
        )
        for node in sorted(graph.nodes, key=attrgetter("id")):
            if node.prefix != prefix:
                continue
            jd = dict(
                uri=node.id,
                curie=node.curie,
                label=node.lbl,
                definition=node.definition,
                synonyms=[synonym.val for synonym in node.synonyms],
                xrefs=[xref.curie for xref in node.xrefs],
                deprecated=node.deprecated,
                replaced_by=node.replaced_by,
                alternative_ids=node.alternative_ids,
                namespace=node.namespace,
                created_by=node.created_by,
                creation_date=node.creation_date,
            )
            jv[node.luid] = {k: v for k, v in jd.items() if v}
            print(  # noqa:T201
                node.curie,
                node.lbl or "",
                " | ".join(synonym.val for synonym in node.synonyms),
                " | ".join(xref.curie for xref in node.xrefs),
                "true" if node.deprecated else "",
                node.replaced_by or "",
                " | ".join(node.alternative_ids),
                node.namespace or "",
                node.created_by or "",
                node.creation_date or "",
                sep="\t",
                file=file,
            )

    nodes_json_path.write_text(json.dumps(jv, indent=2, ensure_ascii=False, sort_keys=True))
    summary_path.write_text(
        json.dumps(
            {
                "prefix": prefix,
                "license": graph.license,
                "roots": graph.roots,
                "version": graph.version,
                "version_iri": graph.version_iri,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
            indent=2,
            sort_keys=True,
        )
    )

    click.echo(f"Wrote nodes index to {nodes_tsv_path}")


if __name__ == "__main__":
    main()
