"""Command line interface for :mod:`bioontologies`.

Why does this file exist, and why not put this in ``__main__``?
You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m bioontologies`` python will execute``__main__.py`` as a script.
  That means there won't be any ``bioontologies.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``bioontologies.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/latest/setuptools/#setuptools-integration
"""

import json
import logging
from pathlib import Path

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
@click.option("--graph-id", help="The IRI of the graph to parse. Set this if it can't be guessed")
@click.option(
    "--save-obograph",
    is_flag=True,
    help="Save intermediate OBO Graph JSON file if conversion from OWL is required",
)
def index(prefix: str, graph_id: str | None, directory: Path | None, save_obograph: bool):
    """Generate a node index file."""
    from .robot import get_obograph_by_prefix

    prefix = bioregistry.normalize_prefix(prefix)

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
    summary_path = directory.joinpath("summary.json")

    df = graph.get_nodes_df(sep=" | ")
    df.to_csv(nodes_tsv_path, sep="\t", index=False)

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
