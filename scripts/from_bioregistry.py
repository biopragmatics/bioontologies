# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "bioontologies",
#     "bioregistry",
# ]
#
# [tool.uv.sources]
# bioontologies = { path = "../", editable = true }
# bioregistry = { path = "../../bioregistry", editable = true }
# ///

"""Parse all OWL / OBO Graph JSON files listed in Bioregistry."""

import bioregistry
import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import bioontologies


@click.command()
def main() -> None:
    """Parse all OBO ontologies."""
    prefixes = [
        resource.prefix
        for resource in bioregistry.resources()
        if resource.get_download_owl() or resource.get_download_obograph()
    ][282:]
    for prefix in tqdm(prefixes, desc="Parsing OWL ontologies", unit="ontology"):
        tqdm.write(click.style("\n" + prefix, fg="green"))
        with logging_redirect_tqdm():
            document = bioontologies.get_obograph_by_prefix(prefix, cache=False)
            try:
                graph = document.squeeze(standardize=True, prefix=prefix)
            except ValueError:
                tqdm.write(f"[{prefix}] failed to parse")
            else:
                tqdm.write(
                    click.style(f"[{prefix}] parsed {graph.title} - v{graph.version}", fg="red")
                )


if __name__ == "__main__":
    main()
