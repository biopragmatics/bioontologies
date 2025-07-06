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
from bioontologies.robot import ROBOTError


@click.command()
def main() -> None:
    """Parse all OBO ontologies."""
    prefixes = [
        resource.prefix
        for resource in bioregistry.resources()
        if resource.get_download_owl() or resource.get_download_obograph()
    ]
    for prefix in tqdm(prefixes, desc="Parsing OWL ontologies", unit="ontology"):
        tqdm.write(click.style(f"\n[{prefix}]", fg="green"))
        with logging_redirect_tqdm():
            try:
                document = bioontologies.get_obograph_by_prefix(prefix, cache=False, reason=False)
            except ROBOTError as e:
                tqdm.write(click.style(f"[{prefix}] {e}", fg="red"))
                continue

            try:
                graph = document.squeeze(
                    standardize=True, prefix=prefix, tqdm_kwargs={"leave": False}
                )
            except ValueError as e:
                tqdm.write(click.style(f"[{prefix}] failed to parse\n\n{e}", fg="red"))
            else:
                if graph.version:
                    msg = f"[{prefix}] parsed {graph.title} - v{graph.version}"
                else:
                    msg = f"[{prefix}] parsed {graph.title}"

                tqdm.write(click.style(msg, fg="green"))


if __name__ == "__main__":
    main()
