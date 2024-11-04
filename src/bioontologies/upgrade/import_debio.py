"""Import rewrites from DeBiO."""

import requests

from bioontologies.upgrade import insert

URL = "https://raw.githubusercontent.com/biopragmatics/debio/main/releases/current/debio.json"
OBO_PURL = "http://purl.obolibrary.org/obo/"


def main():
    """Import rewrites from DeBiO."""
    for node in requests.get(URL, timeout=5).json()["graphs"][0]["nodes"]:
        if node.get("type") != "PROPERTY":
            continue
        identifier = node["id"].removeprefix("http://purl.obolibrary.org/obo/debio_")
        for xref_curie in node.get("meta", {}).get("xrefs", []):
            xref_prefix, xref_identifier = xref_curie["val"].split(":", 1)
            if xref_prefix != "obo":
                continue
            insert(f"{OBO_PURL}{xref_identifier}", "debio", identifier)


if __name__ == "__main__":
    main()
