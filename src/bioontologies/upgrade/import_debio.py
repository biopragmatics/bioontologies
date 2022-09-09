"""Import rewrites from DeBiO."""

import requests

from bioontologies.upgrade import insert

URL = "https://raw.githubusercontent.com/biopragmatics/debio/main/releases/current/debio.json"
OBO_PURL = "http://purl.obolibrary.org/obo/"


def main():
    """Import rewrites from DeBiO."""
    for node in requests.get(URL).json()["graphs"][0]["nodes"]:
        if node.get("type") != "PROPERTY":
            continue
        identifier = node["id"].removeprefix("http://purl.obolibrary.org/obo/debio_")
        for xref in node.get("meta", {}).get("xrefs", []):
            xp, xi = xref["val"].split(":", 1)
            if xp != "obo":
                continue
            insert(f"{OBO_PURL}{xi}", "debio", identifier)


if __name__ == "__main__":
    main()
