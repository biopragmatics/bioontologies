"""Constants."""

import bioregistry

CANONICAL = {
    "mamo": "http://identifiers.org/mamo",
    "swo": "http://www.ebi.ac.uk/swo/swo.json",
    "ito": "https://identifiers.org/ito:ontology",
    "apollosv": "http://purl.obolibrary.org/obo/apollo_sv.owl",
    "cheminf": "http://semanticchemistry.github.io/semanticchemistry/ontology/cheminf.owl",
    "dideo": "http://purl.obolibrary.org/obo/dideo/release/2022-06-14/dideo.owl",
    "micro": "http://purl.obolibrary.org/obo/MicrO.owl",
    "ogsf": "http://purl.obolibrary.org/obo/ogsf-merged.owl",
    "mfomd": "http://purl.obolibrary.org/obo/MF.owl",
    "one": "http://purl.obolibrary.org/obo/ONE",
    "ons": "https://raw.githubusercontent.com/enpadasi/Ontology-for-Nutritional-Studies/master/ons.owl",
    "ontie": "https://ontology.iedb.org/ontology/ontie.owl",
}

IRI_TO_PREFIX = {v: k for k, v in CANONICAL.items()}
for resource in bioregistry.resources():
    owl_iri = resource.get_download_owl()
    if owl_iri:
        IRI_TO_PREFIX[owl_iri] = resource.prefix
