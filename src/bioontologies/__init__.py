# -*- coding: utf-8 -*-

"""Tools for biomedical ontologies."""

from .robot import convert_to_obograph, get_obograph_by_iri, get_obograph_by_prefix

__all__ = [
    "convert_to_obograph",
    "get_obograph_by_prefix",
    "get_obograph_by_iri",
]
