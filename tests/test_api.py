# -*- coding: utf-8 -*-

"""Test API functions."""

import unittest

from bioontologies.robot import ParseResults, convert_to_obograph_remote


class TestAPI(unittest.TestCase):
    """Test parsing a remote file."""

    def test_parse_owl(self):
        """Test parsing a remote JSON graph, should take less than a minute."""
        uri = "https://raw.githubusercontent.com/pato-ontology/pato/master/pato.owl"
        result = convert_to_obograph_remote(uri)
        self.assertIsInstance(result, ParseResults)
