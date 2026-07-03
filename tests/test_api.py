"""Test API functions."""

import unittest

from bioontologies.robot import ParseResults, convert_to_obograph_remote

from robot_obo_tool import is_available


@unittest.skipUnless(is_available(), "Robot is not available")
class TestAPI(unittest.TestCase):
    """Test parsing a remote file."""

    def test_parse_owl(self) -> None:
        """Test parsing a remote JSON graph, should take less than a minute."""
        uri = "https://raw.githubusercontent.com/pato-ontology/pato/master/pato.owl"
        result = convert_to_obograph_remote(uri)
        self.assertIsInstance(result, ParseResults)
        graph = result.squeeze(standardize=False)
        self.assertEqual("PATO - the Phenotype And Trait Ontology", graph.name)
        self.assertEqual("quality", graph.default_namespace)
        self.assertIn("http://purl.obolibrary.org/obo/PATO_0000001", graph.roots)
