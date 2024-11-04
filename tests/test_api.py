"""Test API functions."""

import unittest

from bioontologies.robot import ParseResults, convert_to_obograph_remote, is_available


class TestAPI(unittest.TestCase):
    """Test parsing a remote file."""

    def test_robot_is_available(self):
        """Test ROBOT is available."""
        self.assertTrue(is_available())

    def test_parse_owl(self):
        """Test parsing a remote JSON graph, should take less than a minute."""
        uri = "https://raw.githubusercontent.com/pato-ontology/pato/master/pato.owl"
        result = convert_to_obograph_remote(uri)
        self.assertIsInstance(result, ParseResults)
        graph = result.squeeze()
        self.assertEqual("PATO - the Phenotype And Trait Ontology", graph.title)
        self.assertEqual("quality", graph.default_namespace)
        self.assertIn("http://purl.obolibrary.org/obo/PATO_0000001", graph.roots)
