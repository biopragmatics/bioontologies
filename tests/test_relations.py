"""Test relationship name caching."""

import unittest

from bioontologies.relations import get_normalized_label


class TestRelations(unittest.TestCase):
    """Test relationship name caching."""

    def test_get_normalized_label(self):
        """Test getting a normalized label."""
        for expected, i in [
            (
                "proximally_connected_to",
                "http://purl.obolibrary.org/obo/uberon/core#proximally_connected_to",
            ),
            ("type", "rdf:type"),
            ("is_a", "rdfs:subClassOf"),
            ("may_be_identical_to", "iao:0006011"),
            ("has_part", "bfo:0000051"),
        ]:
            with self.subTest(input=i):
                self.assertEqual(expected, get_normalized_label(i))
