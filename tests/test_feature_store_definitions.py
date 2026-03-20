from __future__ import annotations

import unittest

from feature_store import feature_definitions


class FeatureStoreDefinitionTests(unittest.TestCase):
    def test_file_sources_use_plain_filesystem_paths(self) -> None:
        self.assertTrue(feature_definitions.features_source.path.endswith("features.parquet"))
        self.assertTrue(feature_definitions.targets_source.path.endswith("targets.parquet"))
        self.assertFalse(feature_definitions.features_source.path.startswith("file:"))
        self.assertFalse(feature_definitions.targets_source.path.startswith("file:"))

