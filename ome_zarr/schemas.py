import json
import os
from typing import Dict

from jsonschema import RefResolver

SCHEMAS_PATH = "schemas"


class LocalRefResolver(RefResolver):
    def resolve_remote(self, url: str) -> Dict:
        # Use remote URL to generate local path
        rel_path = url.replace("https://ngff.openmicroscopy.org", SCHEMAS_PATH)
        curr_dir = os.path.dirname(__file__)
        path = os.path.join(curr_dir, rel_path)
        path = os.path.normpath(path)
        # Load local document and cache it
        document = load_json(path)
        self.store[url] = document
        return document


def load_json(path: str) -> Dict:
    with open(path) as f:
        document = json.loads(f.read())
    return document


def get_schema(version: str, strict: bool = False) -> Dict:
    schema_name = "strict_image.schema" if strict else "image.schema"
    curr_dir = os.path.dirname(__file__)
    # The paths here match the paths in the ngff repo (and public schemas)
    path = os.path.join(
        curr_dir, SCHEMAS_PATH, version, "schemas", "json_schema", schema_name
    )
    path = os.path.normpath(path)
    return load_json(path)
