import copy
from typing import Dict

# import requests


def get_schema(version: str) -> Dict:

    # Possible strategy for loading schemas, but probably want
    # to package schema with the release.
    # url = (
    #     "https://raw.githubusercontent.com/ome/ngff/main/"
    #     "0.3/schemas/json_schema/image.schema"
    # )
    # r = requests.get(url)
    # return r.json()

    # For now, embed the schemas below and simply return the corrent one

    if version == "0.3":
        return image_schema_3
    elif version == "0.1":
        return image_schema_1
    else:
        raise ValueError(f"Version {version} not supported")


def get_strict_schema(version: str) -> Dict:

    if version == "0.3":
        return merge(copy.deepcopy(image_schema_3), image_strict_3)
    elif version == "0.1":
        return merge(copy.deepcopy(image_schema_1), image_strict_1)
    else:
        raise ValueError(f"Version {version} not supported")


def merge(destination: Dict, source: Dict) -> Dict:
    """
    deep merge of source into destination dict
    https://stackoverflow.com/questions/20656135/python-deep-merge-dictionary-data
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge(node, value)
        else:
            destination[key] = value

    return destination


image_strict_1 = {
    "properties": {
        "multiscales": {
            "items": {"required": ["version", "name", "type", "metadata", "datasets"]}
        }
    }
}

image_strict_3 = {
    "properties": {
        "multiscales": {
            "items": {
                "required": ["version", "name", "type", "axes", "metadata", "datasets"]
            }
        }
    }
}

image_schema_1 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://localhost:8000/image.schema",
    "title": "NGFF Image",
    "description": "JSON from OME-NGFF .zattrs",
    "type": "object",
    "properties": {
        "multiscales": {
            "description": "The multiscale datasets for this image",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "datasets": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                    "version": {"type": "string", "enum": ["0.1"]},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string"},
                            "version": {"type": "string"},
                        },
                    },
                },
                "required": ["datasets"],
            },
            "minItems": 1,
            "uniqueItems": True,
        },
        "omero": {
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "window": {
                                "type": "object",
                                "properties": {
                                    "end": {"type": "number"},
                                    "max": {"type": "number"},
                                    "min": {"type": "number"},
                                    "start": {"type": "number"},
                                },
                                "required": ["start", "min", "end", "max"],
                            },
                            "label": {"type": "string"},
                            "family": {"type": "string"},
                            "color": {"type": "string"},
                            "active": {"type": "boolean"},
                        },
                        "required": ["window", "color"],
                    },
                }
            },
            "required": ["channels"],
        },
    },
    "required": ["multiscales"],
}


image_schema_3 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://localhost:8000/image.schema",
    "title": "NGFF Image",
    "description": "JSON from OME-NGFF .zattrs",
    "type": "object",
    "properties": {
        "multiscales": {
            "description": "The multiscale datasets for this image",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "datasets": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                    "version": {"type": "string", "enum": ["0.3"]},
                    "axes": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string", "pattern": "^[xyzct]$"},
                    },
                },
                "required": ["datasets", "axes"],
            },
            "minItems": 1,
            "uniqueItems": True,
        },
        "omero": {
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "window": {
                                "type": "object",
                                "properties": {
                                    "end": {"type": "number"},
                                    "max": {"type": "number"},
                                    "min": {"type": "number"},
                                    "start": {"type": "number"},
                                },
                                "required": ["start", "min", "end", "max"],
                            },
                            "label": {"type": "string"},
                            "family": {"type": "string"},
                            "color": {"type": "string"},
                            "active": {"type": "boolean"},
                        },
                        "required": ["window", "color"],
                    },
                }
            },
            "required": ["channels"],
        },
    },
    "required": ["multiscales"],
}
