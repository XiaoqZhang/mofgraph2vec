"""Helpers for working with (nested) dictionaries."""
from typing import Any


def put(d: dict, keys: str, item: Any):
    """ Set a value in a nested dictionary.
    Taken from https://stackoverflow.com/questions/12414821/checking-a-nested-dictionary-using-a-dot-notation-string-a-b-c-d-e-automatica
    """
    if "." in keys:
        key, rest = keys.split(".", 1)
        if key not in d:
            d[key] = {}
        put(d[key], rest, item)
    else:
        d[keys] = item


def get(d: dict, keys: str):
    """Get a value from a nested dictionary.
    Taken from https://stackoverflow.com/questions/12414821/checking-a-nested-dictionary-using-a-dot-notation-string-a-b-c-d-e-automatica
    """
    if "." in keys:
        key, rest = keys.split(".", 1)
        return get(d[key], rest)
    else:
        return d[keys]