"""Shared module stubs for the test suite.

These are unit tests over fakes: no Neo4j, Qdrant or VLM is required. The stubs
live here rather than in a test module because installing them into
`sys.modules` is a process-wide side effect — when a single test file did it,
whichever file imported first decided what every later file saw, and the
retriever tests broke only when the whole suite ran together.

`qdrant_client` is stubbed unconditionally: importing the real client pulls in
FastEmbed/ONNX native runtimes at collection time. The `models` namespace has to
stay faithful enough for the filter-building code under test, so each stub class
just records its keyword arguments as attributes — the same shape assertions
read (`filter.must[0].range.gte`).
"""
import sys
import types


class _Model:
    """Attribute bag standing in for a qdrant_client pydantic model."""

    _fields = ()

    def __init__(self, **kwargs):
        for field in self._fields:
            setattr(self, field, None)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        shown = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                          if v is not None)
        return f"{type(self).__name__}({shown})"


class Filter(_Model):
    _fields = ("must", "must_not", "should")


class FieldCondition(_Model):
    _fields = ("key", "match", "range")


class Range(_Model):
    _fields = ("gt", "gte", "lt", "lte")


class IsEmptyCondition(_Model):
    _fields = ("is_empty",)


class PayloadField(_Model):
    _fields = ("key",)


class MatchValue(_Model):
    _fields = ("value",)


class MatchAny(_Model):
    _fields = ("any",)


def _install_stubs():
    qdrant_stub = types.ModuleType("qdrant_client")
    qdrant_stub.QdrantClient = object
    qdrant_stub.models = types.SimpleNamespace(
        Filter=Filter, FieldCondition=FieldCondition, Range=Range,
        IsEmptyCondition=IsEmptyCondition, PayloadField=PayloadField,
        MatchValue=MatchValue, MatchAny=MatchAny)
    sys.modules["qdrant_client"] = qdrant_stub

    try:
        import neo4j  # noqa: F401
    except ModuleNotFoundError:
        neo4j_stub = types.ModuleType("neo4j")
        neo4j_stub.GraphDatabase = object()
        sys.modules["neo4j"] = neo4j_stub

    try:
        import dotenv  # noqa: F401
    except ModuleNotFoundError:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda: None
        sys.modules["dotenv"] = dotenv_stub


_install_stubs()
