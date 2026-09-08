"""Ephemeral, synchronous validation DAGs, never a cache across work stages.

Only small dependency results are memoized. Boundary functions join their
caller's pass or open a new one; producers must call them separately before
and after doing work. ``fresh_pass`` overrides an outer pass for internal
before/after checks. No decorator here belongs on an entire producer/trainer.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json


_current = ContextVar("learned_partition_validation_pass", default=None)


class ValidationPass:
    def __init__(self):
        self.results = {}
        self.active = set()
        self.references = {}
        self.claims = {}
        self.calls = {}
        self.hits = {}

    def digest(self, ref):
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("validation reference needs exact path and hash")
        path, digest = ref["path"], ref["sha256"]
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("invalid validation reference identity")
        if path in self.references and self.references[path] != digest:
            raise ValueError("conflicting hashes for one path in a validation pass")
        self.references[path] = digest

    def reference(self, ref, *, role, binding):
        self.digest(ref)
        path = ref["path"]
        claim = _key((role, binding))
        if path in self.claims and self.claims[path] != claim:
            raise ValueError("conflicting roles or bindings in a validation pass")
        self.claims[path] = claim


def _typed(value):
    kind = type(value)
    if kind is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("validation dictionaries require string keys")
        return ["dict", [[k, _typed(value[k])] for k in sorted(value)]]
    if kind in (list, tuple):
        return [kind.__name__, [_typed(v) for v in value]]
    if kind in (type(None), bool, int, float, str):
        return [kind.__name__, value]
    raise ValueError("unsupported validation key type")


def _key(value):
    return hashlib.sha256(json.dumps(_typed(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@contextmanager
def validation_pass(*, fresh=False):
    existing = _current.get()
    if existing is not None and not fresh:
        yield existing
        return
    session = ValidationPass()
    token = _current.set(session)
    try:
        yield session
    finally:
        _current.reset(token)
        # Counters remain inspectable, but no data survive the boundary.
        session.results.clear()
        session.active.clear()
        session.references.clear()
        session.claims.clear()


def boundary(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with validation_pass():
            return function(*args, **kwargs)
    return wrapped


def fresh_pass(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with validation_pass(fresh=True):
            return function(*args, **kwargs)
    return wrapped


def claim_reference(ref, *, role, binding):
    session = _current.get()
    if session is not None:
        session.reference(ref, role=role, binding=binding)


def claim_digest(ref):
    session = _current.get()
    if session is not None:
        session.digest(ref)


def memoized(function):
    """Type-preserving small results, copied so consumers cannot poison siblings.

    No pass means no memo. Exceptions are never cached; recursive entry into
    an unfinished node fails closed. Payload loaders intentionally do not use
    this decorator (snapshots are a bounded, explicitly scoped exception).
    """
    signature = inspect.signature(function)
    name = function.__module__ + "." + function.__qualname__

    @wraps(function)
    def wrapped(*args, **kwargs):
        session = _current.get()
        if session is None:
            return function(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        key = (name, _key(arguments.arguments))
        if key in session.active:
            raise ValueError("cyclic validation dependency")
        if key in session.results:
            session.hits[name] = session.hits.get(name, 0) + 1
            return deepcopy(session.results[key])
        session.active.add(key)
        session.calls[name] = session.calls.get(name, 0) + 1
        try:
            result = function(*args, **kwargs)
            session.results[key] = deepcopy(result)
            return result
        finally:
            session.active.remove(key)
    return wrapped
