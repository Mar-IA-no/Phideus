"""Lossless typed trees over the authenticated stage store; never pickle."""
from __future__ import annotations

import math

import numpy as np


def pack(value):
    arrays = {}
    def visit(node):
        if isinstance(node, np.generic):
            if node.dtype.kind not in "biuf" or not np.isfinite(node):
                raise ValueError("payload scalar must be finite and numeric")
            return ["numpy_scalar", [node.dtype.str, node.item()]]
        if isinstance(node, np.ndarray):
            if node.dtype.kind not in "biuf" or not np.isfinite(node).all():
                raise ValueError("payload arrays must be finite numeric/bool values")
            key = f"a{len(arrays):06d}"
            arrays[key] = node.copy()
            return ["array", key]
        if type(node) in (dict, list, tuple):
            if type(node) is dict:
                if any(type(k) not in (str, int) for k in node):
                    raise ValueError("payload mapping keys must be strings or integers")
                return ["dict", [[visit(k), visit(v)] for k, v in node.items()]]
            return ["tuple" if type(node) is tuple else "list", [visit(v) for v in node]]
        if node is None or type(node) in (str, bool, int, float):
            if type(node) is float and not math.isfinite(node):
                raise ValueError("nonfinite payload scalar")
            return ["scalar", node]
        raise ValueError(f"unsupported payload value: {type(node).__name__}")
    return {"schema": "measurement-typed-tree-v1", "tree": visit(value)}, arrays


def unpack(metadata, arrays):
    if set(metadata) != {"schema", "tree"} or metadata["schema"] != "measurement-typed-tree-v1":
        raise ValueError("invalid typed payload schema")
    used = set()
    def visit(node):
        if not isinstance(node, list) or len(node) != 2:
            raise ValueError("invalid typed payload node")
        kind, value = node
        if kind == "numpy_scalar" and isinstance(value, list) and len(value) == 2:
            dtype = np.dtype(value[0])
            if dtype.kind not in "biuf" or type(value[1]) not in (bool, int, float):
                raise ValueError("invalid numpy payload scalar")
            result = np.asarray(value[1], dtype=dtype)[()]
            if not np.isfinite(result):
                raise ValueError("nonfinite numpy payload scalar")
            return result
        if kind == "array":
            if value in used or value not in arrays:
                raise ValueError("duplicate or missing payload array")
            used.add(value)
            return arrays[value].copy()
        if kind == "scalar":
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("invalid payload scalar")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("nonfinite payload scalar")
            return value
        if kind in ("list", "tuple") and isinstance(value, list):
            result = [visit(v) for v in value]
            return tuple(result) if kind == "tuple" else result
        if kind == "dict" and isinstance(value, list):
            result = {}
            for k, v in value:
                key = visit(k)
                if type(key) not in (str, int) or key in result:
                    raise ValueError("invalid or duplicate payload mapping key")
                result[key] = visit(v)
            return result
        raise ValueError("unknown typed payload node")
    result = visit(metadata["tree"])
    if used != set(arrays) or any(a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("unexpected or nonfinite payload arrays")
    return result


def stage(store, folder, identity, *, produce, validate, check):
    """Inside a measured/admitted phase; reuse payloads before calling producer."""
    check()
    restored = store.completed(folder, identity)
    if restored is None:
        result = produce()
        check()
        validate(result)
        metadata, arrays = pack(result)
        store.publish_stage(folder, identity, metadata, arrays)
        restored = store.completed(folder, identity)
    value = unpack(restored[1], restored[2])
    validate(value)
    check()
    return restored[0], value
