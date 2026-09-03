"""Independent logical predicates over sealed Wave 49 structural truth."""

from __future__ import annotations


def logical_structural_set(family: str) -> list[str]:
    if family in {"PROP", "AFFINE_OFFSET", "POWER_NONUNIT", "SATURATING"}:
        return ["AFFINE_OFFSET", "PROP"] if family == "PROP" else [family]
    if family in {"PIECEWISE_AFFINE", "OFFSET_QUADRATIC"}:
        return []
    raise ValueError(f"Unknown logical family: {family}")


def logical_properties(family: str, params: dict[str, float], domain: tuple[float, float]) -> list[str]:
    """Evaluate exact properties without importing generator decisions."""
    lo, hi = domain
    if not (0 < lo < hi):
        raise ValueError("Wave 49 v1 expects a strictly positive domain")
    props: set[str] = set()
    if family == "PROP":
        props.update({"smooth", "origin_passing", "affine", "homogeneous_degree_1"})
        props.add("monotone_increasing" if params["k"] > 0 else "monotone_decreasing")
    elif family == "AFFINE_OFFSET":
        props.update({"smooth", "affine"})
        props.add("monotone_increasing" if params["b"] > 0 else "monotone_decreasing")
        if params["a"] == 0:
            props.update({"origin_passing", "homogeneous_degree_1"})
    elif family == "POWER_NONUNIT":
        props.update({"smooth", "origin_passing", f"homogeneous_degree_{params['p']:g}"})
        direction = params["a"] * params["p"]
        props.add("monotone_increasing" if direction > 0 else "monotone_decreasing")
    elif family == "SATURATING":
        props.update({"smooth", "origin_passing", "saturating", "monotone_increasing"})
    elif family == "PIECEWISE_AFFINE":
        if params["b1"] > 0 and params["b2"] > 0:
            props.add("monotone_increasing")
        elif params["b1"] < 0 and params["b2"] < 0:
            props.add("monotone_decreasing")
        if not (lo < params["knot"] < hi):
            props.add("affine")
    elif family == "OFFSET_QUADRATIC":
        props.add("smooth")
        derivative_lo = params["b"] + 2.0 * params["c"] * lo
        derivative_hi = params["b"] + 2.0 * params["c"] * hi
        if derivative_lo > 0 and derivative_hi > 0:
            props.add("monotone_increasing")
        elif derivative_lo < 0 and derivative_hi < 0:
            props.add("monotone_decreasing")
    else:
        raise ValueError(f"Unknown logical family: {family}")
    return sorted(props)
