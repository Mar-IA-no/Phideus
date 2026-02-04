#!/usr/bin/env python3
"""Audit: Verify all bias_control dependencies."""

import sys


def check_dependencies():
    """Check all required and optional dependencies."""
    results = {}

    # Critical dependencies
    critical = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("librosa", "librosa"),
        ("pretty_midi", "pretty_midi"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]

    for pkg, import_name in critical:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            results[pkg] = f"✅ OK (v{version})"
        except ImportError:
            results[pkg] = "❌ MISSING"

    # Optional dependencies
    optional = [
        ("sklearn", "sklearn"),
        ("umap", "umap"),
        ("matplotlib", "matplotlib"),
    ]

    for pkg, import_name in optional:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            results[pkg] = f"✅ OK (optional, v{version})"
        except ImportError:
            results[pkg] = "⚠️ Missing (optional)"

    return results


def main():
    print("=" * 60)
    print("BIAS_CONTROL DEPENDENCY AUDIT")
    print("=" * 60)

    results = check_dependencies()

    print("\nCritical Dependencies:")
    critical_ok = True
    for pkg in ["torch", "transformers", "librosa", "pretty_midi", "numpy", "tqdm"]:
        status = results.get(pkg, "❌ UNKNOWN")
        print(f"  {pkg:20s} {status}")
        if "MISSING" in status:
            critical_ok = False

    print("\nOptional Dependencies:")
    for pkg in ["sklearn", "umap", "matplotlib"]:
        status = results.get(pkg, "⚠️ Unknown")
        print(f"  {pkg:20s} {status}")

    print("\n" + "=" * 60)
    if critical_ok:
        print("STATUS: ✅ ALL CRITICAL DEPENDENCIES INSTALLED")
        return 0
    else:
        print("STATUS: ❌ MISSING CRITICAL DEPENDENCIES")
        return 1


if __name__ == "__main__":
    sys.exit(main())
