"""Detached Ed25519 attestation helpers for the Wave 49 protocol.

The private key must remain outside both the repository and benchmark artifact.
Verification receives a separately trusted public key, so an artifact cannot
replace its truth, commitments, and trust root as one self-consistent bundle.
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .wave49_schema import canonical_json, sha256_bytes, sha256_file


class AttestationError(RuntimeError):
    """Raised when the detached signing contract is unavailable or invalid."""


def _run_openssl(args: list[str], payload: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["openssl", *args],
        input=payload,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AttestationError(f"OpenSSL attestation operation failed: {detail}")
    return completed.stdout


def public_key_der_from_private(private_key_path: Path) -> bytes:
    return _run_openssl([
        "pkey", "-in", str(private_key_path), "-pubout", "-outform", "DER",
    ])


def public_key_der(public_key_path: Path) -> bytes:
    return _run_openssl([
        "pkey", "-pubin", "-in", str(public_key_path), "-outform", "DER",
    ])


def trusted_public_key_fingerprint(public_key_path: Path) -> str:
    return sha256_bytes(public_key_der(public_key_path))


def sign_attestation(
    payload: dict[str, Any],
    private_key_path: Path,
    trusted_public_key_path: Path,
) -> dict[str, Any]:
    private_key_path = Path(private_key_path)
    trusted_public_key_path = Path(trusted_public_key_path)
    if not private_key_path.is_file():
        raise AttestationError(f"attestation private key missing: {private_key_path}")
    if not trusted_public_key_path.is_file():
        raise AttestationError(f"trusted attestation public key missing: {trusted_public_key_path}")
    derived = public_key_der_from_private(private_key_path)
    trusted = public_key_der(trusted_public_key_path)
    if derived != trusted:
        raise AttestationError("attestation private key does not match trusted public key")
    encoded = canonical_json(payload).encode("utf-8")
    with tempfile.NamedTemporaryFile(prefix="wave49-attestation-payload-") as payload_file:
        payload_file.write(encoded)
        payload_file.flush()
        signature = _run_openssl([
            "pkeyutl", "-sign", "-rawin", "-inkey", str(private_key_path),
            "-in", payload_file.name,
        ])
    return {
        "algorithm": "Ed25519",
        "payload": payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "trusted_public_key_sha256": sha256_bytes(trusted),
    }


def verify_attestation(receipt: dict[str, Any], trusted_public_key_path: Path) -> None:
    trusted_public_key_path = Path(trusted_public_key_path)
    if receipt.get("algorithm") != "Ed25519":
        raise AttestationError("unsupported semantic attestation algorithm")
    trusted = public_key_der(trusted_public_key_path)
    if receipt.get("trusted_public_key_sha256") != sha256_bytes(trusted):
        raise AttestationError("semantic attestation trust-root mismatch")
    try:
        signature = base64.b64decode(receipt["signature_base64"], validate=True)
    except (KeyError, ValueError) as exc:
        raise AttestationError("invalid semantic attestation signature encoding") from exc
    payload = canonical_json(receipt.get("payload")).encode("utf-8")
    with tempfile.NamedTemporaryFile(prefix="wave49-signature-") as signature_file, \
            tempfile.NamedTemporaryFile(prefix="wave49-attestation-payload-") as payload_file:
        signature_file.write(signature)
        signature_file.flush()
        payload_file.write(payload)
        payload_file.flush()
        _run_openssl([
            "pkeyutl", "-verify", "-rawin", "-pubin",
            "-inkey", str(trusted_public_key_path),
            "-sigfile", signature_file.name,
            "-in", payload_file.name,
        ])


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}
