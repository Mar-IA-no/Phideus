# Wave 49: relational-family benchmark

This directory contains the executable classical benchmark used to test whether
noisy observations support a proportional, affine-offset, non-unit power, or
saturating relation, a set of compatible families, or abstention outside the
catalogue. It is a synthetic protocol test, not evidence of a universal natural
geometry.

## Requirements

- the repository Python environment;
- a system `openssl` build with Ed25519 support;
- an Ed25519 private key kept outside the repository and run artifact;
- the tracked trusted public key in `keys/wave49_attestation_public.pem`.

Generate a local signing key only when establishing a new trust root:

```bash
mkdir -p "$HOME/.config/phideus"
openssl genpkey -algorithm ED25519 \
  -out "$HOME/.config/phideus/wave49_attestation_private.pem"
chmod 600 "$HOME/.config/phideus/wave49_attestation_private.pem"
openssl pkey \
  -in "$HOME/.config/phideus/wave49_attestation_private.pem" \
  -pubout -out experiments/geometria_proporcional/keys/wave49_attestation_public.pem
```

Never commit the private key. Replacing the tracked public key establishes a
different trust root and must be treated as a protocol change.

## Run

From the repository root:

```bash
source venv/bin/activate
python experiments/geometria_proporcional/run_wave49_classical.py all \
  --output-dir data/geometria_proporcional/wave49 \
  --attestation-private-key "$HOME/.config/phideus/wave49_attestation_private.pem"
```

Run a small end-to-end smoke with `--smoke`. Existing non-empty output is never
replaced unless `--force` is explicit.

Validate an existing artifact without regenerating it:

```bash
python experiments/geometria_proporcional/run_wave49_classical.py check \
  --output-dir data/geometria_proporcional/wave49
```

An exact replay can reuse the sealed generation, identity, and commitment keys
from an earlier artifact:

```bash
python experiments/geometria_proporcional/run_wave49_classical.py all \
  --output-dir data/geometria_proporcional/wave49-replay \
  --replay-secrets-from data/geometria_proporcional/wave49 \
  --attestation-private-key "$HOME/.config/phideus/wave49_attestation_private.pem"
```

The generated data are intentionally gitignored. Preserve the whole artifact,
including manifests, sealed truth, predictions, evaluations, mutations, source
snapshots, and attestation. The tracked public key is the trust source; a copy
inside a run artifact is lineage evidence only.

