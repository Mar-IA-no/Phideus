# HANDOFF → Claude UNC (Mendieta): EN N-adapt partial rerun

> Handoff de Claude LOCAL → Claude UNC. Tarea: reentrenar SOLO el régimen N-adapt de
> Voz Expresiva Fase 1 EN, con el fix B2 del calib_manifest, en Mendieta.
> Generado 2026-06-27. Código en `main` commit `6149d92`.

## Por qué

El cierre original de Fase 1 EN (commit `bc34c12`, `data/voz_expresiva/1/`) tenía un bug en
`build_calib_manifest`: reinstanciaba `RandomState(42)` dentro del loop de speakers → como los
10 speakers comparten inventario en el mismo orden, `rng.choice` elegía **las mismas 25 utts de
calibración para los 10 speakers**. Esto afecta SOLO el régimen **N-adapt** (que usa las 25 utts
de calibración per-speaker). El régimen **N-strict** (primario, el claim fuerte) NO usa
calibración → NO fue afectado.

**Estrategia B-partial** (consensuada con Codex): N-strict se hereda intacto del cierre original;
solo se reentrena N-adapt con el fix. Esto preserva el N-strict primario hardware-limpio (3090) y
ahorra ~3.3 h vs un rerun completo.

**Fix B2** (ya en `main`): `_speaker_calib_seed(spk) = sha256(f"42:{spk}")` → 25 utts
independientes por speaker. El manifest es determinístico (solo selección de índices), así que el
SET de calibración será idéntico al que produciría local — la diferencia de hardware solo afecta
el training de los pesos.

## Caveat de hardware (declarado)

ZH y el N-strict EN corrieron en la RTX 3090 local. Este N-adapt corre en A30 (Mendieta). Eso
mete una diferencia de hardware en el contraste cross-language **secundario** (N-adapt). El
**primario (N-strict) queda limpio** (EN N-strict heredado de 3090 vs ZH N-strict 3090). Anotar
en el reporte. (Alternativa hardware-100%-limpia: correr local; se eligió Mendieta para liberar
la GPU local para el frente Atención Armónica.)

## Qué necesita UNC

1. **Código**: `git pull origin main` (commit `6149d92`: fix B2 + `--limit-norms` + guardrails).
2. **Caches WavLM EN + descriptor EN** — para que N-adapt use las MISMAS features que el
   N-strict heredado:
   - **Recomendado**: transferir desde local/raid1
     `data/voz_expresiva/wavlm_cache/` (21.7 GB: `wavlm_features.npy` + `wavlm_lengths.npy` +
     `wavlm_index.json`) y `data/voz_expresiva/descriptors_cache/` (243 MB: `family_A.npy`).
   - **Fallback**: regenerar en Mendieta con
     `1_precache_wavlm.py --language EN` + `1_precache_descriptors.py` (necesita ESD EN +
     `microsoft/wavlm-large`). El forward de WavLM frozen es ~determinístico → features ~idénticas.
   - NO se necesitan los artefactos N-strict de `1/` en Mendieta (el merge se hace LOCAL).

## Comando del rerun (UNC arma el sbatch — Claude LOCAL NO escribe SLURM)

```bash
python experiments/voz_expresiva/1_train.py \
    --cache-root <RUTA_A>/data/voz_expresiva \
    --output-dir <RUTA_SALIDA>/1_en_calibfix \
    --epochs 30 --batch-size 64 --device cuda \
    --limit-norms adapt
```

- Auto-detecta los 10 speakers EN del índice (`get_speaker_pool`).
- Hiperparámetros = defaults, idénticos al cierre original (AdamW lr=1e-3, wd=1e-4, batch 64,
  30 epochs, cosine + warmup 1 epoch, sin early-stopping).
- `--limit-norms adapt` → SOLO N-adapt: 4 configs × 10 folds × 3 seeds = **120 runs**, ~3.3 h en A30.
- Recursos sbatch sugeridos: 1 GPU, ~4 h wall. Si los caches están transferidos, no hace falta precache.

## Qué devolver a LOCAL

Todo el directorio `1_en_calibfix/`:
- `uar_results.json` (120 records adapt, con `calib_seed`, `calib_seed_effective`, `calib_hash`).
- `embeddings/` (`fold{N}_{config}_adapt_seed{S}.npy`).
- `predictions/` (`fold{N}_{config}_adapt_seed{S}.json`).
- `calib_manifest.json` (con el fix B2, `calib_seed_effective` por speaker).

## Qué hace LOCAL después (NO UNC)

1. Crear `data/voz_expresiva/1_en_calibfix/` local; copiar N-strict de `1/` (records + embeddings
   + predictions) + el N-adapt que devuelve UNC.
2. Merge `uar_results.json`: N-strict heredado (con `calib_seed_effective=None`) + N-adapt nuevo.
3. Escribir `PROVENANCE.md` (N-strict 3090 heredado / N-adapt A30 Mendieta).
4. Reportes: `1_report.py --results-dir 1_en_calibfix --label-self EN` (consolida EN), luego
   `1_report.py --results-dir 1_zh --compare-against 1_en_calibfix --label-self ZH --label-other EN`.

## Verificación al recibir de UNC

- 120 records adapt; cada uno con `norm_condition="adapt"`, `calib_seed_effective` no nulo.
- `calib_manifest.json`: las 25 utts por speaker DIFIEREN entre speakers (fix B2 — no las mismas).
- Distribución de emociones de las 25 calib: NO balanceada (label-agnostic).
