# Auditoría objetiva — fallo post-monitor del replay Wave 59

**Dictamen: `PASS-CON-RIESGOS`**

El diagnóstico causal es correcto: el replay no muestra divergencia científica. Falló por un falso negativo del comparador operacional. Sin embargo, ese defecto apareció después de abrir validation y monitor; por el protocolo congelado no puede corregirse y reaplicarse al mismo draw para obtener una adjudicación prospectiva.

## Findings

### ALTO — El único fallo observado es un hash operacional derivado

Los recibos de generación primario y replay difieren únicamente en el campo legítimo `execution_mode`:

- primario `recovery`: `generation_receipt.json:3`;
- replay `replay`: `generation_receipt.json:3`.

El preparador incorpora ese modo al recibo en `prepare_wave56_fresh.py:3655-3671`. Después, cada `preparation_receipt.json` enlaza el SHA-256 bruto de su propio recibo de generación en `prepare_wave56_fresh.py:3746-3760`. Por eso aparecen hashes diferentes:

- primario `bb0b7d19...`: `wave59_fresh_hgb_guard_bracket_v1/preparation_receipt.json:10-14`;
- replay `50ba3691...`: `wave59_fresh_hgb_guard_bracket_v1_replay.failed_.../preparation_receipt.json:10-14`.

El error está en que `_normalize_operational()` elimina `execution_mode`, timestamp, duración, RSS y otros campos operacionales, pero no elimina ni normaliza `generation_receipt_sha256`: `run_wave59_hgb_guard_bracket.py:1777-1802`. Luego `compare_runs()` exige igualdad del recibo de preparación normalizado y convierte cualquier diferencia en fallo global: `run_wave59_hgb_guard_bracket.py:1875-1931`.

Recomputación independiente, sin abrir material secreto:

- `scientific_exact`: `22/22`, ninguna divergencia;
- `scientific_array_exact`: `21/21`, ninguna divergencia;
- generación normalizada: igualdad exacta;
- preparación normalizada: una sola diferencia, `generation_receipt_sha256`;
- invariantes de ambas preparation attestations: iguales y ambas firmas válidas;
- portable states: iguales; los 16 modelos satisfacen sus hashes internos en ambos runs;
- inventario del replay: sin faltantes, extras, overlaps ni paths sin clasificar.

Además, la comparación de preparación ya registró `28/28` checks exactos, incluidos escrow, freezes, benchmark, logits y cinco bundles: `preparation_replay.json:2-31`.

La implementación reconoce esta clase de diferencias en otro punto: al comparar preparation attestations elimina precisamente los records de los dos receipts y el journal de preparación, además de rol y modo: `run_wave59_hgb_guard_bracket.py:1805-1823`. El tratamiento del `preparation_receipt` quedó inconsistente con ese criterio.

**Conclusión:** bug del comparador operacional; no divergencia científica.

### ALTO — La reanudación hash-idéntica no puede cambiar el resultado

`restore_identical_hash_attempt()` copia todo el intento salvo metadatos de fallo: `run_wave59_hgb_guard_bracket.py:932-974`. Por tanto, restaura intacto el mismo `preparation_receipt.json` con el hash bruto divergente.

Las fases con journals existentes se reutilizan sin recalcularse: `run_wave59_hgb_guard_bracket.py:895-919`. La evaluación monitor ya promovida también se valida y reutiliza: `run_wave59_hgb_guard_bracket.py:2925-2939`. Finalmente se vuelve a invocar el mismo comparador: `run_wave59_hgb_guard_bracket.py:2984-2991`.

El archivo firmado de fallo confirma que el replay alcanzó `last_state=COMPLETE` y `maximum_truth_materialized=monitor`: `FAILURE.json:4-9`. La atestación del fallo es válida y enlaza correctamente inventario y registro: `failure_attestation.json:2-10`.

**Conclusión:** reanudar con hashes y código idénticos repetirá determinísticamente el mismo falso negativo. No aporta información nueva y no debe ejecutarse.

### ALTO — Un fix posterior no puede adjudicar este draw

El contrato es explícito:

- después de abrir truth validation o monitor, cualquier delta invalida el intento para adjudicación prospectiva y sólo un protocolo y draw nuevos pueden continuar: `WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md:330-347`;
- la config codifica `delta_after_validation_or_monitor_truth = INVALID-PROSPECTIVE-ATTEMPT`: `wave59_fresh_hgb_guard_bracket.json:229-233`;
- el recovery plan exige ejecutar ambos paquetes sin alterar código, config ni autoridad: `WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:481-488`.

La implementación también lo fuerza materialmente: cada fuente debe ser limpia e idéntica al blob de HEAD, `run_wave59_hgb_guard_bracket.py:351-443`, y la autoridad de recovery exige que HEAD siga siendo exactamente el commit de auditoría final, rechazando commits posteriores o worktree sucio: `prepare_wave56_fresh.py:1438-1475`.

Esto incluye un fix puramente operacional del comparador. Su inocuidad científica no crea una excepción al protocolo predeclarado.

### MEDIO — La prueba de replay no cubrió la cadena real de receipts

El test sintético construye placeholders genéricos para los artefactos de preparación que faltan: `tests/test_wave59_prospective.py:157-230`. El test de replay, además, sustituye la verificación real de invariantes de attestation: `tests/test_wave59_prospective.py:322-340`.

Así, no reprodujo el caso real:

```text
generation receipt recovery
        ↓ hash bruto distinto
preparation receipt primary

generation receipt replay
        ↓ hash bruto distinto
preparation receipt replay
```

Un protocolo futuro necesita un test realista `recovery ↔ replay` que conserve esta diferencia de modo y exija igualdad semántica.

## Estado científico admisible

Los artefactos actuales permiten afirmar como observación:

- ambas ejecuciones produjeron los mismos artefactos científicos públicos y arrays;
- la preparación fue exacta `28/28`;
- la divergencia operacional normalizada se reduce a un hash derivado;
- el replay formal permanece fallido.

No permiten marcar la condición prospectiva `replay_exact=true`. Los propios resultados conservan:

- incompatibility: `aggregate_without_replay=true`;
- harm: `aggregate_without_replay=false`;
- ambos: `replay_exact=PENDING` y `aggregate_with_replay=null`;

en `analysis.json:911-951`.

## Opciones conformes

1. **Recomendada ahora: preservar sin adjudicar.** Conservar íntegros el primario y el replay fallido, publicar una auditoría post hoc de equivalencia y mantener `aggregate_with_replay=null`. No editar receipts, analysis ni manifests.

2. **Nuevo protocolo y draw.** Corregir antes del freeze el comparador con normalización tipada: validar primero que cada hash bruto enlace su propio recibo, comparar luego los recibos de generación normalizados y reemplazar el hash derivado por un marcador semántico. Añadir pruebas realistas, auditar, congelar y recién entonces crear un draw nuevo.

3. **Checker independiente post hoc.** Puede certificar la igualdad observada de los artefactos existentes, pero sólo como evidencia diagnóstica secundaria. No transforma este intento en adjudicación prospectiva.

No son opciones conformes ni la reanudación hash-idéntica ni parchear el código y volver a correr sobre el mismo draw. No se declara `GO/NO-GO`.
