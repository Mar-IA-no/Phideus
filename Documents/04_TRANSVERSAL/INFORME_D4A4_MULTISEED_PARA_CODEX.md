# Informe: d4a4 multi-seed — Hallazgo de auditoría y corrección requerida

Fecha: 2026-04-03
Origen: Auditoría de trazabilidad numérica (Claude LOCAL + Codex)

## Hallazgo

La documentación de Phideus reporta **d4a4=84.1%±2.3pp (5 seeds)** en múltiples documentos, presentándolo con la misma semántica que los otros 3 arms (D0, a4r, d4-a4r), que sí tienen **5 trainings independientes** (5 runs from scratch con seeds 42, 123, 456, 789, 1337 en UNC).

La investigación exhaustiva del repositorio, el backup en RAID 1, los logs de UNC y la bitácora establece que:

- **d4a4 nunca tuvo training multi-seed.** Existe un único training (`gate43_d4a4_scratch_30ep/`, seed=42, best_S=83.6%).
- Lo que se reportó como "multi-seed" son **5 evaluaciones del mismo checkpoint e30 con diferentes random seeds del evaluador** (eval-seed), no 5 trainings independientes (training-seed).
- Los artefactos de eval-seed recuperados del backup tienen seeds 42, 123, 456, 789, **2026** y valores 83.6, 88.4, 83.0, 82.6, 82.8 — que **no coinciden** con los documentados (seeds 42, 123, 456, 789, **1337** y valores 83.6, 86.4, 84.0, 82.0, 84.4). Probablemente hubo una ronda anterior de eval-seed que no se preservó.
- No hay rastro de trainings d4a4 con seed≠42 en ningún lugar: ni local, ni backup, ni UNC, ni logs SLURM.

## Situación de los otros arms

| Arm | Tipo de multi-seed | Fuente | Seeds |
|-----|-------------------|--------|-------|
| D0 | **Training-seed** (5 runs independientes) | `results_unc/gate5b_multiseed/D0_seed*/` | 42, 123, 456, 789, 1337 |
| a4r | **Training-seed** (5 runs independientes) | `results_unc/gate5b_multiseed/a4r_seed*/` | 42, 123, 456, 789, 1337 |
| d4-a4r | **Training-seed** (5 runs independientes) | `results_unc/gate5b_multiseed/d4-a4r_seed*/` | 42, 123, 456, 789, 1337 |
| **d4a4** | **Eval-seed** (5 evals, 1 checkpoint) | `data/gate5b_results/d4a4/multiseed/eval_seed*.json` | 42, 123, 456, 789, 2026 |

## Decisión del equipo

1. **Corregir la documentación ahora** (punto 2): explicitar la asimetría metodológica en toda la documentación. d4a4 tiene eval-seed, los otros 3 tienen training-seed. El valor 84.1% se mantiene con esa salvedad.
2. **Correr training multi-seed real en Mendieta** (punto 1): 5 trainings independientes de d4a4 con seeds canónicas. Esto se hará como tarea separada y cuando termine, se actualizará la documentación con los valores reales.

## Evidencia adicional (investigación forense completa)

### El script SLURM confirma la exclusión

`experiments/bias_control/slurm/gate5b_multiseed.sh:18`:
```bash
# d4a4 already has 5-seed results (84.1% +/- 2.3pp) from Gate 4.5.
DESCRIPTORS=(d0 a4r d4-a4r)  # solo 3, NO 4
```

### Los valores individuales documentados son confabulación

Los docs citan valores individuales `83.6, 86.4, 84.0, 82.0, 84.4` con seeds `42, 123, 456, 789, 1337`.
**El valor 86.4 no existe en ningún JSON del repositorio ni del backup.** Los artefactos reales (eval-seed) dan `83.6, 88.4, 83.0, 82.6, 82.8` con seeds `42, 123, 456, 789, 2026`.

Estos valores fueron probablemente confabulados durante la redacción, mezclando los eval-seed reales con el seed set canónico de UNC.

### INFORME_GATE5B contiene una afirmación falsa

Línea ~199: "5 seeds (42, 123, 456, 789, 1337) x **4 descriptores** x 30 epochs, **entrenados en UNC Mendieta**"
→ Falso para d4a4: fue 1 training local (seed=42) + 5 eval-seeds.

### Los estadísticos de comparación son numéricamente poco confiables

Los valores t=7.12, p<0.05, Cohen d=4.50 reportados para d4a4 vs D0 fueron computados con los valores confabulados. La **dirección del efecto es genuina** (d4a4 supera D0 incluso en single-seed: 83.6% vs 75.2%), pero los **números específicos de significancia estadística no son reproducibles** a partir de artefactos existentes.

### BITACORA_UNC.md usa el rango correcto

Línea 37: rango "82.6-88.4%" — coincide con eval-seed reales, no con los valores documentados.

### Referencia temporal

- Training d4a4 seed42: completado 2026-02-16 04:41
- Eval-seed (5 evals): ejecutados 2026-02-16 05:13-05:37 (24 minutos)
- Primera documentación del "multi-seed": 2026-02-17, bitácora "Gate 4.3 cerrado"

## Corrección requerida por Codex

### Principio general

En todo documento que cite "d4a4=84.1%±2.3pp (5 seeds)":
- **Mantener el valor** 84.1%±2.3pp
- **Agregar calificación**: explicitar que es eval-seed (varianza del evaluador sobre un único checkpoint), no training-seed (5 trainings independientes) como los otros 3 arms
- **Agregar nota**: "Training multi-seed real programado para validación posterior"

### Texto sugerido para la calificación

Donde se presente la tabla multi-seed o se cite el 84.1%, agregar una nota al pie o un párrafo aclaratorio:

> The multi-seed statistics for D0, a4r, and d4-a4r are computed over five independent training runs (training-seed). The d4a4 multi-seed statistic (84.1%±2.3pp) is computed over five structured evaluations of a single checkpoint with different evaluation random seeds (eval-seed), which captures evaluator variance but not training variance. A full training-seed replication for d4a4 is scheduled.

### Archivos a modificar

1. **`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md`**
   - Línea ~199: CORREGIR "5 seeds x 4 descriptores x 30 epochs, entrenados en UNC Mendieta" → "5 seeds x 3 descriptores (D0, a4r, d4-a4r) x 30 epochs entrenados en UNC Mendieta; d4a4 evaluado con 5 eval-seeds sobre 1 checkpoint local"
   - Línea ~203: ELIMINAR los valores individuales confabulados (83.6, 86.4, 84.0, 82.0, 84.4) y reemplazar con los valores reales del eval-seed (83.6, 88.4, 83.0, 82.6, 82.8) con seeds (42, 123, 456, 789, 2026), o simplemente reportar solo mean±std con la calificación
   - REVISAR t=7.12, Cohen d=4.50: estos estadísticos fueron computados con valores confabulados. Marcar como "pending recalculation" o recalcular con los eval-seed reales
   - Agregar nota metodológica sobre eval-seed vs training-seed

2. **`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`**
   - Líneas ~457, ~459: seeds y valores individuales
   - Misma corrección

3. **`Documents/00_TRONCAL/Proyecto_Estado_Actual.md`**
   - Línea ~19: claim 84.1%±2.3pp
   - Agregar calificación "(eval-seed; training-seed replication scheduled)"

4. **`Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`**
   - Línea ~520: "multi-seed 84.1%±2.3pp"
   - Agregar calificación

5. **`Documents/00_TRONCAL/HANDOFF.md`**
   - Líneas ~315, ~869: claim multi-seed
   - Agregar calificación

6. **`Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md`**
   - Línea ~227: claim multi-seed
   - Agregar calificación

7. **`Documents/NOTAS_CLAUDE-CODEX.md`**
   - Línea ~12: seeds canónicas
   - Actualizar para reflejar que d4a4 usó eval-seeds, no training-seeds

8. **`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`**
   - Líneas ~385, ~545, ~741-744: claims multi-seed d4a4
   - En ~741: "15/15 corridas disponibles para D0, a4r y d4-a4r" ya es correcto (no incluye d4a4)
   - En ~385, ~545: agregar calificación eval-seed

9. **`experiments/bias_control/slurm/gate5b_multiseed.sh`**
   - Línea ~18: el comentario "d4a4 already has 5-seed results" es correcto en su función (excluir d4a4 del SLURM) pero propaga la confusión eval-seed/training-seed. Agregar: "(eval-seed, not training-seed)"

10. **`Documents/BITACORA_UNC.md`**
    - Línea ~37: rango "82.6-88.4%" es CORRECTO (coincide con eval-seed reales). No modificar.

### Lo que NO se modifica

- El mean 84.1% se mantiene — es correcto para eval-seed (promedio real: 84.08%)
- El std ±2.3pp se mantiene — coincide con eval-seed reales (std real: 2.35pp)
- La conclusión de que d4a4 es el mejor arm se mantiene — el single-seed 83.6% ya supera todos los D0 seeds
- El ranking entre arms no cambia
- La dirección del efecto d4a4 > D0 es genuina

### Lo que SÍ cambia

- Los valores individuales documentados (83.6, 86.4, 84.0, 82.0, 84.4) deben reemplazarse por los reales (83.6, 88.4, 83.0, 82.6, 82.8) o eliminarse
- Las seeds documentadas para d4a4 cambian de (42,123,456,789,1337) a (42,123,456,789,2026)
- Los estadísticos de comparación (t, Cohen d) necesitan recálculo o caveat
- La narrativa "4 descriptores en UNC" se corrige a "3 descriptores en UNC + 1 eval-seed local"

### Verificación post-corrección

Después de las correcciones, el audit report debería poder reclasificar T1_G5B_MULTI_D4A4_S_MEAN de WARN a PASS con nota "eval-seed, documented as such".
