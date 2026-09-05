## Reauditoría Wave 58

**Veredicto: `REVISE`.**

La revisión resolvió sustancialmente el dictamen anterior, pero quedan dos findings altos de reproducibilidad y tres medios. No corresponde implementar todavía.

### Findings

1. **ALTO — El ledger histórico aún no es ejecutable de forma unívoca.**

   En `WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:112-122`:

   - `P1` usa la abreviatura ambigua `0.1..0.8`;
   - `P2` agrupa “normal y restricciones por shards” bajo los mismos IDs;
   - cada ID `P3` representa dos variantes de `class_weight`;
   - en `P4`, la clase del guard `I` no queda determinada en todos los IDs.

   Esto contradice la obligación de reproducir todos los IDs y conservarlos individualmente (`:119-122,266-268`).

   **Corrección:** expandir el ledger a una fila por corrida efectiva, con ID único, lista literal de cuantiles, clases de cada guard, `class_weight`, selector, restricciones y seeds. Si el transcript no permite reconstruir algo, marcarlo `HISTORICAL-SPEC-INCOMPLETE`, sin prometer reproducción exacta.

2. **ALTO — El source binding omite el propio contrato auditado y el replay no cubre todos los artefactos científicos.**

   Los hashes añadidos en `:54-68` son correctos, incluidas las dependencias transitivas inspeccionadas. Sin embargo:

   - el plan Wave 58 no aparece entre las fuentes ligadas;
   - el manifest futuro liga código/config/tests/commit, pero no el plan ni esta auditoría (`:298-302`);
   - el replay sólo exige igualdad de `config.json`, `analysis.json`, `REPORT.md` y `scores_and_masks.npz` (`:291-296`), aunque `fit/` y `select/` contienen estados, scores, grillas y freezes científicos obligatorios (`:282-284`).

   **Corrección:** después de la revisión final, congelar SHA-256 del plan y auditoría aceptada. Comparar en replay todos los artefactos científicos de FIT y SELECT, incluidos estados, scores, grillas y freezes; excluir únicamente `runtime.json` y campos operativos enumerados.

3. **MEDIO — `LEGACY-W57` está aislado, pero su recomputación puede ser tautológica.**

   `:190-194` dice que consume la config y los arrays publicados y debe igualarlos, pero no ordena explícitamente:

   1. reajustar Ridge/Logistic desde `gate_fit_bundle`;
   2. puntuar y seleccionar desde `gate_select_bundle`;
   3. evaluar desde `sealed_monitor_bundle`;
   4. usar `selection_arrays` y `result_arrays` sólo como referencias posteriores.

   Una implementación que simplemente relea los outputs satisfaría literalmente el texto sin reproducir Wave 57.

   **Corrección:** fijar esa secuencia y exigir igualdad de estados/scores además de thresholds, máscaras, acciones y métricas.

4. **MEDIO — Los kwargs están completos, pero la transformación de inputs y el estado HGB siguen subespecificados.**

   `:147-178` congela correctamente kwargs, versión y `sample_weight=1/d_t`. No obstante, sólo Logistic menciona escalado ponderado. Falta declarar si:

   - Ridge también usa el `WeightedScaler` Wave 57;
   - HGB recibe features crudas o escaladas;
   - el estado HGB se preserva mediante qué formato exacto para cumplir la reconstrucción de scores exigida en `:179-180,320`.

   **Corrección:** congelar transformación por familia y el schema/serializer del estado HGB, o declarar que los scores son la autoridad de reanálisis y limitar consecuentemente el claim de reconstrucción.

5. **MEDIO — El bootstrap descriptivo no está completamente congelado.**

   `:334-337` fija 5.000 réplicas y seed `5807`, pero no fija generador, orden canónico, percentiles ni que los mismos índices pareados se reutilicen entre todos los brazos y métricas de cada split.

   **Corrección:** especificar `PCG64`, orden lexicográfico de `pair_token`, intervalo percentil `[2.5,97.5]`, unidad primaria completa y matriz de índices compartida; añadir prueba de determinismo, rango y pairing.

### Verificación de los diez findings anteriores

| Finding anterior | Estado |
|---|---|
| 1. Roster/ledger | **Parcial** — factorial canónico resuelto; ledger histórico aún ambiguo |
| 2. Fórmulas/polaridades | **Resuelto** |
| 3. Selectores/tie-breaks | **Resuelto** |
| 4. `LEGACY-W57` aislado | **Parcial** — aislamiento correcto; recomputación no explícita |
| 5. Contratos y sample weights | **Parcial** — kwargs/pesos resueltos; transformación/estado HGB pendientes |
| 6. Factorial/ablaciones | **Resuelto** — `36` candidatos y ablaciones matched |
| 7. Pareto/nominación adaptativa | **Resuelto** |
| 8. FIT/SELECT/MONITOR | **Resuelto** — staging físico y allowlists adecuados |
| 9. Binding/inmutabilidad/replay | **Parcial** — inmutabilidad fuerte; binding y cobertura de replay incompletos |
| 10. Tests/incertidumbre/threads | **Parcial** — tests y threads adecuados; bootstrap incompleto |

No encontré leakage no declarado: el monitor está deliberadamente abierto, se accede después de los freezes y la nominación queda marcada `OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION`. Tampoco encontré reclamos prospectivos indebidos, uso de GPU ni un problema material de RAM.

El dictamen anterior quedó preservado verbatim en [408_wave58_open_model_class_diagnostic_plan_audit.md](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/408_wave58_open_model_class_diagnostic_plan_audit.md).

**SHA-256:** `6208be17437be6755794c17e0970203b572c7860bcce6415a3a140f88048c7be`

No modifiqué ningún otro archivo.
