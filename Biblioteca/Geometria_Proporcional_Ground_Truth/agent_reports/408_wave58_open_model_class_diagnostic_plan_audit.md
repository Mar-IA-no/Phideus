## Auditoría independiente — Wave 58

**Veredicto: `REVISE`.**

El plan reconoce correctamente que los tres splits están abiertos, registra la adaptación conocida y limita cualquier resultado a generación de hipótesis. Sin embargo, todavía no define un experimento ejecutable de manera unívoca: inventario, targets, selectores multi-guard, tie-breaks y contrato HGB contienen vacíos que impedirían atribuir diferencias a la clase de modelo o reproducir la selección.

### Findings altos

1. **El candidate roster no es exhaustivo ni internamente consistente.**

   - Los probes abarcaron daño, compatibilidad, accuracy y cola ([plan:24-33](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)).
   - La lista de targets contiene cinco elementos, pero no `accuracy_loss` ([plan:75-83](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)).
   - `accuracy_loss` reaparece como guard lineal ([plan:93-99](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)).
   - `tail_breach` se define, pero no tiene modelo en el inventario. HGB sólo cubre `harm` y `posterior_incompatibility` ([plan:101-109](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)).
   - No se preservan specs/resultados crudos de los probes, por lo que no puede verificarse la afirmación «ninguna variante intentada será eliminada» ([plan:89-92](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)).

   **Corrección:** congelar una tabla canónica con un ID por candidato y columnas para proposer, guard(s), target, polaridad, `class_weight`, selector, grillas, seed y estado esperado. Incluir todos los brazos efectivamente inspeccionados o declarar que el ledger histórico es incompleto. Definir `accuracy_loss` o retirarlo, y decidir explícitamente si `tail_breach` tendrá modelos.

2. **La selección conjunta y las conjunciones de guards no están definidas.**

   El selector sólo enumera `proposer × guard` ([plan:121-136](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)), pero las comparaciones exigen guards «en conjunción» y un brazo con dos guards HGB ([plan:143-147](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)). Faltan:

   - semántica exacta de conjunción;
   - si cada threshold se calcula condicionado a las propuestas o a la autorización anterior;
   - producto cartesiano completo para dos guards;
   - `hard_only` y tratamiento de ausencia de candidatos factibles;
   - soporte mínimo del proposer;
   - orden total de tie-break, aunque los tests lo presuponen ([plan:178-180](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md));
   - tratamiento de cuantiles que producen thresholds duplicados.

   **Corrección:** definir formalmente máscaras, universo de cuantiles, grilla `q_proposer × q_guard1 × q_guard2`, AND/orden de aplicación, mínimos globales y por shard, terminales y clave total de desempate. Añadir además el desempate entre familias que produciría la candidata futura.

3. **La “reproducción exacta Wave 57” no es compatible con el contrato común de Wave 58.**

   Wave 58 agrega cuantiles `0.7` y `0.9` al guard ([plan:116-119](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)); Wave 57 usó `[0.1,…,0.6,0.8]` ([config Wave57:60-63](experiments/geometria_proporcional/configs/wave57_contextual_tail_guard_fresh.json)). Además, Wave 57 aplicaba sólo accuracy/compatibilidad al proposer ([worker:260-263](experiments/geometria_proporcional/_wave57_phase_worker.py)) y añadía worst regret recién al guard ([worker:319-323](experiments/geometria_proporcional/_wave57_phase_worker.py)). El plan Wave 58 presenta las tres restricciones como contrato común.

   **Corrección:** separar un brazo de replay legacy que consuma sin cambios la config Wave 57 y compruebe igualdad de thresholds, máscaras, acciones y métricas contra `selection_arrays.npz` y `result_arrays.npz`. El resto de Wave 58 puede usar la grilla ampliada, pero no llamarse reproducción exacta.

4. **La comparación Ridge/Logistic frente a HGB no mantiene congelado el estimando de ajuste.**

   Ridge y Logistic Wave 57 usaron `sample_weight=1/d_t` ([worker:108-115](experiments/geometria_proporcional/_wave57_phase_worker.py); [wave56_contextual_gate.py:180-187](src/geometria_proporcional/wave56_contextual_gate.py)). El plan no dice que HGB reciba esos mismos pesos. Tampoco congela el contrato completo de HGB: faltan `loss`, `max_depth`, `max_features`, `max_bins`, `categorical_features`, constraints, `tol`, `warm_start`, `class_weight`, versión sklearn y enlace explícito de las seeds a `random_state` ([plan:101-109](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)).

   Además, `class_weight="balanced"` multiplicado por `sample_weight` constituye otra función objetivo; no es una comparación pura de clase lineal contra no lineal.

   **Corrección:** exigir pesos token-wise idénticos en todas las celdas matched; congelar todos los kwargs efectivos y sklearn `1.8.0`; preservar estado/score reconstruible; tratar `class_weight` como factor separado; registrar fallos de una clase/no convergencia como candidatos intentados no evaluables.

5. **El uso adaptativo del monitor está bien declarado, pero la recomendación futura no tiene regla auditable.**

   HGB ya fue elegido después de inspeccionar el monitor ([plan:24-41](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)). Por tanto, ese monitor no puede arbitrar de nuevo entre familias como si fuera evaluación final. El plan promete recomendar una candidata si encuentra una «región Pareto favorable» ([plan:188-196](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)), pero no define Pareto, tolerancias, métricas obligatorias ni resolución de desacuerdo validation/monitor.

   **Corrección:** definir antes de implementar:

   - cuáles métricas/direcciones forman Pareto;
   - tolerancias y soporte;
   - regla exacta para nominar cero o una candidata;
   - qué papel tiene cada split;
   - etiqueta obligatoria `OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION`.

   “Atribución” debe limitarse a descomposición dentro de este draw, no causalidad generalizable.

6. **Targets y polaridades no alcanzan para implementación inequívoca.**

   `accuracy_loss` no tiene fórmula; `tail_breach` no especifica claramente si compara regret fila a fila con `max_p regret_hard(t,p)`; tampoco se fija la clase positiva ni si todos los guards autorizan por score bajo. La métrica terminal sí está bien definida por `action_metric_arrays`: promedio de 24 políticas por token y máximo intratoken para `worst_regret` ([wave55_policy_bridge.py:74-95](src/geometria_proporcional/wave55_policy_bridge.py)).

   **Corrección:** escribir fórmulas exactas para cada target usando `authorized_actions` y `constrained_regret`, su dominio `primary ∩ disagreement`, polaridad, epsilon y regla de autorización. Mantener explícita la mediación fila-modelo → máscara → acción → métrica token-wise.

### Findings medios

7. **Las ablaciones no garantizan una atribución factorial.**

   El 2×2 Ridge/HGB proposer × Logistic/HGB harm sólo está implícito y no se exige bajo ambos selectores. El brazo triple HGB no incluye su correspondiente `HGB gain + HGB posterior_incompatibility` sin harm, por lo que no aísla el aporte del segundo guard.

   **Corrección:** congelar el factorial 2×2 bajo selector secuencial y conjunto, más las ablaciones `harm-only`, `alternative-only` y `harm+alternative` para cada conjunción que se interprete.

8. **Replay e inmutabilidad necesitan un contrato más fuerte.**

   Los ocho hashes publicados coinciden con los archivos actuales. Sin embargo, `action_metric_arrays` depende transitivamente de `wave52_policy.py` ([wave55_policy_bridge.py:9-10](src/geometria_proporcional/wave55_policy_bridge.py)), cuyo hash no está congelado. “El script no modifica inputs” ([plan:167-168](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md)) es una intención, no una prueba.

   **Corrección:** congelar dependencias transitivas y código Wave 58; verificar SHA-256 pre/post; ejecutar con inputs read-only; hacer replay desde un output vacío con refit y reselección; enumerar JSON/NPZ deterministas y exclusiones operativas.

9. **La matriz de tests es insuficiente para los riesgos anteriores.**

   A [plan:170-182](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md) deben añadirse tests de:

   - fórmula/polaridad de todos los targets;
   - `sample_weight` y `class_weight` efectivos;
   - contrato completo HGB y reconstrucción de scores;
   - conjunción de guards y producto cartesiano;
   - thresholds duplicados, una sola clase y ninguna celda factible;
   - reproducción exacta Wave 57;
   - monitor inaccesible hasta congelar FIT/SELECT;
   - hashes pre/post e inputs read-only;
   - replay independiente, no copia de outputs;
   - límite efectivo de threads.

10. **Falta incertidumbre descriptiva.**

    Los shards del mismo draw no estiman variación entre realizaciones; R405 ya advierte ese límite ([R405:111-113](Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/405_wave57_contextual_tail_guard_execution_results_audit.md)). Para orientar un futuro protocolo conviene preservar deltas pareados e intervalos bootstrap por `pair_token`, claramente etiquetados como condicionales y adaptativos. No deben convertirse en confirmación post-selección.

### Recursos

La estimación de RAM es plausible: los cinco NPZ congelados suman aproximadamente `54.17 MiB` expandidos. No obstante, el runtime `<2 min` y el techo `<2 GiB` sólo podrán sostenerse después de cerrar el roster y decidir cuántas máscaras del producto cartesiano se materializan. Deben imponerse realmente cuatro threads mediante variables BLAS/OpenMP o `threadpoolctl`; declarar `cpu_threads=4` no limita por sí solo HGB.

### Aspectos que pasan

- El carácter abierto/adaptativo y la falta de autoridad prospectiva están declarados honestamente.
- Los hashes de las ocho fuentes listadas son correctos.
- La población, las 24 políticas, penalidad `1.25` y estimandos terminales coinciden con Wave 57.
- Validation y monitor se reportarán separados.
- Se preservan brazos desfavorables y referencias hard/oracle.
- CPU-only es apropiado; no hay justificación técnica para GPU.

En síntesis: Wave 58 puede convertirse en un diagnóstico útil para congelar una hipótesis futura, pero el texto actual todavía permitiría varias implementaciones materialmente diferentes y una nominación post-hoc. Debe revisarse antes de implementar.
