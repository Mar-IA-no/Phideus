# PLAN_EJECUCION_POST_DEC005_CODEX

Ultima actualizacion: 2026-02-11
Estado: operativo (pre-ejecucion)
Front: BIAS_CONTROL

> [!NOTE]
> Addendum de vigencia (2026-02-17): este plan v1.0 queda como antecedente histórico.
> Bloque A y Gate 4.3 ya cerraron; el frente transiciona a Gate 4.4.
> Versión operativa actual: `PLAN_EJECUCION_POST_DEC005_v1.1.md` + roadmap vigente.

## 1) Objetivo operativo inmediato
Cerrar BIAS_CONTROL con evidencia causal util para decision de roadmap, evitando iteraciones largas sin señal.

Preguntas que este plan resuelve:
1. La via adapter/unfreezing controlado recupera A2M sin perder M2A y hard-neg?
2. Conviene abrir Gate 4.2 (teorico-controlado) o cerrar rama ratios?
3. Como dejamos un paquete visual y generativo para comparacion humana (audio/MIDI + embeddings)?

## 2) Estado de partida (baseline de decision)
Baseline oficial actual: Gate 2 epoch45 (structured pool 256/500/seed42)
- A2M R@10: 34.4%
- M2A R@10: 37.6%
- hard_neg_acc: 80.4%

Lectura DEC-005 (diagnostico):
- Fine-tuning posterior tendio a degradar A2M.
- El problema no invalida la hipotesis de ratios en general; invalida variantes concretas testeadas hasta ahora.
- Se prioriza control de estabilidad y causalidad antes de nueva exploracion amplia.

## 3) Ruta principal (prioridad alta)
### Bloque A — Adapter/Unfreezing controlado + S-control
Objetivo: recuperar o superar baseline en metrica balanceada sin degradacion fuerte por direccion.

Experimentos A/B/C (runs cortos y comparables):
1. Run A: Adapter-only
- Unfreeze: solo adapter(s) nuevo(s) + proyecciones finales
- Audio encoder principal: frozen
- Duracion sugerida: 5 epocas

2. Run B: Partial unfreeze audio
- Unfreeze: ultimos bloques del audio encoder (profundidad acotada)
- Sin adapter extra
- Duracion sugerida: 5 epocas

3. Run C: Hibrido
- Unfreeze: adapter + ultimos bloques audio
- Duracion sugerida: 5 epocas

Criterios minimos para seguir a 15-30 epocas:
- S = min(A2M R@10, M2A R@10) >= baseline - 1.0pp
- hard_neg_acc >= baseline - 1.0pp
- sin deriva asimetrica grave (|A2M - M2A| no se dispara)

Si un run cumple umbral, escala a fase larga; si no, se cierra ese subcamino.

## 4) Ruta secundaria (prioridad media)
### Bloque B — Gate 4.2 como ejercicio teorico controlado
Solo si Bloque A queda inconcluso o si aparece una hipotesis ratio fuerte y acotada.

Reglas de Gate 4.2:
1. Maximo 1-2 variantes de descriptor (no abanico grande).
2. Cada variante con hipotesis causal explicita y umbral predefinido.
3. Misma comparabilidad estricta (structured pool canonico).
4. Clausula anti-goalpost: si no alcanza umbral, se cierra 4.2.

## 5) Paquete visual + generativo (prioridad media-alta)
Objetivo: habilitar comparacion humana de modelos para inspeccion cualitativa y comunicacion del estado.

### C1. Visualizaciones embeddings (estandarizadas)
Artefactos bajo `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/` (duplicado curado aprobado por usuario):
- UMAP multigate
- Bridges cross-modal
- Heatmaps similitud
- Hubness distributions
- Snapshot ejecutivo DEC005

### C2. Generaciones/Resintesis para escucha comparativa
Objetivo: comparar Gate2 vs Gate4/4.1 y futuros A/B/C en ejemplos pareados.

Plan tecnico minimo:
1. Seleccionar subset fijo de segmentos (seed fija).
2. Recuperar top-k retrievals por modelo (A2M y M2A).
3. Exportar paquete de escucha:
- audio query
- audio retrieved top1/topk
- MIDI GT y MIDI retrieved (renderizado a audio)
4. Guardar manifiesto JSON con ids, scores y rutas.

Criterio de utilidad:
- Permite detectar errores que metricas agregadas esconden (desfase temporal, timbre, confusiones recurrentes).

## 6) Regimen de comparabilidad (obligatorio)
No se acepta resultado fuera de este protocolo:
- pool_size=256
- n_queries=500
- seed=42
- mismo split
- mismas metricas primarias

Metricas primarias:
- A2M R@10
- M2A R@10
- hard_neg_acc
- S=min(A2M, M2A)

Metricas secundarias:
- MRR A2M/M2A
- mean rank
- separacion/bridge distance (diagnostico)

## 7) Entregables por ciclo
Por cada run:
1. JSON de resultados structured pool.
2. Resumen corto en docs curados (snapshot + indice visual).
3. Decision explicita: KEEP / SCALE / DROP.

Al cierre del bloque:
1. Tabla comparativa final contra Gate2.
2. Recomendacion de continuidad (seguir, cerrar rama, o redisenar).

## 8) Orden de ejecucion recomendado
1. Cerrar documentalmente estado DEC005 + curaduria visual actual.
2. Ejecutar Bloque A (A/B/C cortos) con evaluacion estructurada inmediata.
3. Si hay ganador claro, escalar solo ese.
4. Si no hay ganador, evaluar Gate 4.2 acotado.
5. Consolidar paquete visual+generativo y decision de roadmap.

## 9) Riesgos y mitigaciones
1. Riesgo: seguir iterando sin señal causal.
- Mitigacion: umbrales ex-ante + anti-goalpost.

2. Riesgo: ruido por falta de comparabilidad.
- Mitigacion: protocolo canonico fijo.

3. Riesgo: sobre-optimizar una direccion (M2A) perdiendo A2M.
- Mitigacion: usar S=min() como gate principal.

## 10) Nota de gobernanza
- Collab mode: OFF salvo activacion explicita del usuario.
- Este plan no habilita entrenamientos automaticamente; define la secuencia y los criterios.
