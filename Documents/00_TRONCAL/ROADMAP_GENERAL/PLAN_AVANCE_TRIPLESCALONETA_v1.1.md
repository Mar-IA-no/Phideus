# Plan Maestro De Avance Triplescaloneta (v1.1)

> [!IMPORTANT]
> **Documento operativo vigente** para el avance de Triplescaloneta.
> `v1.0` queda archivado solo para trazabilidad historica.

## Resumen
Esta version v1.1 mantiene el objetivo del plan v1.0 y agrega criterios causales y de falsabilidad mas estrictos para evitar sesgos de interpretacion, variables fantasma y avance por inercia.

Meta central: cerrar Escalon 1-C con evidencia canonica, y avanzar a Escalon 2 y 3 con un marco metodologico comun, comparable y auditable.

---

## Estado de Partida (2026-02-12)
1. Escalon 1-C (BIAS_CONTROL) sigue activo con `Bloque A` en ejecucion.
2. Diagnostico Gate 6 + Gate 4.2 cerrado y documentado.
3. Se confirma que el avance a escalones siguientes depende de evidencia canonicamente comparable.

Referencias operativas:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md`
- `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`

---

## 1) Hipotesis Formales

### H3a (Escalon 1: Audio<->MIDI)
Un descriptor relacional basado en ratios puede capturar estructura cross-modal entre audio y MIDI de manera estable bajo protocolo canonico.

### H3b (Escalon 2: Speech<->EGG)
La representacion relacional puede transferirse a dos sensores fisicos distintos del mismo fenomeno vocal, superando baseline lineal.

### H3c (Escalon 3: ECG<->PPG)
La formulacion relacional, adaptada a eventos temporales, puede alinear dominios fisiologicos de distinta naturaleza de medicion.

### H3-meta (Generalidad)
Existe un nucleo de invariantes relacionales compartidos entre escalones, con componentes especificos por dominio/sensor.

---

## 2) Decision Gate F1->F2 (Salida De Cada Escalon)

Definicion metrica canonica primaria:
- `S = min(A2M@10, M2A@10)`

Definicion de calidad de discriminacion:
- `hard_neg = accuracy_vs_same_piece` (o analogo equivalente segun dominio)

Control:
- `S_control`, `hard_neg_control` deben venir del run control reproducible del mismo escalon.

| Salida | Criterio Minimo | Accion |
|---|---|---|
| SCALE (pilot) | `S_run - S_control >= +1.5pp` y `hard_neg_run >= hard_neg_control - 1pp`, con CI estable | Escalar piloto del escalon (fase siguiente acotada) |
| GO fuerte | `SCALE` + `S_run >= S_floor_escalon` | Escalar completo del escalon |
| INCONCLUSO | Senal debil/inestable o CI solapada con control | Repetir con ajuste metodologico acotado |
| DROP | `S_run < 30%` o `hard_neg_run < 75%` o degradacion estructural persistente | Cerrar linea y abrir post-mortem tecnico |

Nota: los umbrales numericos pueden adaptarse por escalon solo si queda documentado el racional y se mantiene comparabilidad interna.
Nota de precedencia: cuando exista plan operativo del escalon, su definicion de `S_floor_escalon` y criterios locales tiene precedencia (ejemplo actual Escalon 1-C: `S_floor_escalon=33.4%`).

---

## 3) Kill Switch v2 (Programa Falsable)

Se activa freeze de rama experimental cuando se cumplan simultaneamente:
1. Dos NO-GO consecutivos en el mismo escalon.
2. Protocolo canonico completo (pool/config/seeds) aplicado en ambos intentos.
3. Baseline lineal y control causal incluidos.
4. Sin leakage de identidad/split.
5. Sin alertas anti-variable-fantasma sin resolver.

Si falta cualquiera de esos prerequisitos, no aplica kill switch: primero se corrige metodologia.

Salida obligatoria al activar kill switch:
- `DecisionGateRecord` + post-mortem tecnico + propuesta de pivote o cierre.

---

## 4) Contratos Metodologicos (Schema + Instancias)

### 4.1 Schema comun (propuesto)
- `Documents/00_TRONCAL/ROADMAP_GENERAL/contracts/schema/ExperimentContract.yaml`
- `Documents/00_TRONCAL/ROADMAP_GENERAL/contracts/schema/EvaluationBundle.schema.json`
- `Documents/00_TRONCAL/ROADMAP_GENERAL/contracts/schema/DecisionGateRecord.template.md`

### 4.2 Campos minimos

#### ExperimentContract (YAML)
- `escalon_id`
- `dominio_origen`
- `dominio_destino`
- `dataset`
- `split_policy`
- `descriptor_version`
- `training_regime`
- `eval_protocol_id`
- `seed_policy`
- `loss_function`
- `freeze_policy`
- `anti_ghost_checks`
- `baseline_comparison`

#### EvaluationBundle (JSON)
- `pool_config`
- `retrieval_metrics`
- `hard_negative_metrics`
- `ci_bootstrap`
- `decision_flags`
- `run_metadata`

#### DecisionGateRecord (Markdown)
- `hipotesis`
- `resultado`
- `go_no_go`
- `riesgos`
- `next_action`

### 4.3 Primera instancia (template operativo)
- `Documents/00_TRONCAL/ROADMAP_GENERAL/contracts/instances/escalon_1c_bloqueA/`

Objetivo: que lo ejecutado en Escalon 1-C quede como plantilla reutilizable para Escalon 2 y 3.

---

## 5) Checklist Pre-Escalon 2 (Speech<->EGG)

Antes de abrir entrenamiento largo de Escalon 2:
1. Confirmar split por speaker (no leakage por identidad).
2. Definir pool de evaluacion canonico para el escalon (tamano, seeds, hard negatives).
3. Ejecutar baseline lineal (CCA/ridge o equivalente) y registrar resultado.
4. Ejecutar run control reproducible (`S_control`, `hard_neg_control`).
5. Pasar preflight anti-variable-fantasma (trainables, optimizer coverage, drift epoch 1).
6. Validar bundle de evaluacion con CI bootstrap.

Sin estos seis puntos, Escalon 2 no abre fase de escala.

---

## 6) Scope y ETA (Plan Operativo)

| Fase | Objetivo | Entregable | ETA referencia |
|---|---|---|---|
| F1 | Cerrar Escalon 1-C | Decision formal + checkpoint ganador + bundle canonico | Corto plazo (run en curso) |
| F2 | Consolidar infraestructura comun | Contratos + checklist + trazabilidad automatizable | 1 ciclo tecnico |
| F3 | Ejecutar Escalon 2 | Resultado GO/NO-GO con baseline lineal y CI | 1-2 ciclos |
| F4 | Ejecutar Escalon 3 | Resultado GO/NO-GO equivalente | 1-2 ciclos |
| F5 | Sintesis transversal | Informe de invariantes vs especificidades | Cierre de ola |

Regla de alcance:
- No abrir VibeTensor spike hasta cerrar F1 del frente activo.
- No abrir Escalon 3 si Escalon 2 no tiene protocolo canonico estabilizado.

---

## 7) Changelog v1.0 -> v1.1

| Cambio | Razon | Impacto |
|---|---|---|
| Se agrega gate F1->F2 con 4 salidas | Evitar avance por inercia | Decisiones mas binarias y auditables |
| Se formaliza H3-meta | Hacer falsable la generalidad inter-escalon | Mejora lectura cientifica transversal |
| Se incorpora Kill Switch v2 | Evitar programa infalsable | Control de costo y foco experimental |
| Se instancian contratos schema+instance (incluyendo campos de loss/freeze/anti-ghost/baseline) | Estandarizar evidencia y comparabilidad | Menor deuda metodologica entre escalones |
| Se alinea semantica `SCALE`/`GO fuerte` con plan operativo de Escalon 1-C | Evitar conflicto terminologico entre plan troncal y plan operativo | Consistencia de decision entre documentos |
| Se define checklist pre-Escalon 2 | Reducir errores de protocolo antes de escalar | Mejor robustez causal |
| Se explicita metrica canonica `S=min(A2M@10,M2A@10)` | Evitar ambiguedad entre documentos | Consistencia operacional |

---

## Supuestos y Defaults
1. Modo collab se activa solo por orden explicita del usuario.
2. Baseline operativo actual de Escalon 1 se mantiene en Gate 2 hasta cierre formal del Bloque A.
3. Resultados historicos de Gate 4.x se usan como control diagnostico, no como baseline primario.
4. Cualquier ajuste de umbral debe quedar versionado en este documento.

---

Actualizado: 2026-02-12 (v1.1, redaccion ejecutada por Codex a pedido del usuario)
