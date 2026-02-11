# Plan Maestro De Avance Trilescaloneta (v1.0)

> [!WARNING]
> **Documento archivado (no operativo).** Se conserva solo por trazabilidad historica.
> Fuente vigente: `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md`.

## Resumen
Este plan define como avanzar de forma rigurosa desde el estado actual de Escalon 1-C (BIAS_CONTROL) hacia Escalon 2 y Escalon 3, preservando comparabilidad cientifica y evitando repetir errores metodologicos (variables fantasma, metricas no canonicas, decisiones por senales ruidosas).
El objetivo no es copiar gates literalmente entre dominios, sino portar un marco causal comun con adaptaciones por modalidad.

## Objetivo y Criterio de Exito
1. Cerrar Escalon 1-C con evidencia canonica y decision formal reproducible.
2. Ejecutar Escalon 2 como primer test de generalidad inter-sensores (no musical) con protocolo equivalente.
3. Ejecutar Escalon 3 como test de generalidad fisiologica temporal.
4. Entregar una conclusion transversal: que parte del lenguaje de ratios es invariante entre dominios y que parte es especifica.
5. Exito global: al menos 2 escalones con senal cross-modal superior a baselines lineales bajo protocolos equivalentes.

## Estado Inicial Confirmado
1. Escalon 1-C activo en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`.
2. Diagnostico post Gate 4.1 completado en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`.
3. Plan operativo vigente de Bloque A en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`.
4. Marco conceptual triplescaloneta en `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`.

## Artefacto Documental a Crear
1. Crear documento central: `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.md`.
2. Actualizar enlace en `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`.
3. Registrar estado de foco en `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`.
4. Preparar version breve para debate con Claude en `COLLAB` cuando se active modo collab.

## Cambios/Interfaces Publicas (metodologicos y de datos)
1. Definir contrato experimental comun `ExperimentContract` en YAML:
   - `escalon_id`
   - `dominio_origen`
   - `dominio_destino`
   - `dataset`
   - `split_policy`
   - `descriptor_version`
   - `training_regime`
   - `eval_protocol_id`
2. Definir contrato de evaluacion `EvaluationBundle` en JSON:
   - `pool_config`
   - `retrieval_metrics`
   - `hard_negative_metrics`
   - `ci_bootstrap`
   - `decision_flags`
3. Definir contrato de decision `DecisionGateRecord` en Markdown:
   - `hipotesis`
   - `resultado`
   - `go_no_go`
   - `riesgos`
   - `next_action`

## Fase 1 - Cierre Formal Escalon 1-C
1. Ejecutar Bloque A completo bajo protocolo canonico homogeneo.
2. Fijar criterios de decision exclusivos por structured protocol.
3. Seleccionar checkpoint ganador por metrica balanceada canonica.
4. Emitir `DecisionGateRecord` de cierre de Escalon 1-C.
5. Marcar explicitamente que resultados historicos quedan como control diagnostico y no como baseline principal.

## Fase 2 - Infraestructura Comun Multi-Escalon
1. Disenar pipeline comun de evaluacion cross-modal para evitar reescrituras ad hoc.
2. Establecer seeds, splits por identidad y hard negatives comparables por dominio.
3. Estandarizar formato de outputs para auditoria cruzada.
4. Implementar checklist anti-variable-fantasma pre-run y post-epoch-1 en todos los escalones.

## Fase 3 - Escalon 2 (Speech<->EGG) en 3 Etapas
1. Etapa S2-P0 (sanidad de datos y alineacion temporal).
2. Etapa S2-P1 (baseline lineal y retrieval no profundo).
3. Etapa S2-P2 (descriptor de ratios + modelo contrastivo/anti-colapso).
4. Aplicar decision GO/NO-GO:
   - GO si supera baseline lineal con margen estable y CI no solapada.
   - NO-GO si no supera baseline o hay senales inestables no reproducibles.

## Fase 4 - Escalon 3 (ECG<->PPG) en 3 Etapas
1. Etapa S3-P0 (deteccion robusta de eventos y calidad de senal).
2. Etapa S3-P1 (baseline temporal relacional simple).
3. Etapa S3-P2 (descriptor ratio-temporal + alignment cross-modal).
4. Aplicar decision GO/NO-GO con mismos principios estadisticos que Escalon 2.

## Fase 5 - Sintesis Transversal Trilescaloneta
1. Construir tabla comparativa inter-escalon con metricas canonicas.
2. Identificar invariantes del lenguaje relacional y fallas especificas de dominio.
3. Decidir si Phideus se formula como:
   - hipotesis agnostica fuerte, o
   - hipotesis agnostica condicionada por familia de sensores.
4. Emitir roadmap siguiente (optimizacion o rediseno teorico).

## Orden de Ejecucion Recomendado
1. Cerrar Escalon 1-C primero.
2. Preparar infraestructura comun.
3. Ejecutar Escalon 2.
4. Ejecutar Escalon 3.
5. Hacer sintesis transversal.
6. Recien ahi definir expansion o pivote mayor.

## Casos de Prueba y Escenarios
1. Escenario A: Escalon 1 mejora pero Escalon 2 falla.
2. Escenario B: Escalon 2 y 3 muestran mejora sobre lineal.
3. Escenario C: mejora unidireccional con degradacion de equilibrio.
4. Escenario D: resultados sensibles a seed o split.
5. Escenario E: drift nulo en modulos supuestamente trainables.
6. Escenario F: ganancias aparentes por leakage de identidad.

## Criterios de Aceptacion por Escalon
1. Protocolos de evaluacion explicitos y reproducibles.
2. Baseline lineal documentado y superado cuando se declara GO.
3. Metricas con intervalos de confianza y hard negatives.
4. Registro anti-variable-fantasma completo.
5. Decision formal GO/NO-GO con trazabilidad documental.

## Riesgos Principales y Mitigaciones
1. Riesgo: extrapolar conclusiones de un dominio a otro sin adaptacion.
2. Mitigacion: mantener invariantes metodologicos y adaptar descriptores por dominio.
3. Riesgo: optimizar por metricas no comparables.
4. Mitigacion: contrato canonico de evaluacion y auditoria obligatoria.
5. Riesgo: deuda documental entre escalones.
6. Mitigacion: actualizacion troncales al cierre de cada etapa.

## Supuestos y Defaults
1. VibeTensor permanece pausado hasta cerrar la etapa activa de Escalon 1-C.
2. Baseline principal actual: Gate 2 epoch45 (Escalon 1).
3. Gate 4 historico se mantiene como control diagnostico, no como baseline primario.
4. Modo collab se activa solo por orden explicita del usuario.
5. Este turno en Plan Mode no ejecuta escritura; al salir de Plan Mode se crea el archivo en Documents.
