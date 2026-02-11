# VIBETENSOR_SPIKE_PLAN

Fecha: 2026-02-11  
Estado: pausado (rama experimental)  
Rama: `exp/vibetensor-spike`  
Worktree recomendado: `/tmp/phideus-vibetensor-spike`

---

## Objetivo

Evaluar integración **selectiva** de kernels de `vibe_kernels` para acelerar partes críticas de Phideus sin alterar el protocolo científico de BIAS_CONTROL.

Este plan **no** reemplaza el roadmap de Gates. Es una línea de infraestructura paralela.

> [!IMPORTANT]
> Reactivacion condicionada: este spike se retoma despues de cerrar `DEC-005` y la auditoria final de `BIAS_CONTROL` (Escalon 1-C).

---

## Alcance y límites

### Incluido

1. Mapeo operación-por-operación entre Phideus y `vibe_kernels`.
2. Microbenchmarks locales en hardware objetivo (RTX 3090).
3. Prototipos de integración puntual con rollback simple.

### Excluido (por ahora)

1. Port completo de Phideus a `vibetensor.torch`.
2. Reescritura de arquitectura de modelos (GRU/LSTM/Transformer completos).
3. Cambios que rompan comparabilidad de métricas históricas del roadmap.

---

## Hallazgos técnicos de partida (auditados)

1. `vibe_kernels` principal trabaja sobre `torch.Tensor` (camino práctico inmediato).
2. `vibetensor.torch.nn` tiene cobertura limitada para un port total.
3. La documentación de rendimiento de VibeTensor está sesgada a H100/Hopper; no extrapolar directo a 3090.
4. Plugin ABI existe y es usable, pero requiere esfuerzo C/CUDA adicional.

---

## Plan por fases

## Fase A — Mapeo de viabilidad (rápida)

Clasificar cada operación crítica de Phideus:
- `drop-in`
- `requiere wrapper/adaptación`
- `no viable hoy`

Objetivo: matriz cerrada de priorización técnica.

## Fase B — Benchmark local (gating)

Benchmark en RTX 3090 de candidatos iniciales:
- `attention`
- `softmax`
- `cross_entropy`
- `AdamW`

Gating sugerido para promoción:
- mejora >= 15% en throughput o step-time,
- sin degradación numérica relevante,
- sin inestabilidad de entrenamiento.

## Fase C — Integración piloto (1 o 2 piezas)

Integrar el/los candidatos que pasen Fase B detrás de flags:
- activación explícita por configuración,
- fallback automático al camino base.

## Fase D — Revisión de impacto

Validar:
- compatibilidad con pipeline actual,
- ausencia de regresiones en métricas de BIAS_CONTROL,
- costo de mantenimiento.

Si no hay mejora sólida, cerrar spike sin promoción.

---

## Reglas de gobernanza

1. `main` mantiene la línea científica y documental oficial.
2. El spike vive en `exp/vibetensor-spike`.
3. Sólo se hace merge/cherry-pick a `main` con evidencia cuantitativa local.
4. Toda recomendación debe incluir riesgo técnico + costo de mantenimiento.

---

## Entregables esperados

1. Matriz de viabilidad técnica por operación.
2. Tabla de benchmarks locales reproducibles.
3. Informe de riesgos y recomendación final (`promover` / `descartar` / `iterar`).
