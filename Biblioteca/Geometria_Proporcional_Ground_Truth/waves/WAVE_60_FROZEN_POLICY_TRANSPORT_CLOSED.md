# Ola 60 — cierre del transporte de políticas congeladas entre draws

> **Estado:** `COMPLETE / FROZEN-POLICY-TRANSPORT / EXACT-SCIENTIFIC-REPLAY / AUDIT-PASS / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Plan científico:** `WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md`
> **Attempt sellado:** `data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4/`
> **Corrección de replay:** `WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json`
> **Auditoría de resultado:** `../agent_reports/509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md`
> **Auditoría final de la corrección:** `../agent_reports/531_wave60_v4_replay_normalization_correction_audit.md`

## Pregunta resuelta

La Ola 59 había probado dos políticas HGB/HGB sobre un draw fresco. Ambas
mejoraban algunas métricas frente a la decisión dura, pero ninguna podía
distinguirse de cinco controles que igualaban el desplazamiento condicional
máximo. La Ola 60 cambió el estimando: transportó la ley completa de Ola 59 a
otra realización independiente, sin refit, recalibración, reselección ni cambio
de thresholds.

La pregunta era si una regla contextual conserva fuera de su draw de ajuste una
ventaja frente a `hard` y, al mismo tiempo, frente a pipelines de control
congeladas. El primer contraste mide eficacia local. El segundo intenta atribuir
esa eficacia a algo más específico que la cobertura, la localización o la
magnitud de la acción.

## Ejecución e integridad

El attempt v4 contiene 65 archivos en primary, 66 en replay y 10 en el par:
141 archivos regulares, 141 inodos distintos, `nlink=1`, sin symlinks ni nodos
especiales. Los manifests cubren 138 archivos y los tres self-manifests cierran
el inventario. La ejecución terminó con ambos roots
`EVALUATED_IMMUTABLE`, terminal pair-level `COMPLETE`, truth ya accedida y
recuperación deshabilitada.

La recomposición independiente de R509 y R531 verificó las 14 acciones, 56
arrays métricos, 301 pair tokens primarios, 5.000 bootstraps, 18 contrastes y 12
soportes. Los ocho hashes científicos son idénticos entre primary y replay. El
`analysis.json` común tiene SHA-256
`f1378cb22f45e10580c18d3f0d5d12d8b6a4ed19d39cea6fd612bf0121bdb5e0`.
La corrida fue CPU-only y no usó ni consultó GPU.

## Resultado prospectivo

Las dos políticas transportaron una mejora completa frente a `hard`: los ocho
intervalos de las cuatro métricas primarias quedaron en la dirección favorable.

| Política congelada | Métrica frente a `hard` | Diferencia media | IC95 |
|---|---|---:|---:|
| incompatibility / media | accuracy | +0.008998 | [+0.000138, +0.017996] |
| incompatibility / media | compatibilidad | +0.014396 | [+0.007198, +0.022425] |
| incompatibility / media | regret | -0.018411 | [-0.028344, -0.009159] |
| incompatibility / media | worst regret | -0.022702 | [-0.044581, -0.002769] |
| harm / cola | accuracy | +0.008721 | [+0.001246, +0.016334] |
| harm / cola | compatibilidad | +0.010520 | [+0.003876, +0.017996] |
| harm / cola | regret | -0.014904 | [-0.023844, -0.006344] |
| harm / cola | worst regret | -0.017996 | [-0.031838, -0.005260] |

El soporte autorizado fue de 46 pair tokens para incompatibility y 35 para
harm. La separación causal exigida, sin embargo, volvió a fallar:

- incompatibility menos la media de sus cinco controles matched, en regret:
  `-0.0006229`, IC95 `[-0.0031285, +0.0015481]`;
- harm menos la media de sus cinco controles matched, en worst regret:
  `-0.0016611`, IC95 `[-0.0140642, +0.0094684]`.

Los dos intervalos incluyen cero. Por eso los patrones terminales permanecen
`incompatibility=false` y `harm=false`.

## Corrección de la autoridad de replay

El comparador histórico publicó `MISMATCH`: 35 de 36 checks verdaderos y una
única diferencia en `operational_semantic:preparation_receipt.json`. No era una
divergencia científica. Cada root ligaba correctamente su propio generation
receipt; esos receipts tenían el mismo contenido salvo el rol deliberado
`execution_mode=recovery` frente a `execution_mode=replay`, y por ello hashes
locales distintos. Los dos `preparation_freeze.json` eran byte-exactos.

El attempt ya era post-truth e inmutable. La resolución no lo reescribió:
construyó fuera de él un artefacto correctivo autenticado, lo publicó mediante
un enlace atómico sin sobrescritura y sometió tanto el publicador como el JSON a
auditorías independientes. R531 cerró `PASS 0/0/0` y activó la vista condicional
normalizada `36/36`. La observación histórica `MISMATCH 35/36` permanece como
registro; la lectura vigente añade que el replay científico fue exacto.

## Lectura

**Observación.** La mejora frente a `hard` transportó a una realización nueva
sin ajustar la ley, y fue más nítida que en Ola 59. Los controles de igual
desplazamiento continuaron dentro de una banda compatible con el efecto
principal.

**Hipótesis.** La pipeline congelada contiene una regularidad transportable,
pero el diseño actual no permite localizarla específicamente en el target HGB,
la representación contextual o la regla del guard. Cobertura, localización y
magnitud de acción siguen siendo explicaciones compatibles.

**Inferencia.** La Ola 60 refuerza un resultado negativo de atribución y cierra
el bracket HGB/HGB vigente como continuación prioritaria. No demuestra que la
política sea inerte, no refuta toda señal contextual y no establece un techo
para arquitecturas proposer/guard. Repetir el mismo roster sobre más draws
añadiría precisión a un estimando que ya mezcla representación y decisión; el
próximo contraste debe separarlas por diseño.

El alcance sigue limitado al generador sintético, la ley Wave 59 congelada y
los intervalos condicionales sin corrección de multiplicidad. No se adquiere
autoridad física. `scientific_decision` permanece `null`,
`decision_authority=user` y `architecture_promoted=false`.
