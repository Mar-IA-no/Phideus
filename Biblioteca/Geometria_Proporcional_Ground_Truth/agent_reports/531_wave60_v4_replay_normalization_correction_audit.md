# R531 — Auditoría final del artefacto de normalización de replay de Ola 60 v4

## Dictamen técnico: PASS

R530 publica un candidato fiel al estado sellado de Ola 60 y a la cadena de
autoridad que resolvió R509. El artefacto conserva la observación histórica
`MISMATCH 35/36`, identifica como única diferencia el enlace operacional de
`preparation_receipt.json`, y deriva una vista condicional normalizada
`36/36` sin cambiar datos, métricas ni patrones científicos. La regeneración
autenticada desde R528/R529 produjo exactamente los mismos bytes que R530.

No encontré findings altos, medios ni bajos. Este `PASS` activa únicamente la
vista corregida condicional definida por R530: no reescribe el attempt, no
convierte el replay histórico en otro artefacto, no promueve una arquitectura
y no constituye una decisión `GO/NO-GO`.

## Identidad, genealogía y closed world

El target R530 es
`bc344ad9026b99c8c37ac008372d0789a15af1b4`, hijo directo único de R529
`d2519e235ccbb10050a62525aaf84e32a626c800`. Su diff añade exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json`.
El archivo físico tiene 24.849 bytes, es regular, `0444`, `nlink=1`, y su
SHA-256 físico y de blob Git es
`01600e6ae26c51f28a3485110fc8561fbd9d0b4fe4fb54849379f106d6aee802`.
La serialización coincide byte a byte con la salida canónica de `build`.

R529 añade solamente su informe como hijo directo de R528; su SHA-256 físico
y de blob es
`81baf07ca28c6d7972314eadd0473b317031a40612606581ee74b00b908f8b82`.
Su autoridad es `PASS 0/0/0` sobre el commit R528
`3cbe8f6f2fa90691b76caf5931cdfec6b1698ccc` y los dos hashes de
implementación declarados. El checker recompuso además R509–R527, sus parents,
pathsets, hashes y veredictos históricos sin reetiquetar los `REVISE` como
`PASS`.

El attempt preserva su closed world: 65 archivos en primary, 66 en replay y
10 en pair, para 141 archivos regulares, 141 inodos distintos y 141 enlaces
únicos. Los manifests cubren 138 archivos y los tres self-manifests completan
el inventario. Tamaño, owner, group, modo y SHA-256 coinciden con cada registro;
no hay symlinks ni nodos especiales. Los hashes target de pair, primary y
replay coinciden con R509 y R530. Ninguna comprobación modificó esos 141
archivos.

## Esquema y condición de activación

El JSON tiene el esquema cerrado
`wave60-v4-replay-normalization-correction-v1` y estado
`CANDIDATE_PENDING_R531_AUDIT`. Exige exactamente:

- auditoría `R531` en el path de este informe;
- scope `WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION`;
- veredicto `PASS` con findings `0/0/0`;
- target cuyo commit sea el parent directo de R531, cuyo path sea el propio
  output R530 y cuyo SHA-256 sea el del artefacto;
- efecto limitado a `ACTIVATES_CONDITIONAL_CORRECTED_VIEW`.

Los tres sentinels permanecen literales como
`DIRECT_PARENT_OF_R531`, `SELF_OUTPUT_RELATIVE` y
`SHA256_OF_THIS_ARTIFACT`. Por eso R530 no se autoactiva: en ese commit aún no
existe una autoridad R531 hija que pueda satisfacerlos. Este informe contiene
la autoridad exacta requerida y su commit exclusivo como hijo directo de R530
completa esa condición externa. La activación no altera el estado histórico
`MISMATCH`; autoriza leer, junto a él, la normalización adjudicada.

## Recomposición del mismatch operacional

La comparación publicada contiene 36 checks: 35 verdaderos y uno falso,
`operational_semantic:preparation_receipt.json`; `mismatches` contiene sólo
`operational:preparation_receipt.json` y el status histórico es `MISMATCH`.
No apareció una segunda diferencia científica, funcional, secreta u
operacional.

Los generation receipts primary/replay tienen SHA-256 locales distintos:

- primary/recovery:
  `847ec250397a459eeb25e4ca6bb10774551e40dbadceaeeac8cb27527c60be44`;
- replay/replay:
  `9088275ca41be5f82d14bb7c227eed36b443e080062786bc77dd8e38483e3e13`.

Cada `preparation_receipt.json` liga exactamente el receipt de su propio root.
Los dos generation receipts tienen el mismo keyset y contenido salvo
`execution_mode`, cuyos valores son deliberadamente `recovery` y `replay`.
Los dos `preparation_freeze.json` son byte-exactos, SHA-256
`ed47e512452ee13689427bca97166c323233c2a8825d89d18cb9c91b73efe6f9`,
y ambos enlaces locales al freeze son válidos. Al preservar los enlaces
locales y comparar entre roots la semántica allowlisted, el único falso pasa a
verdadero y la vista normalizada queda `36/36`.

Los ocho hashes científicos son idénticos entre primary y replay. En
particular, ambos `analysis.json` tienen SHA-256
`f1378cb22f45e10580c18d3f0d5d12d8b6a4ed19d39cea6fd612bf0121bdb5e0`.
El binding del módulo científico usa el hash correcto
`46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
los otros sources, config y autoridades R475/R476 también coinciden con sus
bindings físicos.

## Recomposición numérica independiente

Recalculé desde los NPZ sellados, sin re-forward ni retraining, las 14 acciones,
56 arrays métricos, la máscara primaria de 301 `pair_token`, los 5.000
bootstraps PCG64 seed 6007, las métricas de las 14 políticas, 16 contrastes
principales, dos contrastes contra la media de controles y los 12 soportes.
Los índices bootstrap se reproducen array-exactos y todos los escalares
coinciden exactamente con `analysis.json`.

| Política congelada vs hard | Métrica | Diferencia media | IC95 |
|---|---|---:|---:|
| incompatibility/mean | accuracy | +0.008998 | [+0.000138, +0.017996] |
| incompatibility/mean | compatible | +0.014396 | [+0.007198, +0.022425] |
| incompatibility/mean | regret | -0.018411 | [-0.028344, -0.009159] |
| incompatibility/mean | worst regret | -0.022702 | [-0.044581, -0.002769] |
| harm/tail | accuracy | +0.008721 | [+0.001246, +0.016334] |
| harm/tail | compatible | +0.010520 | [+0.003876, +0.017996] |
| harm/tail | regret | -0.014904 | [-0.023844, -0.006344] |
| harm/tail | worst regret | -0.017996 | [-0.031838, -0.005260] |

Los soportes principales son 46 `pair_token` autorizados para incompatibility
y 35 para harm. No obstante, las condiciones discriminantes contra controles
matched permanecen falsas:

- regret de incompatibility contra la media de cinco controles de
  desplazamiento máximo: `-0.000622923588039867`, IC95
  `[-0.0031284606866002216, 0.0015481381506090807]`;
- worst regret de harm contra su media de cinco controles:
  `-0.0016611295681063123`, IC95
  `[-0.014064230343300111, 0.009468438538205979]`.

Por ello los patrones son `incompatibility=false` y `harm=false` tanto con
`replay_exact=false` en la observación histórica como con
`normalized_replay_exact=true` en la vista corregida. R530 preserva
`scientific_decision=null`, `decision_authority=user`,
`architecture_promoted=false` y `gpu_used_or_queried=false`.

## Observación, hipótesis e inferencia

**Observación.** El attempt es terminal e íntegro; primary y replay son
científicamente exactos; el único check histórico falso proviene de comparar
dos hashes locales de receipts cuyo contenido sólo difiere en el rol de
ejecución. Las métricas y los dos patrones negativos no cambian al normalizar.

**Hipótesis causal acotada.** La representación histórica de falta de replay
exacto se explica por reintroducir en una comparación cross-root una identidad
operacional local después de haber comparado correctamente la semántica del
generation receipt. No hay evidencia en este corpus de una segunda causa
científica.

**Inferencia autorizada.** La vista condicional puede registrar replay
científico exacto y sustituir únicamente la limitación de fase por
`replay_exact_adjudicated_by_r509_findings_resolution_chain`. El resultado
sigue limitado al generador sintético, al draw nuevo y a la ley Wave 59
congelada, con intervalos condicionales sin corrección de multiplicidad. No
identifica por sí solo un efecto del target, no refuta toda señal contextual y
no autoriza declarar un techo.

## Comprobaciones CPU-only

Todas las órdenes fijaron `CUDA_VISIBLE_DEVICES=''`, cuatro threads y
`PYTHONDONTWRITEBYTECODE=1`; no se usó ni consultó GPU. Pytest deshabilitó
plugins externos y cache y usó un basetemp propio bajo `/mnt/m2-1TB`, retirado
después de verificarlo.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| `validate` sobre R530 | PASS | 1.57 s | 766.724 KiB | 0 |
| `check-attempt` real | PASS | 2.26 s | 769.664 KiB | 0 |
| `build` autenticado R528/R529 | PASS; SHA idéntico a R530 | 2.69 s | 769.556 KiB | 0 |
| pytest focal de attempt/autoridad/mismatch/receipts | 6 PASS, 28 deselected | 3.26 s | 754.876 KiB | 0 |
| recomposición independiente de inventario, receipts, hashes, métricas, bootstrap y soportes | PASS | — | — | — |

El runtime no ofrece una introspección independiente del identificador del
modelo ni del esfuerzo, por lo que no afirmo haber verificado desde dentro
`gpt-5.6-sol/high`; sí consta como requisito explícito del dispatch recibido.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R531",
  "scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
  "target": {
    "artifact_commit": "bc344ad9026b99c8c37ac008372d0789a15af1b4",
    "artifact_path": "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json",
    "artifact_sha256": "01600e6ae26c51f28a3485110fc8561fbd9d0b4fe4fb54849379f106d6aee802"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
