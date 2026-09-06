# R517 — Auditoría del plan de corrección de autoridad del hash fuente de Ola 60

## Dictamen técnico: REVISE

El plan R516 se apoya en una premisa factual falsa: R514 no transcribió
`...fc4f...`. El archivo físico de R514 y el blob introducido por su commit
contienen `...fc4e...`, exactamente el mismo SHA-256 que la config R508, el
source físico y el blob Git ligado. R515, por tanto, no omitió una divergencia
entre esas autoridades. Registro **1 HIGH, 0 MEDIUM y 0 LOW**.

La tabla corregida que R516 propone es materialmente correcta, pero no corrige
R514: repite el valor que R514 ya fijaba. En consecuencia, R516 no puede
superseder la autorización sustantiva de R514/R515 ni habilitar la cadena
renumerada R518–R521. R514 y R515 permanecen sustantivamente válidos en este
punto; R516 debe conservarse como evidencia histórica no autorizante.

## Identidad del target

El target auditado es el commit
`d564d1c078248cc0083ecda206db81b0c80752da`, hijo directo de R515
`8b5fb79cb992034021698e7f36da8779d50824d2`. Su diff añade exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN.md`.
El archivo físico coincide byte a byte con el blob Git y su SHA-256 es
`c2b8fec2e6426c1456403bb2ed714d91ff6fe2611530c656f0ea13b976aade6b`.
`git diff --check` pasa.

## R517-01 — HIGH — la divergencia atribuida a R514 no existe

R516 afirma que R514 fijó el source principal como
`46e31fa1096ab4e5a0c5115fc4f16922e165a21dbe47016c42953bf345116c65`
y que esa transcripción difiere en un carácter de tres autoridades físicas.
La inspección literal da el resultado contrario:

| Superficie | Valor observado |
|---|---|
| config física `wave60_frozen_policy_transport.json` | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| config blob en `b156f6857eaa36edc8bda9e7687de7b8e1ea9721` | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| source físico | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| source blob en `b156f6857eaa36edc8bda9e7687de7b8e1ea9721` | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| R514 físico, línea 271 | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| R514 blob en `574f79810e5283dec0478c8c1c53e3496c808e97`, línea 271 | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |

La lectura hexadecimal de la línea del blob R514 confirma los bytes
consecutivos `66 63 34 65`, es decir, `fc4e`. El SHA-256 físico/blob de R514 es
`9f96084160627a4c9102949ff413d98b6c28a22a20bf2b6c4e115e44cc888b78`,
el mismo que R515 auditó. El valor `...fc4f...` aparece una sola vez en el
corpus relevante revisado: en R516, como supuesta cita del error de R514.

Además, la recomputación del mapa completo de `config.source_sha256` confirma
`8/8`: cada uno de los siete sources no autorreferentes coincide con su archivo
físico y su blob en `b156f685...`, y la octava entrada coincide con el
self-binding canónico de la config física y del blob. Esto incluye el valor
`...fc4e...` del source principal. Por esa razón, la declaración `8/8 PASS` de
R515 no omitió la divergencia alegada; no había divergencia que reportar.

### Impacto

La corrección de autoridad es la única razón declarada para reemplazar R515 y
renumerar la implementación, sus auditorías y el artefacto. Al fallar esa
premisa, la nueva autoridad no es necesaria ni suficiente para autorizar R518.
El borrador que contenía `...fc4f...` debe corregirse antes de integrarse, pero
un error de un borrador no modifica retrospectivamente R514 ni invalida R515.

La resolución mínima debe preservar R516 como histórico no autorizante y
retomar una implementación bajo la autoridad correcta ya publicada por
R514/R515, o introducir un nuevo plan que documente explícitamente esta falsa
atribución sin volver a superseder esas autoridades.

## Revisión de cadena, schemas y estado candidato

Separada de su premisa, la mecánica declarada por R516 es internamente
coherente:

- R516 tiene parent, pathset, blob y SHA correctos; el target R517
  `{plan_commit,plan_sha256}` es inequívoco.
- Los paths R517–R521 y los scopes/targets declarados para R517, R519 y R521
  están definidos sin colisiones.
- La transformación del `authority_chain` deriva once claves exactas: las
  siete autoridades históricas R509–R515, el plan R516, su auditoría R517, la
  implementación R518 y su auditoría R519.
- `artifact_status=CANDIDATE_PENDING_R521_AUDIT`, el binding a R521 y
  `authority_effect=ACTIVATES_CONDITIONAL_CORRECTED_VIEW` conservan una
  separación honesta entre candidato y activación.
- R516 incorpora por referencia los demás keysets, bindings, inventario,
  self-manifests y guardas semánticas de R514, y vuelve a exigir la matriz
  adversarial material.

Esas propiedades no resuelven R517-01: una cadena formalmente cerrada no puede
convertir una premisa histórica falsa en autoridad. Por ello no se elevan
findings adicionales sobre esos mecanismos, pero tampoco permiten PASS.

## Comprobaciones ejecutadas

1. Parent directo, diff exclusivo, pathset, blob físico y SHA del target R516:
   PASS.
2. SHA solicitado del plan R516: coincidencia exacta
   `c2b8fec2e6426c1456403bb2ed714d91ff6fe2611530c656f0ea13b976aade6b`.
3. Source principal contra config física, config blob, source físico y source
   blob: `4/4 PASS`, siempre `...fc4e...`.
4. Mapa completo de ocho sources contra estado físico y blobs ligados:
   `8/8 PASS`.
5. R514 físico/blob y lectura literal/hexadecimal de la línea normativa:
   `...fc4e...`; la alegación `...fc4f...` queda refutada.
6. R514 y R515 físico/blob y SHA publicados: `2/2 PASS`; R515 conserva su
   `8/8 PASS` sustantivo sobre source bindings.
7. Cadena R516–R521, paths, scopes, targets, keysets derivados y estado
   candidato: coherentes condicionalmente, pero no autorizantes por R517-01.
8. `git diff --check` del commit objetivo: PASS.

Todas las comprobaciones fueron read-only, con `CUDA_VISIBLE_DEVICES=''` y
`PYTHONDONTWRITEBYTECODE=1` cuando se importó el canonicalizador de self-binding.
No se usó ni consultó GPU, no se ejecutó training, forward, replay ni recovery,
y no se modificaron los archivos auditados.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R517",
  "scope": "R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN",
  "target": {
    "plan_commit": "d564d1c078248cc0083ecda206db81b0c80752da",
    "plan_sha256": "c2b8fec2e6426c1456403bb2ed714d91ff6fe2611530c656f0ea13b976aade6b"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 1,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
