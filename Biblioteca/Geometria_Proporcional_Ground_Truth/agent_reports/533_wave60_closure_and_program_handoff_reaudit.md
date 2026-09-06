# R533 — Reauditoría focal de la resolución del relevo experimental

## Dictamen técnico: PASS

El commit de resolución corrige los dos findings de R532 sin modificar la
evidencia de Ola 60, los estatutos de la cartera ni la condición de cierre del
goal acumulativo. El relevo ya no presupone que tres objetos heterogéneos forman
un factorial: condiciona esa comparación a un gate previo verificable y conserva
una bifurcación honesta si el mapeo común falla. El lifecycle prospectivo también
define ahora roles, orden de apertura, autoridad de escritura y evidencia de
freeze sin depender de un split implícito.

No encontré findings altos, medios ni bajos.

## Resolución de R532-01

`MAPPING-FEASIBILITY` queda situado antes de cualquier comparación neuronal
conjunta. El gate exige una única query, objeto científico, unidad inferencial,
schema público y target lógico. Para cada brazo requiere funciones tipadas
`observación → estado` y `estado → scores de compatibilidad`; por tanto, la IR
común deja de ser una equivalencia nominal.

La resolución separa explícitamente score EIV, calibración conformal, reader y
abstención. EIV no recibe estatuto de encoder ni de celda causal si esa
descomposición altera su significado. Del mismo modo, el núcleo grafo sólo
puede entrar mediante un adapter que preserve orientación, gauge y composición
y que no amplíe el acceso respecto de los demás brazos.

El gate debe declarar executor y checker comunes, marca la utilidad como
sintética y externa, interpreta «mismos bytes» como igualdad del input
autorizado y exige mutaciones contra cambios de target, autoridad y acceso. Si
la transformación no pasa, el diseño ya no fuerza una interacción inválida:
separa el contraste relacional `GENERIC/TYPED × WLS/IRLS` del contraste
set-valued `marginal/joint × hard/contextual`, mantiene EIV como referencia
externa cuando corresponda y difiere el router. Esta condición de salida también
vuelve finito el preflight aunque el mapeo sea negativo.

## Resolución de R532-02

El orden queda fijado como
`train → calibration → selection → freeze auditado → monitor`, con una sola
apertura de cada rol. Train ajusta parámetros; calibration elige únicamente
hiperparámetros, calibradores, thresholds, márgenes y potencia bajo reglas
predeclaradas; selection aplica un orden total congelado sin redefinir target,
métrica, margen o control; monitor sólo adjudica.

La tabla de autoridad asigna query, objeto, schema, target, controles y familia
confirmatoria al design freeze; parámetros al manifest de checkpoints;
calibración y márgenes al calibration freeze; identidad candidata al selection
receipt; y resultados al analysis/replay manifestados. El receipt posterior
liga la elección completa y debe superar auditoría independiente antes de abrir
monitor. La familia confirmatoria queda fijada antes de los draws y los márgenes
antes de selection. Desaparece así la referencia a un `validation` no declarado
y queda auditable quién puede escribir cada campo.

## Identidad y ausencia de regresiones

- El target es hijo directo de R532 y modifica exclusivamente el documento de
  síntesis indicado. Su SHA-256 físico coincide con el binding recibido.
- Las cifras, inferencias y límites de Ola 60 no fueron modificados.
- El texto sigue sin declarar techo, promoción, autoridad física ni
  `GO/NO-GO`.
- El master permanece como unidad independiente y las vistas relacionadas no
  se presentan como réplicas autónomas.
- Controles matched, target-shuffled, shuffles de adapter y replay continúan
  dentro del diseño; ningún módulo recibe crédito por una operación analítica
  externa.
- El preflight sigue siendo CPU-first. Si la etapa neuronal requiere muchas
  horas y GPU resulta materialmente más eficiente, el protocolo se detiene
  antes de CUDA y exige informar objetivo, duración y VRAM.

Esta reauditoría leyó el archivo vigente completo y contrastó la resolución con
R532 y con las fuentes primarias ya auditadas. No usó ni consultó GPU. El
runtime no ofrece introspección independiente del identificador del modelo ni
del esfuerzo; `gpt-5.6-sol/high` constaba como requisito del dispatch, pero no
se afirma como verificado desde dentro.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R533",
  "scope": "WAVE60_CLOSURE_AND_PROGRAM_HANDOFF_RESOLUTION",
  "target": {
    "resolution_commit": "bcf1c09eeff597e661ae92862c5ac412a3aa4fe7",
    "file": "Biblioteca/Geometria_Proporcional_Ground_Truth/PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md",
    "sha256": "888477866168d1f2e5dbb756835ef0ed6521a636ab8cd752aaaa6181dc490db2",
    "resolves": [
      "R532-01",
      "R532-02"
    ]
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
