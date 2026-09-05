# Ola 59 — reauditoría focal posterior a R419

## Dictamen: PASS

La versión vigente resuelve íntegramente la condición de PASS de R419 y no
introduce una contradicción local en el contrato de artefactos, replay,
recuperación o estado portable. La implementación puede comenzar después del
cambio mecánico de estado autorizado al final de este informe; este dictamen no
promueve una arquitectura ni decide `GO/NO-GO`.

## Identidad y alcance

- Plan auditado:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md`.
- SHA-256 observado y esperado:
  `5eb7d5b95632f5ad13f91b2121805d86684b733752d0e7fe87c0b3478c254c2c`.
- R419 contrastado: SHA-256
  `09e60370db167d4bffa7fc03d6d4365f5c0ebe03a94908544a0a3907e69747e9`.
- `HEAD` observado: `8355a21caba67d048269be47b57e45d4ae1b1da7`.

Se releyó el plan completo —626 líneas— y se revisaron únicamente las
reparaciones exigidas por R419 más regresiones locales producidas por ellas.
La auditoría fue CPU-only con `CUDA_VISIBLE_DEVICES=''`; no se consultó web,
GPU ni Mendieta.

## Verificación focal

### Recovery amendment y cardinalidades — PASS

`recovery_amendment.json` pertenece ahora explícitamente a
`scientific_exact` como path condicional (`plan:501-507`). La tabla separa los
dos ejes `run_role = primary|replay` y
`recovery_context = false|true`, y fija las cuatro combinaciones exitosas sin
ambigüedad (`plan:517-526`):

- primary normal: sin receipt de replay ni amendment;
- replay normal: receipt de replay, sin amendment;
- primary recuperado: amendment, sin receipt de replay;
- replay del recuperado: ambos.

El amendment es obligatorio sólo en recovery, debe coincidir con el SHA
pre-oráculo autorizado y ser byte-exacto entre primary y replay
(`plan:528-530`). Así se conserva simultáneamente la clasificación exhaustiva
y la comparación que ya realiza el preparador compartido.

### Árbol de fallo e inventario — PASS

Los intentos `.failed_<timestamp>` tienen schema independiente
`wave59-failed-attempt-v1`, `FAILURE.json` obligatorio y
`failure_inventory.json` (`plan:532-540`). `FAILURE.json` conserva estado,
nivel máximo de truth, rol y contexto sin incluir secretos; el inventario
clasifica los artefactos parciales con las seis clases base y reserva la clase
no solapada `failure_record` para los dos JSON de fallo.

La obligatoriedad queda ligada al último journal: sólo se aceptan artefactos ya
publicados y se rechazan paths posteriores o extras. El árbol fallido no se
compara falsamente contra el primary exitoso. Esto da contrato recuperable a
`FAILURE.json` sin incorporarlo a las cardinalidades del camino feliz.

### Cobertura sin paths huérfanos — PASS

La matriz conserva expansión cartesiana cerrada y prohíbe globs residuales
(`plan:501-515`). Los tests cubren primary/replay × normal/recovery, verifican
el mismo amendment en replay recuperado e inyectan fallos antes y después de
cada apertura de truth. Cada inventario debe finalizar con cero paths
faltantes, extras, solapados o sin clase (`plan:542-545`). Esto satisface la
condición operacional exacta pedida por R419.

### Conteo de freezes — PASS

Replay exige ahora los cinco freezes anteriores —`pre_generation`,
`preparation`, `fit`, `calibration` y `validation`— además de
`monitor_action_freeze.json` (`plan:574-585`). El conteo nominal coincide con
la tabla de `scientific_exact`; ya no hay contradicción cuatro/cinco.

### Tolerancia del scorer portable — PASS

El contraste previo a publicar FIT entre scorer portable y objeto ajustado fija
`rtol=0`, `atol=2e-15`, `equal_nan=True` y acota esa tolerancia al redondeo de
acumulación (`plan:559-562`). La comparación funcional joblib conserva su
contrato independiente exacto `rtol=0`, `atol=0` (`plan:563-568`). No se mezclan
las dos autoridades ni sus tolerancias.

## Regresión local

No se detectó regresión en los requisitos previamente aprobados. La
condicionalidad del amendment es coherente con `scientific_exact`; el schema de
fallo agrega una clase propia sin solaparla con las seis clases exitosas; y los
modos normal/recovery no alteran la separación física de
`MONITOR-APPLY`/freeze/`MONITOR-EVALUATE`, los soportes, los targets de control
ni la autoridad del estado portable.

## Autorización mecánica de cierre

Se autoriza reemplazar **únicamente**, en la línea de estado, la cadena:

```text
REVISED-AFTER-R419 / AWAITING-FOCAL-INDEPENDENT-REAUDIT
```

por:

```text
FINAL-AUDITED-R420
```

sin ninguna otra modificación. La línea resultante debe ser:

```text
> **Estado:** `FINAL-AUDITED-R420 / FRESH-DRAW / CPU-ONLY / NO-GO-NOGO`
```

El SHA-256 esperado del plan después de ese reemplazo byte-exacto es:

```text
7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb
```

Este reemplazo sólo registra la culminación de la auditoría del plan. La
secuencia de implementación, auditoría de implementación, fresh draw, replay y
auditoría de resultados sigue vigente, y `scientific_decision` permanece
`null`.
