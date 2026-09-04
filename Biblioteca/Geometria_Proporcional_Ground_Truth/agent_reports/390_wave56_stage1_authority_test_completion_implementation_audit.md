# R390 — Auditoría independiente de la implementación de cierre de cobertura de autoridad de Wave 56 Stage 1

**Implementation commit:** `68316175067419c914af584e14ec2bafa4ff550b`
**Runner commit:** `7b37b5381b0c7540e86de2d53001903475d321ab`
**Authority commit:** `3f404103111a67721fa7a3d15cbf4ec392025e5f`
**Preparer SHA-256:** `84e6dabd6d3c08bd2569198d65771915a53ae5c8a5082460724bc1e9c8052368`
**Runner SHA-256:** `a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`
**Test SHA-256:** `2c327d84dff2d535775828d1bac4267e17b28d009b27c031937a9adcf476318b`
**Result:** `REVISE`

## Dictamen ejecutivo

I5 materializa correctamente el nuevo campo `authority_commit`, conserva el
runner byte-idéntico desde I3, respeta la cadena P7→R389→I5 y no altera paths
fuera del preparador y el test focal. Los 56 tests pasan en CPU. Sin embargo,
la matriz no cumple toda la cobertura end-to-end que P7 congeló y que R389
ordenó comprobar estado por estado: falta el path faltante de I5, el runner
modificado dentro de I4 y la gramática terminal completa instanciada por cada
uno de R389, R390 y R391. El resultado es `REVISE`.

## Alcance

Leí completos P7, R388 y R389 y contrasté sus obligaciones con el preparador,
el runner y el test vigentes. Inspeccioné commits, parents, diffs, commits de
introducción, blobs y hashes. No abrí escrow, secretos, artefactos sellados,
truth, labels, oracle ni resultados oficiales; no usé web o GPU y no ejecuté
recovery, replay ni fases oficiales.

## Identidad, lineage y blobs

I5 tiene como único parent a R389 y cambia exactamente
`experiments/geometria_proporcional/prepare_wave56_fresh.py` y
`tests/test_wave56_preoracle_recovery.py`. R389 fue introducido por su commit
exclusivo, hijo directo de P7; P7 también fue introducido por un commit
exclusivo. I3 es ancestro de I4, I4 es ancestro de P7 y la cadena directa
P7→R389→I5 coincide con el plan.

Los blobs de I5 y del worktree coinciden con los hashes de cabecera. El runner
en I3, I4 e I5 es byte-idéntico y conserva SHA-256
`a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`.
El plan tiene SHA-256
`21125d2f848e0e41e46b71978215f435b5b9a787a831f2326222cf49c8cb13c2`;
R389 presenta cabecera y decisión terminal `PASS` concordantes. `git diff
--check` no reportó errores.

## Implementación de autoridad

El preparador exige las identidades fijas de I3 e I4, verifica I3→I4 e
I4→P7, exige que I5 descienda directamente de R389, limita los diffs de I3,
I4 e I5 a sus conjuntos previstos y liga los blobs finales de preparador,
runner y test. La captura explícita de `FileNotFoundError` convierte una
auditoría ausente en rechazo controlado. El parser terminal conserva UTF-8/LF,
cabecera única, resultado válido, decisión única y concordante, ausencia de
contenido posterior y rechazo global de fences, comentarios HTML y separadores
no canónicos.

## Finding material

### F1 — Medio — La focal ampliada todavía no materializa toda la matriz de P7

P7:47-57 exige casos end-to-end independientes. La implementación añade diez
negativos y cierra los faltantes enumerados por R388 para runner falso/no
ancestro, paths faltantes en I4 y autoridad de R389. No obstante:

- `build_provenance_repo` siempre modifica preparador y test al crear I5
  (`tests/test_wave56_preoracle_recovery.py:353-359`). Sólo existe I5 con path
  adicional; no existe I5 con preparador faltante ni con test faltante, pese a
  P7:52 y a la condición explícita de R389:79-83.
- `change_runner_after_runner_commit` modifica el runner al crear I5
  (`tests/test_wave56_preoracle_recovery.py:353-359`). No construye un I4 cuyo
  runner difiera de I3, por lo que no ejercita el guard específico de
  `prepare_wave56_fresh.py:1059-1062` exigido por P7:50-51.
- El fixture Git sólo construye para R389 la contradicción
  cabecera-resultado/decisión, y para R390/R391 sólo un subconjunto de las
  formas terminales. Los tests unitarios genéricos cubren ausencia,
  duplicación, contenido posterior y ambos fences sobre `_require_report_fields`,
  pero no son los casos end-to-end separados por R389/R390/R391 que P7:56-57 y
  R389:79-83 exigen literalmente.

Los guards inspeccionados parecen rechazar esos estados, pero la finalidad de
P7 es demostrar cada rechazo en la focal, no inferirlo desde el código. La
corrección puede limitarse al fixture y a nuevos parámetros del test; no exige
cambiar runner, no-redraw ni protocolo científico.

## Deltas de fuente y no-redraw

La lista prospectiva contiene 26 fuentes únicas. Entre el origen
`51aae0715dfe8318f5333c568429c8e9af59f866` e I5 cambian exactamente dos:
preparador y runner. Sus hashes finales coinciden con la cabecera. El test focal
rechaza la pérdida del delta del runner y la aparición de un tercer delta; el
preparador exige igualdad del inventario y el conjunto exacto de dos deltas.

No-redraw no fue alterado por I5. Recovery y replay toman las claves del escrow
reutilizado, rechazan claves externas simultáneas, revalidan físicamente el
origen antes de extraerlas y después de regenerar, y comparan manifest y replay
exactos. La prueba física mantiene `secrets.token_bytes` convertido en fallo si
se intenta un redraw.

## Pruebas

La focal CPU-only terminó `56 passed in 17.57s`, sin fallos ni skips, con GPU
oculta y threads numéricos limitados a uno. No se realizó ninguna ejecución
oficial.

Este dictamen es técnico y pre-oráculo. No autoriza recovery, fases, labels u
oracle y no constituye `GO/NO-GO` científico.

## Machine-verifiable decision

**Final decision:** `REVISE`
