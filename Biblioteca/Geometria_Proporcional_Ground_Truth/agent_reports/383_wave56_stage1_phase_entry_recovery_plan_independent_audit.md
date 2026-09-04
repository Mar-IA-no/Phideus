# R383 — Auditoría independiente del plan de recovery para la entrada a fase de Wave 56 Stage 1

**Plan commit:** `e0392ed94ca0e7073bb0f795717a16fa5c85181e`
**Plan SHA-256:** `2beeded884380087dc850f410da4516cc805e54ae12aefc0678c2489c4aa5e49`
**Result:** `PASS`

## Dictamen ejecutivo

El plan diagnostica correctamente el fallo observado y propone un arreglo
mínimo, ejecutable y cerrado por defecto. El runner actual construye el mapa
esperado únicamente con `train.jsonl`, `val.jsonl` y `lockbox.jsonl`, pero
compara ese mapa contra el inventario recursivo completo de
`benchmark/visible/`. El paquete oficial `PREPARED` contiene además
`calibration_null.jsonl`, declarado y autenticado por el manifest. La
reproducción pública da exactamente un miembro extra y ningún hash divergente
entre los tres splits compartidos.

La corrección propuesta conserva el binding de los tres splits al freeze,
incorpora al esperado todos los miembros `visible/` manifestados y mantiene la
comparación cerrada contra el árbol físico. No introduce un bypass ni relaja la
validación existente de hash y tamaño. El cambio del runner exige realmente un
segundo delta frente al contrato de origen; el schema v3 lo limita a preparador
y runner, liga además el test, y reinicia la autorización mediante una DAG y
artefactos nuevos sin reescribir la cadena v2.

No encontré findings altos, medios o bajos que requieran revisar el plan. El
dictamen es `PASS` técnico pre-oráculo: habilita avanzar a implementación y a
sus auditorías, no ejecutar fases ni declarar `GO/NO-GO`.

## Alcance y restricciones

Leí completos el plan vigente, el runner, el preparador, el test focal de
recovery, el plan de recovery anterior, R381, R382 y el amendment v2. Contrasté
los objetos Git, el config prospectivo y el paquete oficial sólo mediante
manifest, freezes, receipts, copia pública del amendment, nombres, modos,
tamaños y hashes. No abrí escrow, claves, miembros sellados, truth, labels,
oracle ni logits; no usé web ni GPU; no ejecuté recovery, replay o fases
oficiales y no modifiqué código, configs o datos.

## Evidencia del diagnóstico

`P3` resuelve a
`e0392ed94ca0e7073bb0f795717a16fa5c85181e`, tiene como parent único
`2b36c6c8d9ebd77e0fd681d67f8ed1add0b5fc0c` y agrega exclusivamente este
plan. El hash del blob y del worktree del plan es el declarado en la cabecera.

En el runner vigente, `validate_prepared_package` deriva el esperado como:

```text
train.jsonl   -> 18e5e137a3efa95d94e30d5cbfa7b2e3619b837246bbf72d9c9b1d04c2c95feb
val.jsonl     -> 72f42fe6a945987b036496453b6a4bd99e036502a057bf81e3b2753566fc6a06
lockbox.jsonl -> 51934804854bde82e1c17b31bba24ce58910e14e7e44c034936998b2f0ad20da
```

El inventario físico público observado es:

```text
calibration_null.jsonl -> 10e5ef2649dc9b0ace50e6009d178174e773a8af6dad4f5d89b1e799b0b8b6a7
lockbox.jsonl          -> 51934804854bde82e1c17b31bba24ce58910e14e7e44c034936998b2f0ad20da
train.jsonl            -> 18e5e137a3efa95d94e30d5cbfa7b2e3619b837246bbf72d9c9b1d04c2c95feb
val.jsonl              -> 72f42fe6a945987b036496453b6a4bd99e036502a057bf81e3b2753566fc6a06
```

Por tanto, la igualdad actual es falsa únicamente por
`calibration_null.jsonl`. El manifest oficial declara ese mismo cuarto miembro
con igual hash y `5.394.124` bytes. El receipt fija modo `recovery`, el estado
es `PREPARED`, y la metadata pública afirma `oracle_materialized=false`,
`authorized_labels_present=false`, `bundles_present=false` y
`fit_operations=false`; físicamente no existen `phases/` ni
`authorized_labels/`. Esto sostiene la inferencia acotada del plan sin abrir
verdad.

## Corrección y cierre del inventario

La composición propuesta es suficiente si se implementa literalmente:

1. los hashes de los tres splits continúan viniendo del
   `preparation_freeze.json`;
2. el manifest, ya ligado al freeze por su SHA-256, aporta el conjunto completo
   de miembros bajo `visible/`, incluida calibración nula;
3. las entradas compartidas deben coincidir entre ambas fuentes;
4. el mapa resultante debe igualar exactamente `hash_inventory(visible_root)`.

Ese diseño acepta el paquete íntegro y sigue rechazando miembro faltante,
mutación, symlink y archivo extra no manifestado. Además conserva el loop
anterior que verifica hash y tamaño de cada miembro público declarado por el
manifest. No hace falta agregar calibración nula a `visible_sha256`: queda
autenticada por el manifest congelado y por la comparación física cerrada.

## Autoridad v3 y DAG

Los hashes de origen fijados por el plan son correctos en el commit del
contrato del escrow:

- preparador:
  `7ff5919d2b0bdd607ca179180c4f94de3ff5be6e23e6024b21e748d22c61fb44`;
- runner:
  `304d27fa6ee2e6d511c5acef4f19c3990bd3af28cb207c5b43760f8d5efbda15`.

El config prospectivo incluye ambos entre sus 26
`required_execution_sources`. Por ello, tras `I3`, el mapa de ejecución debe
diferir del contrato de origen exactamente en esos dos paths; el test focal es
el tercer path del commit de implementación, pero no un tercer delta del
contrato. Esta distinción del plan es correcta.

El nuevo path v3, los nuevos reportes y la secuencia
`P3 → I3 → A3 → J3 → F3` evitan reutilizar R381, R382 o el amendment v2 como
autoridad para el runner nuevo. El plan exige blobs de los tres paths de `I3`,
attestation canónica con hash del runner, commits documentales exclusivos,
parents directos desde `I3` hasta `F3`, introducciones únicas, `HEAD == F3` y
worktree globalmente limpio. También exige ligar en v3 las identidades exactas
de `P3`, `I3` y `A3`. Es suficiente para conservar procedencia dual y evitar
cambios ejecutables posteriores a la auditoría.

## Preservación, no-redraw y orden fail-closed

La reejecución parte nuevamente del intento fallido exacto fijado en el
amendment, no del primario v2 como fuente de claves. El validador vigente ya
limita recovery amended a ese basename e inventario, revalida el origen antes
de extraer claves y después de regenerar el benchmark, y permite acceder a
`secrets.token_bytes` sólo en un primary fresco sin amendment. El plan conserva
todos esos controles.

Como el primario v2 existe, `--force` es necesario. La transacción vigente
primero valida invocación, fuentes, amendment, DAG, origen y escrow; sólo luego
archiva el output existente con sufijo `superseded`, crea el nuevo primario y,
ante fallo, archiva también el intento nuevo con sufijo `failed`. Así permanecen
recuperables tanto el origen fallido como el primario v2 `PREPARED`. El plan no
autoriza borrado ni redraw.

## Pruebas y verificaciones exigibles en `I3`, `A3` y `F3`

La matriz del plan cubre el hueco real de las suites anteriores: el recorrido
físico de recovery/replay actual llega a `PREPARED`, pero no hace pasar ese
paquete por `validate_prepared_package`. La implementación deberá demostrar:

- red/green del mismo paquete físico: fallo con el runner de `P3` y aceptación
  con el runner de `I3`;
- presencia manifestada de calibración nula y aceptación del árbol intacto;
- rechazo, antes de `begin_or_resume`, de calibración nula mutada o ausente y
  de cualquier visible extra no manifestado;
- delta contractual positivo exactamente `{preparador, runner}` y rechazo de
  cualquier conjunto distinto, incluido un tercer source delta;
- blobs Git exactos de preparador, runner y test y cabecera de `A3` ligada a
  sus tres hashes;
- conservación de los negativos existentes de no-redraw, origen físico, DAG,
  reportes, primary manifest y replay.

La focal vigente de recovery terminó `33 passed in 15.10s` en CPU-only, sin
fallos ni skips. Ese verde confirma el baseline v2, no sustituye los nuevos
positivos y adversariales exigidos por el plan. `F3` deberá repetir focal y la
suite amplia Wave 49–56 antes de autorizar la reejecución oficial.

## Findings priorizados

No hay findings materiales abiertos. Los puntos anteriores son criterios de
verificación ya exigidos por el propio plan, no ampliaciones del alcance ni
condiciones nuevas.

## Decisión

`PASS`. El plan puede implementarse tal como está. Corrige la causa observada
sin relajar integridad, autoriza de forma exacta los dos source deltas
necesarios, preserva el primario `PREPARED` y el origen, mantiene no-redraw y
fail-closed, y define pruebas y una DAG suficientes. Este dictamen no autoriza
todavía recovery, fases, oracle, labels ni ninguna decisión científica.
