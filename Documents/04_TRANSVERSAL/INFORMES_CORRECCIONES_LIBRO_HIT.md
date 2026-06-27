# Informe consolidado: correcciones pendientes del libro HIT tras la auditoria cruzada Phideus ↔ HIT

Fecha: 2026-04-03
Origen: auditoria de trazabilidad numerica de Phideus + auditoria cruzada directa contra el Markdown actual del libro
Archivo auditado: `/mnt/m2-1TB/harmonic-information-theory/Harmonic_Information_Theory_Foundations.md`

---

## Estado del informe

Este documento **reemplaza** el memo previo sobre correcciones del libro.

La version anterior capturaba bien el nucleo de las correcciones numericas mas urgentes, pero **no agotaba** las cuestiones a resolver. En particular:

- identificaba correctamente el problema `d4a4` en `Ch11` y parte de `Appendix B`;
- identificaba correctamente las dos correcciones numericas de `Gate 10` y la de `Gate 8 pca`;
- pero dejaba fuera varias desalineaciones de **semantica**, **estado experimental** e **inventario tecnico** en `Appendix B`, `Appendix C` y la cronologia tecnica del libro.

Por lo tanto, este es el informe que debe tomarse como referencia canónica para la sincronizacion del libro.

---

## Veredicto ejecutivo

### Lo que si queda confirmado

Los claims numericos fuertes de Phideus en el libro siguen concentrados en tres zonas:

1. `Ch5 Table 5.1`
2. `Ch11 §11.5-11.6`
3. `Appendix B`

No aparecieron otros bloques ocultos de metricas duras fuera de esas zonas.

### Lo que el memo previo no alcanzaba a cubrir

Sin embargo, **eso no implica que esas sean las unicas correcciones del libro**. La auditoria cruzada muestra ademas:

1. `d4a4` contamina semanticamente no solo `§11.5` y `Table 11.3b`, sino tambien el prefacio de `Appendix B` y `Table B.1`.
2. `S2-P3` quedo desactualizado en `Appendix B`, `Appendix C` y en la cronologia del apendice tecnico.
3. `Gate 10` ya figura como cerrado en `Appendix B`, pero sigue apareciendo como `retrospective / provisional` y `active` en `Appendix C` y en la cronologia.
4. La fila tecnica de `P3` en `Appendix C` no solo esta vieja: tambien esta **mal nombrada**, porque lista arms `P3-*` sin el mecanismo `pca` que si define el cierre canonico actual.

### Conclusion operativa

El libro necesita:

- las correcciones numericas ya detectadas por el memo previo;
- **y ademas** una pasada de consistencia editorial-tecnica sobre `Appendix B`, `Appendix C` y la cronologia interna de Phideus dentro del manuscrito.

---

## Metodo de auditoria

La lectura se hizo contra:

- el Markdown actual del libro;
- el reporte de auditoria de trazabilidad de Phideus;
- `Proyecto_Estado_Actual.md`;
- `README.md` y `ROADMAP_ESCALON_2.md` de Escalon 2;
- `README.md` de `Gate 10`;
- y los JSON crudos relevantes:
  - `results_unc/gate10_mechanism_sweep/a7-pca_seed42/final_results.json`
  - `results_unc/gate10_mechanism_sweep/a10d-ab_seed42/final_results.json`
  - `results_unc/gate8_conditioned_projections/a4r-pca_seed42/final_results.json`
  - `data/gate5b_results/d4a4/multiseed/eval_seed*.json`
  - `data/lombard/p3_interpretation/p3_full_results.json`

La distincion metodologica aplicada fue:

- **error numerico confirmado**;
- **problema de semantica del claim**;
- **desfase de estado documental**;
- **pendiente de investigacion**.

---

## Hallazgo 1 — Critico: `d4a4` esta mal narrado como multi-seed homogeneo

### Donde aparece

- `Ch11 §11.5`: `L1070`
- `Table 11.3a`: `L1084`
- `Table 11.3b`: `L1092-L1096`
- `Appendix B` prefacio de metricas: `L2033`
- `Table B.1`: `L2039-L2044`

### Problema

La auditoria forense ya establecio que `d4a4` **no** tiene hoy el mismo estatuto que `D0`, `a4r` y `d4-a4r`.

- `D0`, `a4r` y `d4-a4r` si estan respaldados por `5` training-seeds independientes.
- `d4a4` esta hoy respaldado por `5` **eval-seeds** sobre un unico checkpoint `e30`.

Los valores reales de `eval-seed` si sostienen:

- mean `84.1%`
- sd `2.3pp`

Pero **no sostienen**:

- el rango documentado `82.0–86.4`
- ni los estadisticos inferenciales `Cohen d = 4.50`, `p < 0.001`
- ni la formulacion fuerte de "five independent seeds"

### Evidencia material

Los archivos `eval_seed42.json` y `eval_seed2026.json` muestran el mismo checkpoint:

- `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/checkpoint_epoch30.pt`

por lo que no describen training variance sino evaluator variance.

### Correcciones obligatorias

1. `L1070`: cambiar `Across five independent seeds` por una formulacion explicitamente `eval-seed`.
2. `L1070`: eliminar o rebajar `Cohen d = 4.50` y cualquier lectura de distribuciones homogeneas.
3. `L1084`: `84.1% ± 2.3pp` debe quedar calificado como `eval-seed` o `evaluation-seed reference`.
4. `L1092-L1096`: la tabla necesita nota metodologica y la fila `d4a4` debe corregirse:
   - rango `82.0–86.4` → `82.6–88.4`
   - `Cohen d` → `pending` o suprimido
   - `Sig.` → `pending` o suprimido
5. `L2033`: el prefacio de `Appendix B` no puede seguir hablando de `already stabilized multi-seed` como si la comparacion fuese homogenea.
6. `L2039-L2044`: `Table B.1` **si requiere cambio**, aunque mean y sd coincidan.

### Nota importante

La afirmacion del memo previo de que `Table B.1` "no requiere cambio" era demasiado optimista.
El numero `84.1% +/- 2.3pp` puede quedarse, pero la tabla debe dejar de sugerir que la fila `d4a4` tiene el mismo estatuto replicativo que las otras tres.

---

## Hallazgo 2 — Moderado: `Gate 10` tiene dos valores mal copiados en `Table B.6`

### Donde aparece

- `L2104`: `a7-pca`
- `L2106`: `a10d-ab`

### Correcciones confirmadas

| Arm | Libro actual | JSON crudo | Correccion |
|-----|--------------|------------|------------|
| `a7-pca` | `71.8% @ 30` | `71.6% @ 29` | corregir score y epoch |
| `a10d-ab` | `57.2% @ 30` | `57.4% @ 28` | corregir score y epoch |

### Fuente de verdad

- `results_unc/gate10_mechanism_sweep/a7-pca_seed42/final_results.json`
- `results_unc/gate10_mechanism_sweep/a10d-ab_seed42/final_results.json`

---

## Hallazgo 3 — Moderado: el resumen verbal de `Gate 10` tambien necesita ajuste

### Donde aparece

- `L2095`: caption de `Table B.6`
- `L2109`: sintesis verbal del ranking

### Problema

No solo hay dos celdas mal copiadas.

1. El caption dice `Final results, 9/9 arms at 30 epochs`, pero la tabla en realidad mezcla **best epoch dentro de runs de 30 epocas**, no "todo cerro en epoch 30".
2. La linea `concat > FiLM/pca (+2pp)` queda corta despues de corregir `a7-pca`.

### Correccion recomendada

1. Reescribir el caption para que diga algo como:
   - `Best structured results within 30-epoch runs`
   - o equivalente
2. En `L2109`, pasar `+2pp` a `+3pp` si la lectura se deja como gap mean-to-mean.

El `+16pp` frente a `attention bias` sigue siendo defendible como aproximacion.

---

## Hallazgo 4 — Moderado: `Gate 8 pca` tiene epoch incorrecto

### Donde aparece

- `L2074`

### Correccion confirmada

- `a4r-pca`: `82.6% @ 30` → `82.6% @ 25`

### Fuente de verdad

- `results_unc/gate8_conditioned_projections/a4r-pca_seed42/final_results.json`

---

## Hallazgo 5 — Bajo: `shuffled` en `Table B.2` sigue sin resolver la convencion

### Donde aparece

- `L2056`

### Problema

El libro da `73.6%`. El JSON crudo da `73.2%`. La documentacion interna de Phideus uso `73.6%*` como valor operativo de convergencia temprana.

### Opciones validas

1. `73.2%` sin nota
2. `73.6%*` con nota

### Recomendacion

Para publicacion, `73.2%` es mas limpio.

---

## Hallazgo 6 — RESUELTO: `D0 hard-negative accuracy = 80.4%` viene de Gate 2, no de Gate 5B

### Donde aparece

- `L2062`

### Problema original

No se encontraba artefacto crudo actual que sostuviera `80.4%` bajo el protocolo canonico de Gate 5B.

### Resolucion (2026-04-03)

El valor `80.4%` viene de **Gate 2 (Foundation baseline, Run D-02)**, no de Gate 5B. Aparece documentado en:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/01_GATES_0_2_5/GATE_2_FOUNDATION/INFORME_GATE2_COMPLETO.md:25`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md:95`

El D0 de **Gate 5B** (protocolo comparable a los otros arms de la tabla) tiene hard-neg = **94.6%** (`data/gate5b_results/D0/test12_scoreboard.json` → `hard_negatives.accuracy_vs_same_piece`).

### Consecuencia

Table B.3 mezcla dos regimenes en la columna hard-negative:
- `d4a4`, `a4r`, `d4-a4r`: son Gate 5B (94-95%)
- `D0`: es Gate 2 (80.4%)

Esto hace que la diferencia D0 vs guiados parezca de ~14pp cuando en regimen comparable es solo ~0.8pp. La comparacion es engañosa.

### Correccion obligatoria

Reemplazar `80.4%` por `94.6%` (Gate 5B D0), o bien:
- agregar nota explicando que el D0 hard-neg proviene de Gate 2 y no es directamente comparable;
- pero la opcion limpia es usar el valor Gate 5B para que toda la tabla sea del mismo regimen.

Si se corrige a `94.6%`, tambien hay que ajustar:
- `Ch11 L1078`: "Descriptor-guided arms sustain hard-negative accuracy above 94 percent" — esto sigue siendo verdad, pero la diferencia con D0 ya no es dramatica (D0 tambien esta en 94.6%).
- La narrativa de "fine discrimination" como ventaja diferencial de los descriptores queda debilitada en hard-neg (sigue fuerte en CKA y retention).

---

## Hallazgo 7 — Critico: `S2-P3` quedo desactualizado en `Appendix B`

### Donde aparece

- `L2115`: `Phase summary ... up to the opening of S2-P3`
- `L2126`: `S2-P3 ... already implemented and in execution, but does not yet have a stable comparative readout`

### Problema

Ese framing ya no coincide con Phideus.

El estado canónico actual del repo es:

- `P3-D0 = 78.8% @ ep15`
- `P3-A4-16k-pca = 78.2% @ ep25`
- `P3-V4-lin-pca = 76.8% @ ep28`
- `P3-H-series-pca = 75.6% @ ep25`

y la lectura vigente ya no es "opening", sino:

- primera pasada completada;
- sin lift descriptorial sobre `P3-D0`;
- siguiente tarea = comparacion `P2 vs P3`.

### Fuentes de verdad

- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md`
- `data/lombard/p3_interpretation/p3_full_results.json`

### Correccion obligatoria

`Tables B.7-B.8` **si requieren cambio**, al menos en el texto de encuadre, aunque sus numeros de `P2` sigan correctos.

El memo previo fallaba al decir que `Tables B.7, B.8` no requerian cambio.

---

## Hallazgo 8 — Critico: la fila de `P3` en `Appendix C` esta mal nombrada y mal clasificada

### Donde aparece

- `L2221`
- `L2235`

### Problema

La fila actual dice:

- `P3-D0, P3-V4-lin, P3-H-series, P3-A4-16k`
- status `planned / opening`

Pero el cierre tecnico actual del regimen `P3` no usa esos nombres desnudos, sino:

- `P3-D0`
- `P3-V4-lin-pca`
- `P3-H-series-pca`
- `P3-A4-16k-pca`

Ademas:

- `planned / opening` ya no corresponde;
- la insercion descriptorial de `P3` no puede describirse solo como "same descriptor families"; hay que reflejar que el cierre actual esta en la variante `pca`.

### Correcciones obligatorias

1. Corregir nombres de los arms `P3-*`.
2. Corregir el status de `P3`.
3. Reescribir la row arquitectonica de `L2235` para reflejar el regimen actual y no una apertura generica.

---

## Hallazgo 9 — Critico: `Gate 10` esta viejo en `Appendix C` y en la cronologia

### Donde aparece

- `L2211-L2213`: `retrospective / provisional`
- `L2280`: `Gate 10 opens ... | active`

### Problema

Eso contradice el estado actual de Phideus, donde `Gate 10` ya esta:

- cerrado `9/9`
- con ranking estable
- con lectura causal ya fijada

### Correccion obligatoria

1. Las rows de `Gate 10` en `Table C.3` deben dejar de figurar como `retrospective / provisional`.
2. La cronologia de `L2280` debe dejar de decir `opens` / `active`.

Lo correcto hoy es narrarlo como:

- barrido retrospectivo ya cerrado;
- contraste causal ya completado;
- lectura: mecanismo domina sobre descriptor en esa rama.

---

## Hallazgo 10 — Moderado: `S2-P3` tambien quedo viejo en la cronologia del programa

### Donde aparece

- `L2282`

### Problema

La cronologia dice:

- ``S2-P3` opens with `WavLM-Large` frozen on the speech side | active`

Eso ya no coincide con el estado actual del frente.

### Correccion recomendada

Reescribir la fila para que `P3` quede como:

- encoder-regime comparison already completed in its first pass;
- active only in the sense that the **next** step is the comparative readout `P2 vs P3`, no porque falte correr el regimen.

---

## Consolidado final: que si hay que corregir

### Obligatorias antes de publicar

| # | Zona | Lineas | Correccion |
|---|------|--------|------------|
| 1 | `Ch11` | `1070` | `d4a4`: dejar de decir `five independent seeds`; rebajar `Cohen d` y lectura de distribuciones |
| 2 | `Ch11` | `1084` | `Table 11.3a`: calificar `84.1% ± 2.3pp` como `eval-seed` |
| 3 | `Ch11` | `1092-1096` | `Table 11.3b`: nota metodologica + rango `82.6–88.4` + sacar o dejar `pending` `d/p` |
| 4 | `Appendix B` | `2033`, `2039-2044` | `Table B.1` y su encuadre: dejar de tratar `d4a4` como multi-seed homogeneo |
| 5 | `Appendix B` | `2074` | `a4r-pca`: epoch `30` → `25` |
| 6 | `Appendix B` | `2104` | `a7-pca`: `71.8 @ 30` → `71.6 @ 29` |
| 7 | `Appendix B` | `2106` | `a10d-ab`: `57.2 @ 30` → `57.4 @ 28` |
| 8 | `Appendix B` | `2109` | ajustar `+2pp` → `+3pp` si se deja como gap mean-to-mean |
| 9 | `Appendix B` | `2115`, `2126` | `S2-P3` ya no puede figurar como apertura sin lectura estable |
| 10 | `Appendix C` | `2211-2213` | `Gate 10`: sacar `retrospective / provisional` |
| 11 | `Appendix C` | `2221` | corregir nombres reales de los arms `P3-*` y su status |
| 12 | `Appendix C` | `2235` | reescribir la fila arquitectonica de `P3` |
| 13 | Cronologia | `2280` | `Gate 10` ya no `opens`; ya cerro |
| 14 | Cronologia | `2282` | `S2-P3` ya no `opens`; ya corrio su primera pasada |

### Recomendadas

| # | Zona | Linea | Correccion |
|---|------|-------|------------|
| 15 | `Appendix B` | `2056` | `shuffled`: pasar a `73.2%` o agregar nota |
| 16 | `Appendix B` | `2095` | aclarar que `Table B.6` reporta best results within 30-epoch runs, no "todo en epoch 30" |

### Pendiente de investigacion

| # | Zona | Linea | Tema |
|---|------|-------|------|
| 17 | `Appendix B` | `2062` | `D0 hard-negative = 80.4%` sin fuente cruda identificada |

---

## Lo que no requiere cambio por ahora

- `Ch5 Table 5.1` (`L457`): `Escalon 1: S = 84.1%`
- `Table B.5` (`Gate 9 / A10`): valores verificados
- los numeros de `Tables B.7-B.8` para `P2` siguen correctos
- no aparecieron mas metricas duras ocultas de Phideus fuera de las zonas auditadas

---

## Hallazgos adicionales (detectados por Claude LOCAL al auditar este informe)

Estos NO estaban en ningún informe previo — ni en el memo original ni en esta consolidación.

### Hallazgo 11bis — Cronología L2277: Gate 6 dice "active" pero es CLOSED NEGATIVE

Texto actual: `| Downstream extension | Phideus | Gate 6 opens AMT as validation outside retrieval | tested whether... | active |`

Correccion: status `active` → `closed negative`

Fuente: Gate 6 cerrado negativo, confirmado por `results_unc/gate6_amt/expA/` (todas las condiciones = baseline F1=0.3186).

### Hallazgo 12bis — Cronología L2281: Escalón 2 dice "active" pero es CLOSED NULL

Texto actual: `| New sensor-physics front | Phideus | Escalon 2 formalizes Speech↔EGG... | active |`

Correccion: status `active` → `closed null`

Fuente: Escalón 2 CLOSED NULL CONFIRMED (2026-03-20). 15/15 conditions ≈ D0.

### Consolidado actualizado

Estos se suman a los hallazgos 13 y 14 de la cronología (Gate 10 y S2-P3). El total de correcciones obligatorias pasa de 14 a **16**.

---

## Sintesis final

El memo previo identificaba bien los errores numericos mas visibles, pero no alcanzaba para certificar que esas fueran las unicas cuestiones a resolver.

La auditoria cruzada muestra que el libro todavia tiene tres capas de trabajo:

1. **correcciones numericas confirmadas** (`d4a4`, `Gate 10`, `Gate 8 pca`, `shuffled`);
2. **correcciones de semantica del claim** (`d4a4` en `Table B.1` y en el cuerpo del capitulo);
3. **correcciones de estado tecnico** (`S2-P3`, `Gate 6`, `Gate 10` y `Escalon 2` en `Appendix C` y cronologia).

Una vez hecho eso, el libro quedaria alineado con el estado canónico actual de Phideus sin sobredeclarar ni mezclar regimenes de evidencia distintos.
