# Wave 59 replay-normalized successor implementation audit R449

**Implementation commit:** f43507a172b88f1dfd9b4406cdc038257da14b00  
**Module SHA-256:** eb4b77d6ee3469a82542471d95a5e71772170a84ecb80560cf439fe6b07e3ba9  
**Preparer SHA-256:** a11ed33357f3416dd6301b36a84b7a2d500163c1b130061e61e5bbd0e49a0c00  
**Runner SHA-256:** ca26e3b98d9bac5720f5dff7e9dc196faa8747d01843a567d80e69663ae8d1e9  
**Prospective test SHA-256:** 909f74a2547d214dac569bfb509fd409d7494f6e1fa2736a5cbd5901f94eede8  
**Recovery test SHA-256:** 7f13bb12a8c310c424d369b1df62e7a9adc2885d8629963a6c27207970571019  
**Result:** REVISE

## Dictamen

La implementación cierra correctamente la identidad exacta de las dos configs, la base de conteo `eligible_unique_pair_tokens`, el delta de fuentes, la unión exclusiva entre paquetes frescos y recovery, la reconstrucción física de las atestaciones y la secuencia enlace bruto → firma → normalización. Sin embargo, no puede aceptarse todavía: queda un bypass material en la cadena de autoridad.

### BLOCKER — Los dictámenes sucesores no se validan de forma canónica ni contradiction-safe

El plan exige que la auditoría de implementación contenga el commit, los hashes de los cinco archivos, evidencia de tests y un dictamen `PASS`; además declara la auditoría final como autoridad material y fail-closed (`WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:224-230,261-276`).

La implementación no cumple ese cierre:

- `prepare_wave56_fresh.py:1236-1238` sólo busca el hash del implementation commit en cualquier parte del texto y cuenta una aparición de la cadena `## Dictamen: PASS`.
- No valida ninguno de los cinco campos SHA-256 exigidos.
- No exige `**Result:** PASS`.
- No valida el bloque terminal `**Final decision:** PASS`.
- Un informe con resultado y decisión terminal `REVISE`, pero con el commit mencionado y una cadena `## Dictamen: PASS`, sería aceptado.
- Para la auditoría final, `prepare_wave56_fresh.py:1372-1382` exige tres líneas y una aparición de `## Dictamen: PASS`, pero tampoco contrasta la decisión terminal ni prohíbe un dictamen contradictorio.

El repositorio ya dispone del parser estricto `_require_report_fields()` (`prepare_wave56_fresh.py:1036-1110`), que valida layout, unicidad y coherencia con la decisión terminal. La autoridad recovery ya usa ese parser y posee una regresión contra auditorías contradictorias (`tests/test_wave59_preoracle_recovery.py:308-317`); el camino sucesor introdujo un parser más débil.

Corrección necesaria:

- Aplicar validación canónica estricta a la auditoría de implementación y a la auditoría final.
- Ligar los cinco SHA-256 de R449 a los blobs del implementation commit y a los blobs preservados en `HEAD`.
- Exigir coherencia única entre `Result`, `Dictamen` y `Final decision`.
- Agregar pruebas sintéticas que rechacen hashes ausentes o incorrectos, `Result: REVISE`, decisión terminal contradictoria, un `PASS` incluido sólo en prosa y la coexistencia de dictámenes `PASS`/`REVISE`.

Las dos pruebas actuales del delta y la autoridad final quedan deliberadamente omitidas mientras no existan la config y el informe futuros (`tests/test_wave59_prospective.py:186-212`); por tanto, los `2 skipped` informados tampoco cubren este bypass.

### MEDIUM — Un replay fresco fallido se archiva falsamente como recovery

El protocolo sucesor define `primary/replay` fresco sin amendment ni provenance. Aunque el transaction recibe correctamente `recovery_context=None` (`prepare_wave56_fresh.py:4329-4339`), el manejador exterior archiva cualquier `mode=="replay"` con `recovery_context=True` (`prepare_wave56_fresh.py:4351-4362`).

Si falla la actualización del receipt de presupuesto o la republicación de la atestación después de crear el output, el failure record declarará falsamente un contexto recovery y `_artifact_classes()` esperará `recovery_amendment.json` (`run_wave59_hgb_guard_bracket.py:2067-2073`). Esto degrada la trazabilidad del fallo y mezcla las dos ramas que el sucesor pretende mantener exclusivas.

Debe inicializarse el contexto antes del `try` y pasar al archivador la presencia real de autoridad recovery, no inferirla del modo replay. Falta una regresión que provoque un fallo posterior a la preparación en un replay fresco y compruebe `recovery_context=false` y ausencia de amendment.

### Evidencia verificada

- `f43507a172b88f1dfd9b4406cdc038257da14b00` es hijo directo de `008ba4b660caf11d1c14b70779ced0528d25ba5f`.
- El commit modifica exactamente los cinco paths autorizados.
- Los cinco hashes declarados arriba coinciden con los archivos auditados.
- `git diff --check` termina con exit `0`.
- El worktree permanece limpio.
- Evidencia coordinada disponible: `92 passed, 2 skipped` para Wave 59 y `310 passed, 2 skipped` para las nueve suites, en `387.24 s`.
- Esta instancia no reejecutó regresiones, no abrió secretos, escrow ni truth, y no usó GPU, web o Mendieta.
- No se ejecutaron preparación, recovery ni draw.

El dictamen no constituye `GO/NO-GO` científico. La corrección debe producir un nuevo implementation commit auditable antes de congelar la config sucesora.

## Machine-verifiable decision

**Final decision:** REVISE
