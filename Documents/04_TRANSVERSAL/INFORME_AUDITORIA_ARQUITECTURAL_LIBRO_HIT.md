# Auditoría arquitectural completa del libro HIT — Informe consolidado

Fecha: 2026-04-03
Origen: 3 agentes en paralelo (consistencia interna MD, paridad MD↔LaTeX, narrativa/editorial)
Cobertura: 2730 líneas MD + 21 archivos .tex, lectura completa

---

## División de responsabilidades

- **CODEX**: Prosa, narrativa, temporal consistency, editorial, hypothesis framing, Chapter 15 rewrite
- **CLAUDE**: LaTeX sync, tablas, figuras, referencias/bibliografía, glosario, valores numéricos, placeholders técnicos

Cada issue está marcado con `[CODEX]` o `[CLAUDE]` o `[AMBOS]`.

---

## CRITICAL / HIGH (6 issues)

### 1. `[CLAUDE]` Párrafo faltante en LaTeX Ch5 §5.4

Ch5 §5.4 del MD (línea ~443-444) tiene un párrafo completo con Kawai (2023), Zheng et al. (2025), Bahuguna et al. (2025) y Medvedev & Lehmann (2025) que agudiza H5 con evidencia reciente de cross-frequency coupling fisiológico. El LaTeX (`ch05_hypotheses.tex`) **no lo tiene**. Salta de la cita de Levin directo a "The critical point is one of scope."

Además hay prose drift menor en el mismo párrafo: LaTeX dice "consonant organization can couple to affective and dopaminergic processes" mientras MD dice "pleasurable musical experience can couple to affective and reward-related neural processes."

**Acción Claude**: Agregar párrafo al LaTeX. Verificar que las 4 citas existan en `references.bib`.

### 2. `[CODEX]` MERT: ¿330M o ~60M?

El manuscrito dice consistentemente "roughly 330 million parameters" (líneas 968, 989, 2233) para el audio encoder. Pero la implementación real de Phideus usa **MERTEncoderLite** (4 CNN + 4 Transformer, d=1024, **~60M params**), no el MERT-330M completo.

Posibilidades:
- El modelo usa MERT-330M como feature extractor congelado y solo entrena un lite wrapper → hay que aclararlo
- El modelo realmente usa 60M → el manuscrito está mal

**Acción Codex**: Verificar con el equipo cuál es la situación real y corregir la prosa. Si es 330M congelado + lite trainable, escribir algo como "MERT, a pretrained model with 330 million parameters, from which the audio encoder extracts features through a lite adapter of roughly 60 million trainable parameters." Si directamente es 60M, corregir los tres lugares.

### 3. `[CODEX]` CKA +82% atribución ambigua

Línea 1074: "Cross-encoder alignment, measured by CKA, rises by roughly 82 percent in the guided condition, from 0.435 in D0 to as high as 0.794 in the best aligned arm."

El +82% corresponde a **d4-a4r** (0.794), no a d4a4 (0.659 = +51%). Pero la Table 11.3a (línea 1087) lista "+82%" en una tabla donde el context sugiere d4a4. Un lector podría atribuir +82% a d4a4.

**Acción Codex**: Clarificar en la prosa que el +82% es el mejor arm (d4-a4r), y que d4a4 alcanza +51%. La tabla debería decir "+82% (best arm)" o similar para desambiguar.

### 4. `[CODEX]` Escalón 3 marcado "In progress"

Table 11.1 (línea 986) lista Escalón 3 como "In progress". La cronología (línea ~2285) dice "horizon". Ambos son incorrectos — E3 está completo hasta P6 con cross-comparison cerrada.

**Acción Codex**: Actualizar status en Table 11.1 y en la cronología. El status correcto es algo como "Complete through P6; cross-comparison closed."

### 5. `[CODEX]` Escalón 2 descrito con preguntas abiertas

Línea 1107: "The open question is now more precise..."
Línea 1347: "The clearest immediate question in that sense is now Speech-EGG."

Escalón 2 es CLOSED NULL (15/15 conditions ≈ D0, confirmado 2026-03-20). Ya no tiene preguntas abiertas.

**Acción Codex**: Reescribir el cierre de §11.6 para reflejar el null cerrado. Reescribir la referencia en Ch15.

### 6. `[CODEX]` Chapter 15.2 completamente stale

La sección "Immediate questions and programmatic next steps" (líneas 1339-1349) lista como "next steps" cosas ya completadas:
- Multi-seed validation → en curso en Mendieta
- Retrospective mechanism clarification → Gate 10 ya cerrado
- Speech-EGG foundation-encoder test → E2 CLOSED NULL
- "equally immediate task" para retrospective → ya resuelta

**Acción Codex**: Reescritura sustancial de §15.2 para reflejar el estado actual. Las "next steps" reales ahora son: d4a4 training multi-seed (en curso), Escalón 4 (ECG↔PPG, no iniciado), y las preguntas abiertas de Escalón 3 storage/retrieval.

---

## MEDIUM (12 issues)

### 7. `[CLAUDE]` Párrafo Soriano et al. (2025) falta en LaTeX Ch6

MD línea 492 tiene párrafo sobre Soriano et al. (2025) ausente en `ch06_convergence.tex`.

**Acción Claude**: Agregar al LaTeX. Verificar cita en bib.

### 8. `[CLAUDE]` 3 entradas de glosario faltan en LaTeX Appendix A

MD tiene Entropy (1929-1931), Information (1959-1960), The Real (2019-2021). LaTeX no las tiene.

**Acción Claude**: Agregar las 3 entradas al LaTeX `appendices.tex`.

### 9. `[CLAUDE]` Glosario LaTeX dice "Chapters 1-15", debería ser "1-16"

LaTeX `appendices.tex` línea 9 vs MD línea 1893.

**Acción Claude**: Corregir "15" → "16".

### 10. `[CLAUDE]` 4 entradas del glosario con cross-refs off-by-one en LaTeX

| Entrada | MD (correcto) | LaTeX (incorrecto) |
|---------|---------------|---------------------|
| HAT | Chapters 12-16 | Chapters 11-15 |
| Latent geometry | Chapters 11, 13 | Chapters 10, 12 |
| Lissajous | Chapters 11, 12, 13, 15 | Chapters 10, 12, 14 |
| Oscillatory portrait | Chapters 6, 15 | Chapters 6, 14 |

Probable causa: renumeración de capítulos no propagada al glosario LaTeX.

**Acción Claude**: Corregir los 4 cross-refs en LaTeX.

### 11. `[CLAUDE]` Reznikoff & Dauvois y Morley citados sin año (MD línea 546)

Dice: "(Conard et al., 2009; Reznikoff & Dauvois; Morley)"
Debería decir: "(Conard et al., 2009; Reznikoff & Dauvois, 1988; Morley, 2013)"

**Acción Claude**: Corregir en MD y verificar que LaTeX tenga los años.

### 12. `[CLAUDE]` Dos refs con `pp. XX--XX` (MD líneas 1866-1867)

Fernández Méndez (2021a) y (2021b) tienen page numbers placeholder.

**Acción Claude**: Buscar los page numbers reales o eliminar el campo.

### 13. `[CLAUDE]` Trulla et al. (2018) — dos papers sin disambiguar

Líneas 1493 y 1495: dos papers diferentes con mismo autor-año. Necesitan a/b en texto y en refs.

**Acción Claude**: Disambiguar en MD y en LaTeX bib.

### 14. `[CLAUDE]` Strogatz (2000) — libro y paper sin disambiguar

Líneas 1492 y 1494: misma situación.

**Acción Claude**: Disambiguar.

### 15. `[AMBOS]` Lakatos (1970) citado pero ref dice (1978)

Texto líneas 243, 1317 citan "(Lakatos, 1970)" pero refs lista solo (1978). Son obras diferentes.

**Acción Claude**: Agregar entrada Lakatos (1970) a la bibliografía.
**Acción Codex**: Verificar si la cita en el texto debería ser 1970 o 1978.

### 16. `[AMBOS]` Partch (1949) citado pero ref dice (1974)

Texto línea 371 cita "(Partch, 1949)" pero refs lista solo la 2nd ed. (1974).

**Acción Claude**: Agregar 1949 1st ed. a la bibliografía o cambiar la referencia a 1949/1974.
**Acción Codex**: Decidir cuál edición citar.

### 17. `[CODEX]` "distancia epistemologica" sin traducir (línea 1389)

Español en texto inglés. Debería ser "epistemic distance" o "epistemological distance."

**Acción Codex**: Traducir.

### 18. `[CLAUDE]` "ver entrada completa en Seccion" en refs (líneas 1830-1841)

Referencias usan cross-refs en español.

**Acción Claude**: Traducir a "see full entry in Section X.X" en MD y verificar LaTeX.

---

## LOW (11 issues)

### 19. `[CLAUDE]` Tables 12.1 y 13.1 en formato placeholder

Líneas 1201 y 1295 usan `[TABLE X.Y:]` mientras el resto usa Markdown tables renderizadas. No es error — son tablas descriptivas por naturaleza.

**Acción**: No urgente. Podrían convertirse a MD tables si se quiere uniformidad.

### 20. `[CODEX]` eval-seed en 4 formas distintas

`Eval-seed`, `evaluation seeds`, `eval-seed`, `evaluation-seed` en líneas 1084, 1092, 2044, 2048. Convendría unificar.

**Acción Codex**: Elegir una forma canónica y aplicar.

### 21. `[CLAUDE]` "Asociacion" sin acento (línea 373)

Frontmatter usa "Asociación" pero línea 373 dice "Asociacion".

**Acción Claude**: Corregir acento.

### 22. `[CLAUDE]` "En" (español) en ~20 refs de proceedings

"En *Proceedings of NeurIPS*" debería ser "In *Proceedings of NeurIPS*". También hay "Cap." y "(Trabajo original publicado en 1916)".

**Acción Claude**: Traducir todas las instancias en MD y en LaTeX bib.

### 23. `[CLAUDE]` Varela (1975) en refs pero nunca citado (línea 1552)

Referencia huérfana.

**Acción Claude**: Eliminar o citar en texto.

### 24. `[CLAUDE]` Appendix D "present cut" más corto en LaTeX

MD tiene párrafo expandido mencionando Ch8/9/10 storage-retrieval thread que LaTeX omite.

**Acción Claude**: Sincronizar.

### 25. `[CLAUDE]` Appendix F: LaTeX tiene sección extra que MD no tiene

LaTeX tiene "Activation, query, and retrieval" en AppF que MD no tiene. Drift inverso.

**Acción Claude**: Agregar al MD o eliminar del LaTeX. Decisión del usuario.

### 26. `[CLAUDE]` Table 12.1: MD describe 8 columnas, LaTeX tiene 5

MD describe Excitation method, How tuning is performed, Tradeoff — LaTeX las omite.

**Acción Claude**: El LaTeX es más conciso. No es necesariamente error. Decisión del usuario.

### 27. Placeholders [URL OFICIAL] y [MAIL DE CONTACTO]

Líneas 9, 10, 19, 34. Conocido, pendiente del usuario.

**Sin acción hasta que el usuario provea los datos.**

### 28. `[CODEX]` Refs en bibliografía nunca citadas en texto

Múltiples entradas (Foo 2016, Hunt & Schooler 2019, Lots & Stone 2008, etc.) nunca aparecen en el texto. Si la sección de referencias funciona como "References and Further Reading" debería indicarse explícitamente.

**Acción Codex**: Decidir si agregar encabezado "References and Further Reading" o citar las entradas en el texto.

### 29. `[CODEX]` "disciplined" usado 62 veces

Directiva editorial (2026-03-31) dice evitar "discipline/disciplined/disciplinary" como sinónimo de rigor. Muchos usos legítimos (academic disciplines), pero los adjetivales ("disciplined realities", "disciplined way", "disciplined exposure") podrían reemplazarse por "rigorous", "controlled", "methodical".

**Acción Codex**: Revisar los ~62 usos y reemplazar los que sean sinónimos de rigor.

---

## Resumen ejecutivo

| Severidad | Total | Codex | Claude | Ambos |
|-----------|-------|-------|--------|-------|
| Critical/High | 6 | 5 | 1 | 0 |
| Medium | 12 | 2 | 8 | 2 |
| Low | 11 | 3 | 7 | 0 |
| **Total** | **29** | **10** | **16** | **2** |

**Para Codex (10 items de prosa)**: #2 MERT params, #3 CKA atribución, #4 E3 status, #5 E2 open questions, #6 Ch15.2 rewrite, #17 español sin traducir, #20 eval-seed unificar, #28 refs no citadas, #29 "disciplined", más decisiones compartidas en #15 y #16.

**Para Claude (16 items técnicos)**: #1 LaTeX Ch5 párrafo, #7 LaTeX Ch6 párrafo, #8-10 glosario LaTeX, #11-14 bibliografía, #18 español en refs, #21-26 correcciones menores.

Lo más fuerte del libro: STRUCTURAL INDEX 100% correcto, numeración de figuras y tablas impecable, todos los valores numéricos coinciden entre MD y LaTeX, todas las cross-references válidas, todos los apéndices A-F presentes y en orden.
