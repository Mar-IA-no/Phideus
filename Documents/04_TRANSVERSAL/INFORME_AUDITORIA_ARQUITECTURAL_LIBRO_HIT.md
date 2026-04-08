# Auditoría arquitectural completa del libro HIT — Informe consolidado

Fecha: 2026-04-03
Origen: auditoría inicial de Claude + reauditoría completa de Codex contra el manuscrito MD y la documentación canónica actual de Phideus
Cobertura: manuscrito MD completo, contraste con LaTeX, cruce con documentación canónica de Phideus y revisión editorial/arquitectural

---

## División de responsabilidades

- **CODEX**: prosa, narrativa, consistencia temporal, framing de hipótesis, consistencia arquitectural del manuscrito, reescrituras de capítulos y apéndices en Markdown
- **CLAUDE**: sincronización LaTeX, tablas, figuras, glosario, referencias/bibliografía, placeholders técnicos y espejo MD→LaTeX
- **AMBOS**: casos en los que primero hay que fijar una decisión editorial o bibliográfica y luego espejarla en ambos formatos
- **USUARIO / PENDIENTE**: datos o decisiones que no se deben inventar

Cada issue está marcado con `[CODEX]`, `[CLAUDE]`, `[AMBOS]` o `[PENDIENTE]`.

---

## Criterio de esta versión

Este informe **corrige y amplía** la auditoría original de Claude.

Se conservan como válidos la mayoría de sus hallazgos, pero se agregan tres correcciones de fondo que la auditoría inicial no dejó bien explicitadas:

1. `§11.4` contiene una lectura de mecanismos que hoy contradice la lectura canónica de Phideus.
2. El stale-state de `Escalón 2` y `Escalón 3` no se limita a tablas sueltas; también contamina pasajes narrativos de `§11.8` y el cierre de `Appendix D`.
3. No todos los issues asignados a Codex tienen la misma urgencia: algunos son bloqueantes de consistencia arquitectural, otros son limpiezas estilísticas u opciones bibliográficas.

---

## CRITICAL / HIGH (7 issues)

### 1. `[CLAUDE]` Párrafo faltante en LaTeX Ch5 §5.4

Ch5 §5.4 del MD contiene un párrafo completo con Kawai (2023), Zheng et al. (2025), Bahuguna et al. (2025) y Medvedev & Lehmann (2025) que afila `H5` con evidencia reciente de cross-frequency coupling fisiológico. El LaTeX (`ch05_hypotheses.tex`) no lo tiene. Salta de la cita de Levin directo a "The critical point is one of scope."

Además hay drift menor de prosa en el mismo bloque: LaTeX dice "consonant organization can couple to affective and dopaminergic processes" mientras MD dice "pleasurable musical experience can couple to affective and reward-related neural processes."

**Acción Claude**: agregar el párrafo al LaTeX y verificar que las cuatro citas existan en `references.bib`.

### 2. `[CODEX]` MERT: ¿330M o ~60M?

El manuscrito dice consistentemente "roughly 330 million parameters" para el audio encoder en `§11.2`, la figura arquitectural y `Appendix C`. Pero la implementación canónica de Escalón 1 / Gate 5B parece apoyarse en `MERTEncoderLite`, mientras `MERT-330M` aparece explícitamente como régimen diferencial en Gate 7.1.

La ambigüedad no es cosmética. Hay dos lecturas distintas posibles:

- el libro quiso describir el backbone conceptual preentrenado y debería aclarar la relación `MERT backbone + lite adapter`;
- o el libro simplemente sobredimensionó el encoder real del régimen canónico.

**Acción Codex**: fijar primero con el equipo cuál es la formulación correcta. No corregir a ciegas. Una vez definida, propagarla en los tres lugares del manuscrito.

### 3. `[CODEX]` CKA `+82%` atribuido de manera ambigua

En `§11.5`, el texto dice que el alineamiento bajo CKA "rises by roughly 82 percent in the guided condition". Pero ese `+82%` corresponde al mejor arm alineado (`d4-a4r = 0.794`), no a `d4a4` (`0.659`, aproximadamente `+51%` respecto de `D0`).

La tabla de evidencia convergente vuelve a dejar la atribución ambigua.

**Acción Codex**: aclarar explícitamente que:

- `d4-a4r` produce el mayor salto de CKA;
- `d4a4` produce el mejor retrieval y la mejor retención;
- la tabla no debe dejar que el lector lea `+82%` como si fuera el efecto de `d4a4`.

### 4. `[CODEX]` `§11.4` dice algo que hoy contradice la lectura canónica de mecanismos

En `§11.4`, el manuscrito afirma que "attention-based injection systematically outperforms simple concatenation." Esa frase ya no es defendible como lectura global del programa.

Choca al menos con dos cosas:

- Escalón 1, donde el arm canónico más fuerte del capítulo sigue siendo `d4a4` por concatenación dual;
- Gate 10, donde la lectura cerrada del branch retrospectivo es `concat > FiLM/pca >> attention bias`.

La idea general de que "route matters" es correcta. Lo que está mal es la conclusión específica que da el párrafo.

La corrección, sin embargo, no debería pasarse al extremo opuesto. El problema no es que las variantes de atención hayan quedado "descartadas" o empíricamente irrelevantes. La lectura más fina hoy es otra:

- la concatenación puede cerrar con mejores scores en algunos brazos canónicos o retrospectivos;
- las variantes basadas en atención pueden quedar en banda similar o competitiva según el frente;
- y su valor no debe leerse solo por score bruto, sino también por su perfil mecanístico y computacional, porque permiten comprimir secuencias largas mediante una interfaz descriptor-guiada mucho más corta, reduciendo costo de procesamiento y carga de interacción.

En otras palabras, el punto fuerte de atención no tiene por qué formularse como "gana en performance" sino como "puede sostener rendimiento comparable bajo un régimen de interacción más eficiente y más interpretable para ciertos problemas de compresión descriptor→secuencia". Eso es compatible con la evidencia actual y además preserva el interés arquitectural de `reverse cross-attention` sin entrar en contradicción con `d4a4` ni con Gate 10.

**Acción Codex**: reescribir `§11.4` para que diga algo compatible con el estado actual del programa. La formulación correcta es más bien:

- el mecanismo importa tanto como el descriptor;
- distintos mecanismos habilitan distintos efectos;
- el programa ya mostró que no existe una superioridad universal de las variantes de atención sobre la concatenación;
- cuando se quiera justificar el interés de atención, la defensa debe pasar por eficiencia mecánica/computacional y compresión de interacción, no por una supuesta dominancia universal en score.

### 5. `[CODEX]` Escalón 3 sigue marcado como proceso abierto en demasiados lugares

Claude marcó bien dos lugares, pero el problema es más amplio.

Hoy `Escalón 3` está desactualizado al menos en:

- `Table 11.1`, donde aparece como `In progress`;
- `Appendix D`, donde aparece como `horizon`;
- `§11.8`, que todavía dice "Escalon 3 will bring ratio into a domain...";
- el párrafo de cierre de `Appendix D`, que sigue describiendo la arquitectura como si Phideus tuviera "one active new-domain front" y dos horizontes.

Eso ya no coincide con la lectura canónica actual de Phideus, donde Escalón 3 está cerrado al menos hasta `P6` con comparación cruzada ya establecida.

**Acción Codex**: actualizar todos esos pasajes para que la arquitectura del libro deje de leer Escalón 3 como promesa futura.

### 6. `[CODEX]` Escalón 2 todavía está escrito como pregunta abierta

El problema no se agota en `§11.6` o en `§15.2`.

Sí, hay frases explícitamente stale:

- "The open question is now more precise..."
- "`Speech↔EGG`" como tarea inmediata principal.

Pero además el cierre cronológico del libro sigue organizando Phideus como si Escalón 2 fuera todavía "the active new-domain front", cuando la lectura canónica actual es `closed null`, con `P3` primera pasada ya completada y la nueva pregunta situada en el contraste `P2 vs P3`.

**Acción Codex**: reescribir el cierre de `§11.6`, la parte correspondiente de `§15.2` y el párrafo final de `Appendix D` para reflejar el cierre null actual.

### 7. `[CODEX]` Chapter 15.2 quedó arquitecturalmente viejo

La sección "Immediate questions and programmatic next steps" hoy mezcla varias capas temporales.

Presenta como tareas inmediatas cosas que ya no lo son:

- el clarificado descriptor × mechanism retrospectivo ya quedó cerrado por Gate 10;
- `Speech↔EGG` ya no es la pregunta inmediata principal;
- el bloque futuro sigue organizado como si el programa estuviera en el corte previo a `S2-P3` y previo a la estabilización de Escalón 3.

**Acción Codex**: reescritura sustancial de `§15.2`. Las tareas inmediatas reales hoy son de otro tipo: cierre homogéneo de `d4a4` training multi-seed, preguntas abiertas de almacenamiento/recuperación en Escalón 3, protocolización comparativa de Beacon/PMP, y la eventual apertura de Escalón 4.

---

## MEDIUM (12 issues)

### 8. `[CLAUDE]` Párrafo Soriano et al. (2025) falta en LaTeX Ch6

MD tiene un párrafo sobre Soriano et al. (2025) ausente en `ch06_convergence.tex`.

**Acción Claude**: agregarlo al LaTeX y verificar la cita en bibliografía.

### 9. `[CLAUDE]` Tres entradas de glosario faltan en LaTeX Appendix A

MD tiene `Entropy (1929-1931)`, `Information (1959-1960)` y `The Real (2019-2021)`. LaTeX no.

**Acción Claude**: agregarlas al LaTeX `appendices.tex`.

### 10. `[CLAUDE]` Glosario LaTeX dice "Chapters 1-15", debería decir "1-16"

`appendices.tex` todavía quedó desfasado respecto de la renumeración real del libro.

**Acción Claude**: corregir `15` → `16`.

### 11. `[CLAUDE]` Cuatro entradas del glosario con cross-refs off-by-one en LaTeX

Las entradas `HAT`, `Latent geometry`, `Lissajous` y `Oscillatory portrait` quedaron con referencias de capítulos corridas en LaTeX respecto del MD.

**Acción Claude**: propagar la renumeración correcta en el glosario LaTeX.

### 12. `[CLAUDE]` Reznikoff & Dauvois y Morley citados sin año

Hay una cita en el cuerpo del libro que quedó sin año en dos referencias.

**Acción Claude**: corregir en MD y verificar espejo en LaTeX.

### 13. `[CLAUDE]` Dos referencias con `pp. XX--XX`

Fernández Méndez (2021a) y (2021b) quedaron con placeholders de páginas.

**Acción Claude**: completar páginas reales o eliminar el campo.

### 14. `[CLAUDE]` Trulla et al. (2018): dos papers sin disambiguar

Hay dos referencias diferentes con mismo autor-año y falta `a/b`.

**Acción Claude**: desambiguar en texto y bibliografía.

### 15. `[CLAUDE]` Strogatz (2000): libro y paper sin disambiguar

Misma situación: dos objetos bibliográficos distintos bajo el mismo año.

**Acción Claude**: desambiguar.

### 16. `[AMBOS]` Lakatos (1970) citado, pero la bibliografía solo fija (1978)

Acá hay dos problemas separados:

- decisión editorial/conceptual: qué texto quiere citar realmente el manuscrito;
- normalización bibliográfica: agregar la entrada correspondiente al formato final.

**Acción Codex**: decidir si el cuerpo del libro debe citar `1970` o `1978`.
**Acción Claude**: reflejar esa decisión en bibliografía/LaTeX.

### 17. `[AMBOS]` Partch (1949) citado, pero la bibliografía solo fija (1974)

Misma lógica que en Lakatos. Puede resolverse:

- con doble fecha `1949/1974`,
- o con entrada separada,
- o con sustitución editorial si se decide que la edición citada es la segunda.

**Acción Codex**: fijar la convención correcta.
**Acción Claude**: implementarla en bibliografía y espejo LaTeX.

### 18. `[CODEX]` "distancia epistemologica" quedó sin traducir

Hay una frase en inglés que conserva el sintagma en español.

**Acción Codex**: traducir a `epistemic distance` o `epistemological distance` y unificar con el resto del tono del capítulo.

### 19. `[CLAUDE]` En referencias aparece "ver entrada completa en Seccion"

Hay cross-refs bibliográficos todavía en español dentro de una bibliografía inglesa.

**Acción Claude**: traducir en MD y espejar en LaTeX.

---

## LOW (11 issues)

### 20. `[CLAUDE]` Tables 12.1 y 13.1 siguen en formato placeholder

No es error fuerte. Son tablas descriptivas y su formato actual es defendible.

**Acción Claude**: opcional. Solo si se busca uniformidad de render.

### 21. `[CODEX]` `eval-seed` aparece en varias formas

Hoy coexisten `Eval-seed`, `evaluation seeds`, `eval-seed` y `evaluation-seed`.

**Acción Codex**: elegir una forma canónica y unificarla.

### 22. `[CLAUDE]` "Asociacion" sin acento

Pequeña inconsistencia ortográfica.

**Acción Claude**: corregir.

### 23. `[CLAUDE]` "En" en español dentro de refs de proceedings

Hay varias entradas bibliográficas parcialmente en español dentro de un aparato bibliográfico en inglés.

**Acción Claude**: traducir esas fórmulas en MD y/o `.bib`.

### 24. `[CLAUDE]` Varela (1975) aparece en referencias pero no en el texto

Referencia huérfana.

**Acción Claude**: eliminar o introducir cita real si corresponde.

### 25. `[CLAUDE]` Appendix D "present cut" más corto en LaTeX

MD tiene una expansión conceptual que LaTeX no refleja.

**Acción Claude**: sincronizar.

### 26. `[CLAUDE]` Appendix F: LaTeX tiene una sección extra que MD no tiene

Hay drift inverso entre fuentes.

**Acción Claude**: normalizar después de decisión editorial.

### 27. `[CLAUDE]` Table 12.1: MD describe 8 columnas, LaTeX usa 5

No es necesariamente error. Puede leerse como simplificación de diseño.

**Acción Claude**: solo tocar si se decide explícitamente igualar densidad entre ambos formatos.

### 28. `[PENDIENTE]` Placeholders `[URL OFICIAL]` y `[MAIL DE CONTACTO]`

No se deben inventar.

**Sin acción** hasta que el usuario provea esos datos.

### 29. `[CODEX]` Bibliografía con entradas nunca citadas

Esto no es un bug arquitectural duro. Es una decisión editorial pendiente.

Opciones válidas:

- convertir la sección en `References and Further Reading`;
- o podar la bibliografía hasta dejar solo las entradas realmente citadas.

**Acción Codex**: decidir política bibliográfica. No es bloqueo inmediato para consistencia arquitectural.

### 30. `[CODEX]` "disciplined" y familia léxica sobreusados

Claude detectó bien el patrón, pero no todos los usos son problemáticos. Hay usos legítimos referidos a `disciplines`, `transdisciplinary`, etc. El problema real está en los adjetivales usados como sinónimo de rigor.

**Acción Codex**: limpieza estilística selectiva, no barrido ciego. Prioridad baja.

---

## Prioridad real para ejecución

### Bloque Codex verdaderamente urgente

Si hay que priorizar, las correcciones de Markdown que sí cambian la arquitectura legible del libro son:

1. `§11.4` mecanismo vs descriptor.
2. `§11.5` y `Table 11.3a` para desambiguar el `+82%` CKA.
3. Status narrativo de `Escalón 3` en `Table 11.1`, `§11.8`, `Appendix D` y cierre cronológico.
4. Status narrativo de `Escalón 2` en `§11.6`, `§15.2` y cierre cronológico.
5. Reescritura completa de `§15.2`.
6. Traducción de `distancia epistemologica`.
7. Unificación `eval-seed`.

### Bloque Codex que requiere decisión antes de corregir

Estos puntos no deberían tocarse mecánicamente:

1. `MERT 330M` vs `MERTLite`.
2. `Lakatos 1970` vs `1978`.
3. `Partch 1949` vs `1949/1974`.
4. Política de bibliografía no citada.

### Bloque Claude claramente técnico

Todo lo relativo a:

- párrafos faltantes en LaTeX;
- glosario LaTeX;
- referencias/bibliografía;
- cross-refs de capítulos;
- drift MD↔LaTeX en apéndices;
- detalles formales de render o placeholders técnicos.

---

## Resumen ejecutivo

| Severidad | Total | Codex | Claude | Ambos | Pendiente |
|-----------|-------|-------|--------|-------|-----------|
| Critical/High | 7 | 6 | 1 | 0 | 0 |
| Medium | 12 | 1 | 9 | 2 | 0 |
| Low | 11 | 3 | 7 | 0 | 1 |
| **Total** | **30** | **10** | **17** | **2** | **1** |

**Para Codex (10 issues propios)**: #2, #3, #4, #5, #6, #7, #18, #21, #29, #30; más decisiones compartidas en #16 y #17.

**Para Claude (17 issues técnicos)**: #1, #8-15, #19, #20, #22-27.

**Pendiente del usuario (1)**: #28.

La principal corrección introducida por esta reauditoría de Codex es el nuevo issue **#4**: la lectura de mecanismos en `§11.4` no solo estaba incompleta, sino que hoy entra en contradicción directa con la lectura canónica del programa. Ese punto no estaba bien capturado en la versión inicial del informe y debe tratarse como corrección arquitectural de alta prioridad.
