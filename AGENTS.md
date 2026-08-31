# AGENTS.md

Instrucciones base para agentes en este repositorio (persistentes).

## Scope

Estas reglas aplican a cualquier agente que trabaje en `/mnt/m2-1TB/Phideus`.
Si hay conflicto, prevalece:

1. Instrucción explícita del usuario.
2. Este `AGENTS.md`.
3. Reglas específicas del agente (`CODEX.md`, `CLAUDE.md`).

## Bootstrap obligatorio de contexto

Al iniciar sesión/tarea, leer en este orden:

1. `AGENTS.md`
2. Reglas del agente activo:
   - Codex: `CODEX.md`
   - Claude: `CLAUDE.md`
3. Memoria privada del agente (si corresponde).
4. Estado operativo:
   - `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
   - `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`
   - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`

Objetivo: continuidad entre sesiones y resiliencia a compactaciones de contexto.

## Gobernanza de roles (por defecto)

- Claude: implementación y ejecución experimental.
- Codex: auditoría técnica, documentación, consistencia metodológica y trazabilidad.

Este reparto puede cambiar solo por instrucción explícita del usuario.

## Lectura/edición cruzada de reglas

- Codex puede leer `CLAUDE.md`, pero nunca editarlo.
- Claude puede leer `CODEX.md`, pero nunca editarlo.

## Memorias privadas por agente

- Codex: `.codex/memory.md` (privada).
- Claude: su `memory.md` privado (fuera de alcance de Codex).
- Backup operativo de Claude: `.claude/memory.md` (vetado para Codex: no lectura, no escritura).

Regla de aislamiento:

- Codex no inspecciona ni usa memoria privada de Claude.
- Claude no inspecciona ni usa `.codex/memory.md`.

## Colaboración entre agentes

- Modo por defecto: `OFF` salvo orden explícita del usuario.
- En `OFF`: no coordinación agente-agente vía archivos de coordinación.
- En `ON`: coordinación explícita con trazabilidad y handoff verificable.

## Norma metodológica para investigación de frontera

1. No declarar “techo” ni cerrar hipótesis con evidencia corta (pocas épocas, 1 seed, ventana parcial).
2. Separar explícitamente:
   - Observación (dato),
   - Hipótesis (explicación),
   - Inferencia (conclusión).
3. Priorizar comparabilidad estricta (misma receta/schedule/protocolo) antes de inferir causalidad.
4. Permitir hipótesis de alto potencial exploratorio sin desacoplarse de datos.

## Preservación de artefactos experimentales

Regla transversal vigente desde 2026-06-28: todo experimento nuevo debe preservar, por defecto, los artefactos reutilizables necesarios para reanálisis sin re-forward ni re-training innecesario.

Mínimo obligatorio salvo justificación explícita:

- checkpoints reproducibles (`last_epoch` siempre; checkpoints intermedios o best según el protocolo del frente);
- estados crudos de evaluación por unidad analítica (por ejemplo matrices/logits por mezcla, utterance o escena), no solo métricas agregadas ni vistas reducidas;
- configs, seeds, manifests, splits y versiones de dataset/modelo suficientes para reproducir el experimento;
- documentación de qué artefactos son canónicos, cuáles son diagnósticos y cuáles son regenerables.

Si una métrica, calibración, regla de clustering, umbral, bootstrap o análisis alternativo puede necesitarse razonablemente después, el estado que lo habilita debe guardarse durante el experimento. La purga selectiva de artefactos pesados se decide recién al cierre formal del frente, nunca antes de que la auditoría confirme qué hace falta conservar.

## Máximas epistemológicas vigentes (Phideus)

1. Escalón 1 debe leerse como validación fuerte de la mecánica descriptor-guided y de la reorganización geométrica del espacio latente; no clausura por sí solo la tesis fuerte de armonía natural.
2. Escalón 2 debe formular explícitamente la hipótesis de **armonía natural** en coordenadas físicamente naturales del fenómeno medido.
3. No confundir bajo una misma etiqueta:
   - dinámica temporal del oscilador,
   - estructura armónica natural intra-frame,
   - controles no-ratio,
   - variantes perceptuales/logarítmicas de comparación.
4. Los descriptores primarios deben derivarse preferentemente de invariantes físicos del fenómeno; los descriptores perceptuales o logarítmicos quedan como brazos comparativos explícitos, no como default.
5. La utilidad ingenieril de un descriptor no equivale automáticamente a validación de la tesis fuerte de Harmonic Information Theory.
6. En nuevos dominios, comenzar con baselines trazables y comparables antes de introducir asimetrías fuertes o foundation encoders.

## Política documental pública y privada

1. `README.md` y la documentación pública deben presentar el proyecto como estado canónico actual, no como diario de tanteos, tropiezos o negociaciones internas.
2. La documentación pública no debe exponer documentos locales de continuidad, cuadernos reflexivos privados, skills internas ni otros artefactos de uso exclusivo de agentes.
3. En documentación pública usar rutas relativas al repositorio o placeholders públicos (`$REPO`, `$HOME`, `$USER`), nunca paths privados o machine-specific.
4. `Documents/Skills/` es la capa pública de skills compartibles; solo deben aparecer allí skills estables, reutilizables y exportables fuera del flujo interno del repo.
5. Cuando cambien descriptores, roadmap o framing científico, la actualización documental debe propagar explícitamente la nueva lectura a los documentos transversales y al marco epistemológico cuando corresponda.

## Regla documental transversal (Phideus)

Mantener sincronizados cuando cambien descriptores/roadmap:

- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md`

No actualizar `BACKPROPAGANDO_PHIDEUS.md` salvo pedido explícito del usuario.

## Wiki viva de Phideus

`Documents/05_WIKI/` es la capa de síntesis acumulativa del proyecto. Para
consultas sobre el mapa global, relaciones entre frentes, roadmaps o alternativas:

1. leer `Documents/05_WIKI/index.md`;
2. usar `LLM_CONTEXT.md` para contexto integral;
3. volver a las fuentes enlazadas antes de afirmar números o decisiones;
4. actualizar la wiki cuando cambie el estado, roadmap, framing o arquitectura
   de un frente;
5. ejecutar `venv/bin/python scripts/lint_phideus_wiki.py --write-catalog`.

La wiki no reemplaza informes ni artefactos, no funciona como bitácora o backlog
y no declara GO/NO-GO. Toda página de frente separa estado de arquitectura,
experimento, evidencia y decisión según `Documents/05_WIKI/SCHEMA.md`.

## Estilo narrativo explicativo (objetivos obligatorios)

Para estos documentos, el estilo debe ser narrativo-explicativo (tomando como referencia tonal
`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md`):

- `README.md`
- `Documents/00_TRONCAL/bitacora_desarrollo.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/**`

Condiciones:

- Mantener intacto el diseño visual ya adoptado en cada documento.
- `README.md` y `MARCO_EPISTEMOLOGICO_PHIDEUS.md` funcionan como anclas de tono para la capa pública canónica: narrativa explicativa, continuidad argumental, estado actual y voz conceptual clara; no reducirlos a changelog o minuta salvo pedido explícito del usuario.
- No forzar este tono narrativo en documentación operativa fuera de estos objetivos, salvo pedido explícito.

## Operación con Orca

1. Elegir el mecanismo por responsabilidad: subagente interno para fan-out breve y efímero; worker Orca cuando importen Task/Dispatch verificables, supervisión, continuidad, preguntas, otro workspace o cierre auditable; handoff simple sólo si el receptor asume ownership completo.
2. Todo worker nuevo usa esfuerzo `high` por defecto. `xhigh` exige dificultad o valor diagnóstico concreto y justificado; una auditoría no lo habilita automáticamente.
3. Para workers cross-workspace, indicar siempre el `--worktree` destino explícito, también al reutilizar `--terminal`. Ante `terminal_worktree_mismatch`, verificar primero `worktreeId` y selector; no matar ni recrear recursos por reflejo. El mensaje recursivo `017` corrige la interpretación amplia de `014`.
4. Continuidad e independencia son incompatibles: reutilizar terminal para continuaciones dependientes; lanzar agente y terminal nuevos para auditorías ciegas, réplicas o arbitrajes.
5. Al cerrar un frente, inventariar y cerrar sólo recursos propios por su protocolo y handle exacto; verificar después con `tab list`, `terminal list`, estados de workers/worktrees y, si aplica, listeners. Nunca cerrar recursos ajenos o de ownership incierto.
6. Un hallazgo reproducible de Orca se informa con versión, reproducción mínima, esperado/real, IDs, receipts y workaround. La decisión de abrir issue o PR upstream pertenece al administrador raíz.
7. `m2-alert` se usa sólo si la intervención de Mariano es indispensable después de agotar recuperaciones autónomas. Antes se publica una nota inmutable en inbox con `request_id`, evidencia, próximo responsable y condición de reanudación; Telegram no sustituye inbox, Git ni bitácora.

## 📡 Mensajes recursivos (estructura /mnt/m2-1TB)

<!-- puntero-mensajes-recursivos: no duplicar; canal en /mnt/m2-1TB/MENSAJES_RECURSIVOS.md -->
Este proyecto es parte de la **estructura multi-agente de `/mnt/m2-1TB`**: cada proyecto tiene sus instancias de agentes, y hay un Claude y un Codex **administradores** cuya raíz es `/mnt/m2-1TB` (mantienen la memoria colectiva y la infra común). Su canal de directivas es **`/mnt/m2-1TB/MENSAJES_RECURSIVOS.md`** (append-only, numerado, solo lo escriben ellos).

**Al arrancar sesión:** leé ese archivo. Si hay mensajes posteriores al último que este proyecto integró, leelos, **interpretalos y administralos como mejor convenga a tu contexto singular** (memoria, bitácora, directivas locales), y dejá registro en la bitácora del proyecto: `mensaje recursivo NNN integrado`.
