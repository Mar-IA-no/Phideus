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
   - `Documents/00_TRONCAL/HANDOFF.md`
   - `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
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

## Regla documental transversal (Phideus)

Mantener sincronizados cuando cambien descriptores/roadmap:

- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md`

No actualizar `BACKPROPAGANDO_PHIDEUS.md` salvo pedido explícito del usuario.

## Estilo narrativo explicativo (objetivos obligatorios)

Para estos documentos, el estilo debe ser narrativo-explicativo (tomando como referencia tonal
`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md`):

- `README.md`
- `Documents/00_TRONCAL/bitacora_desarrollo.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/**`

Condiciones:

- Mantener intacto el diseño visual ya adoptado en cada documento.
- No forzar este tono narrativo en documentación operativa fuera de estos objetivos, salvo pedido explícito.
