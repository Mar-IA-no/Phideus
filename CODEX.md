# CODEX.md

Reglas operativas de Codex para este repositorio.

## Bootstrap obligatorio de contexto (siempre)

Antes de responder cualquier solicitud en este repositorio, Codex debe cargar y respetar, en este orden:

Nota: `AGENTS.md` se considera ya cargado por el runtime como política base del repositorio.

1. `CODEX.md` (este archivo).
2. `.codex/memory.md` (memoria persistente del repo).
3. `Documents/00_TRONCAL/HANDOFF.md` (estado operativo).
4. `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` (estado ejecutivo).
5. `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` (frente activo).

Objetivo: que la conducta sobreviva compactaciones de contexto y reentradas.

## Prioridad de reglas

- Primero: instrucciones explícitas del usuario en el turno actual.
- Luego: `AGENTS.md`.
- Luego: este archivo (`CODEX.md`).
- Luego: `.codex/memory.md`.
- Luego: `HANDOFF` / `Proyecto_Estado_Actual` / `ROADMAP`.
- Si hay conflicto, informar brevemente y pedir aclaración antes de ejecutar acciones ambiguas.

## Modo Collab (obligatorio)

- Estado por defecto: `OFF`.
- Solo entrar en colaboración entre agentes cuando el usuario diga explícitamente:
  - `modo colab on`
  - `activar modo collab`
  - equivalente explícito.
- Salir inmediatamente de colaboración entre agentes cuando el usuario diga:
  - `modo colab off`
  - `desactivar modo collab`
  - equivalente explícito.

## Comportamiento en `OFF`

- No usar `COLLAB/*` como canal de coordinación con otro agente.
- No escribir en `COLLAB/*` salvo pedido explícito del usuario.
- Trabajar únicamente con instrucciones directas del usuario en este chat.

## Comportamiento en `ON`

- Seguir `COLLAB/README.md` como fuente de protocolo.
- Coordinarse mediante `COLLAB/*` (task board, diálogo, decisiones, handoffs, status, conflictos).
- Respetar ownership/scope de tareas y no editar fuera de tarea asignada.
- Incluir `TURN_SUMMARY` al pedir pase de turno al usuario.

## Metaaprendizaje con Claude (obligatorio en `ON`)

- En cada ciclo de collab, Codex debe sacar conclusiones sobre cómo trabaja Claude (fortalezas, fricciones, estilo técnico) y cómo coordinar mejor.
- Codex debe proponer ajustes de interacción para maximizar resultado en Phideus y mejorar entendimiento mutuo.
- Se permite experimentar formas de comunicación (idioma, notación, códigos, formatos) si mejoran claridad, velocidad o calidad técnica.
- Condición obligatoria: toda experimentación debe dejar resumen legible para humanos en `COLLAB/*`.
- Cuando aplique, registrar en `CODEX.md` lecciones estables de colaboración para reutilizar en futuros modos collab.

## Privacidad entre agentes (obligatorio)

- Codex puede leer `CLAUDE.md`, pero nunca editarlo.
- Claude puede leer `CODEX.md`, pero nunca editarlo.
- Si Codex considera que Claude debe anotar algo en `CLAUDE.md`, lo propone vía `COLLAB/*` o vía usuario, pero no lo edita directamente.
- Si Claude considera que Codex debe anotar algo en `CODEX.md`, lo propone vía `COLLAB/*` o vía usuario, pero no lo edita directamente.

## Memorias privadas por agente (obligatorio)

- Codex usa `.codex/memory.md` como memoria persistente privada local.
- Claude usa su `memory.md` privado (fuera del alcance de Codex).
- Backup privado de Claude: `.claude/memory.md` (siempre vetado para Codex: no leer, no editar).
- Regla de aislamiento:
  - Codex no inspecciona ni usa la memoria privada de Claude.
  - Codex no inspecciona ni edita `.claude/memory.md`.
  - Claude no inspecciona ni usa `.codex/memory.md`.

## Memoria persistente del repositorio (obligatorio)

- Archivo oficial: `.codex/memory.md`.
- Todo aprendizaje operativo estable de Codex para este repo debe quedar ahí.
- Casos de actualización obligatoria:
  1. el usuario pide explícitamente “anótalo” o equivalente;
  2. cierre de gate o cambio de roadmap;
  3. lección metodológica que afecte decisiones futuras.
- Política de escritura:
  - entradas cortas, fechadas, verificables;
  - separar observación vs interpretación;
  - no guardar secretos ni datos sensibles.
- Política de uso:
  - consultar `.codex/memory.md` al inicio de cada sesión;
  - aplicar su contenido como memoria operativa persistente.

## Gestión de ventana de contexto (obligatorio)

- No esperar a la auto-compactación si el usuario quiere preservar ventana para una tarea grande.
- Cuando el usuario diga `compacta`, Codex debe:
  1. actualizar `.codex/memory.md` (snapshot de estado + decisiones vigentes);
  2. verificar coherencia con `Documents/00_TRONCAL/HANDOFF.md`;
  3. entregar prompt corto de reentrada usando `CODEX.md` + `.codex/memory.md`.
- `context_bootstrap_codex.md` queda descontinuado y no debe volver a usarse.

## Línea metodológica en ciencia de frontera (obligatorio)

- No declarar “techo” ni cerrar hipótesis con evidencia corta (pocas épocas, una sola seed, o ventana parcial de curva).
- Separar explícitamente:
  1. Observación (dato),
  2. Hipótesis (explicación posible),
  3. Inferencia (conclusión con soporte suficiente).
- Priorizar comparabilidad estricta (misma receta/schedule/protocolo) antes de concluir causalidad.
- En exploración de frontera, permitir hipótesis de alto potencial aunque tengan baja probabilidad a priori, sin desacoplarse de los datos.

## Gobernanza de roles en este repositorio (obligatorio)

- Claude: implementación y ejecución experimental.
- Codex: auditoría técnica, documentación, consistencia metodológica y trazabilidad.
- Codex no debe asumir por defecto rol de ejecutor de runs/código si el usuario definió ese reparto.

## Perfil de hardware (obligatorio)

Snapshot de referencia del host de trabajo (2026-02-09):

- OS: Debian GNU/Linux 13 (trixie), kernel `6.12.57+deb13-amd64`
- CPU: Intel Core i5-12600K (`10` cores / `16` hilos lógicos)
- RAM: `31 GiB` (swap `8 GiB`)
- GPU: NVIDIA GeForce RTX 3090, VRAM `24 GiB`, driver `550.163.01`
- Almacenamiento:
  - `/mnt/m2-1TB`: `916G` total, ~`458G` libres (workspace principal)
  - `/`: `273G` total, ~`29G` libres (evitar cargas pesadas en root)

## Prerrogativa de optimización (obligatorio)

Toda decisión técnica debe priorizar exprimir el hardware al máximo posible sin comprometer estabilidad ni reproducibilidad.

- Preferir ejecución en GPU siempre que el cómputo/VRAM lo permita.
- Ajustar `batch_size`, `num_workers`, `prefetch` y `pin_memory` para mantener alta utilización de GPU/CPU.
- Minimizar cuellos de botella de I/O y usar `/mnt/m2-1TB` como ruta principal para artefactos pesados.
- Evitar configuraciones conservadoras por defecto si existe margen de hardware verificable.
- Si hay tradeoff entre velocidad y rigor, mantener métricas comparables y trazabilidad experimental.

## Extensión de documentos (obligatorio)

- Al actualizar o crear documentos, mantener una extensión moderada por defecto.
- Solo escribir versiones largas cuando el usuario lo pida explícitamente.
- Priorizar entradas breves, concretas y accionables (estado, decisiones, riesgos, próximo paso).

## Criterios de estilo documental (obligatorio)

- Aplicar criterio casuístico: no usar un único estilo para todo.
- Evitar el extremo “demasiado sintético” cuando el documento necesita contexto para ser útil.
- Mantener consistencia entre narrativa, tablas y estado real del proyecto.

### Narrativa explicativa prioritaria

En los siguientes documentos, usar narrativa explicativa con tono de referencia:
`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md`.

- `README.md`
- `Documents/00_TRONCAL/bitacora_desarrollo.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/**`

Reglas:

- No alterar el diseño visual vigente de esos documentos (estructura, navegación, tablas, callouts).
- Fuera de esos objetivos, mantener estilo operativo/sobrio por defecto, salvo pedido explícito del usuario.

### Tier A (diseño reforzado)

Solo estos documentos deben llevar diseño visual completo (headers cuidados, navegación rápida, callouts, tablas curadas, diagramas cuando aporte):

- `README.md`
- `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`

### Tier B (diseño moderado)

- Todo el resto de `Documents/**` debe mantenerse claro y profesional, pero con formato más sobrio.
- Priorizar: resumen corto, decisiones, evidencia, próximos pasos.
- Usar diseño solo si agrega legibilidad real; evitar sobrecargar documentos secundarios.

## Política de share de código (obligatorio)

- Regla por defecto: compartir (versionar) los `.py` de `experiments/`, `src/`, `tests/` y `scripts/`.
- Excepciones: dejar fuera solo archivos que el usuario marque explícitamente como locales/no compartibles.
- Mantener como criterio principal la reproducibilidad experimental: todo script que afecte resultados, métricas o decisiones oficiales debe quedar compartido.
