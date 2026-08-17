# Protocolo Operativo Codex ↔ Claude

> [!NOTE]
> **Documento interno de operación.** Se conserva como protocolo de roles, pero no forma parte de la navegación pública ni del estado científico de Phideus. La fuente normativa vigente para agentes es `AGENTS.md`; una instrucción explícita del usuario prevalece sobre este reparto por defecto.

**Estado**: vigente  
**Actualizado**: 2026-03-21  
**Alcance**: Escalón 3 como regla inmediata y frentes siguientes como criterio por defecto, salvo instrucción explícita del usuario.

## Propósito

Este protocolo no intenta mezclar a Codex y Claude en una sola voz. Hace lo contrario: fija una división de trabajo estable para aprovechar mejor las fortalezas reales de cada agente sin perder trazabilidad metodológica.

La regla de fondo es simple:

- **Codex** debe concentrarse en método, semántica experimental, auditoría y documentación.
- **Claude** debe concentrarse en implementación operativa, ejecución, uso de recursos y corridas largas.

Cuando esa división se respeta, Phideus gana dos cosas a la vez:

1. una capa científica más rigurosa en diseño, métricas y lectura de resultados;
2. una capa técnica más sólida para correr, monitorear y sostener experimentos reales.

## Reparto por defecto

### Codex

Responsable principal de:

- formulación de hipótesis;
- diseño experimental;
- semántica de métricas;
- definición de positivos / negativos / pools / splits;
- criterios `GO / NO-GO`;
- auditoría de implementación;
- auditoría de resultados;
- separación explícita entre observación, hipótesis e inferencia;
- documentación canónica y trazabilidad.

### Claude

Responsable principal de:

- implementación operativa de scripts;
- CLI, logging y barras de progreso;
- tuning técnico de `batch_size`, `num_workers`, `device`, memoria y rutas;
- ejecución de corridas;
- monitoreo en `tmux`;
- recuperación ante fallos;
- manejo práctico de checkpoints y artefactos;
- repetición de runs y relanzamientos comparables.

## Workflow recomendado

### 1. Codex define el experimento

Codex entrega una especificación breve, ejecutable y metodológicamente cerrada:

- pregunta experimental;
- protocolo;
- métricas;
- condiciones de comparabilidad;
- criterio de decisión;
- riesgos metodológicos;
- qué no puede cambiar sin invalidar la comparación.

Entregable esperado:

- un plan corto y preciso;
- una lectura clara de qué cuenta como evidencia y qué no.

### 2. Claude implementa y ejecuta

Claude toma esa especificación y la traduce en una corrida real:

- implementa el script;
- arma CLI y logs;
- deja monitoreo utilizable;
- corre el experimento;
- guarda artefactos consistentes.

Entregable esperado:

- script funcional;
- comando exacto;
- logs monitoreables;
- paths de salida claros;
- artefactos finales reproducibles.

### 3. Codex audita

Con el diff o los resultados en mano, Codex vuelve a entrar para:

- detectar bugs metodológicos;
- revisar si las métricas significan lo que dicen significar;
- auditar si el claim se sostiene;
- decidir si el gate abrió de verdad o no;
- proponer correcciones concretas si hace falta rerun.

Entregable esperado:

- findings priorizados;
- veredicto;
- lectura separada en observación / hipótesis / inferencia.

### 4. Claude corrige y relanza

Si la auditoría detecta problemas:

- Claude aplica el parche;
- relanza;
- conserva trazabilidad;
- deja un artefacto canónico limpio.

### 5. Codex cierra documentación

Cuando el resultado ya está estable:

- roadmap;
- criterios;
- README del frente;
- estado actual;
- bitácora;
- y documentos transversales si cambió el framing científico.

## Regla de ownership

No conviene que ambos agentes editen el mismo archivo durante la misma fase de trabajo.

La regla operativa recomendada es:

- **Codex diseña y audita**;
- **Claude implementa y corre**;
- **Codex interpreta y documenta**.

Si un archivo está en fase de ejecución, su ownership práctico debería quedar del lado de Claude.  
Si un documento está en fase de cierre metodológico o lectura canónica, su ownership práctico debería quedar del lado de Codex.

## Cuándo no hace falta involucrar a Codex

Claude puede avanzar directo cuando la tarea sea:

- puramente técnica;
- un relanzamiento idéntico;
- tuning de recursos;
- monitoreo;
- organización de logs;
- `tmux`, barras de progreso, housekeeping operativo;
- bugs de implementación sin impacto metodológico.

## Cuándo no hace falta involucrar a Claude

Codex puede avanzar directo cuando la tarea sea:

- interpretación de resultados;
- corrección de criterios `GO / NO-GO`;
- lectura metodológica del frente;
- actualización de roadmap;
- documentación canónica;
- auditoría de validez experimental.

## Regla para scripts mixtos

En scripts donde la semántica científica y la operación están muy mezcladas, la secuencia correcta es:

1. **Codex especifica y audita**;
2. **Claude implementa la versión operativa final**.

Eso aplica especialmente a:

- evaluadores;
- scripts de métricas;
- protocolos de `OOD`;
- gates con criterio experimental fino;
- y cualquier script donde un cambio chico pueda alterar la lectura científica.

## Aplicación inmediata en Escalón 3

En Escalón 3, la regla práctica queda así:

- Codex define y audita `P3`, `P4`, criterios de `P5/P6`, lectura de `P2` y framing de probes.
- Claude implementa scripts, corre `P1/P2/P4`, monitorea recursos y mantiene artefactos de ejecución.
- Codex relee resultados y decide si el frente abrió o no el siguiente gate.

En este frente, `P4` es el ejemplo más claro del protocolo: el valor central está en la semántica del gate, pero su ejecución práctica depende mucho más de una capa operativa robusta.

## Regla final

La colaboración más productiva entre ambos no es que los dos hagan de todo.  
Es esta:

- **Codex = método, auditoría, documentación, consistencia**
- **Claude = implementación, ejecución, recursos, monitoreo**

Ese reparto no es decorativo. Es una decisión operativa para aumentar calidad, velocidad y claridad en Phideus.
