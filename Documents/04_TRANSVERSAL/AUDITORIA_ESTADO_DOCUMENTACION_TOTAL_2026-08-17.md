# Auditoría del estado documental de Phideus

**Fecha:** 2026-08-17
**Alcance:** documentación versionada del repositorio y puntos de entrada canónicos
**Antecedente:** `AUDITORIA_ESTADO_DOCUMENTACION_TOTAL_2026-04-03.md`

---

## 1. Qué se auditó

La revisión partió de una distinción necesaria: un documento antiguo no está equivocado por conservar el estado desde el cual fue escrito. Phideus necesita preservar sus hipótesis fallidas, sus planes ejecutados y sus lecturas provisionales porque allí también vive la trazabilidad del programa. El problema aparece cuando un snapshot histórico se presenta como estado actual, cuando dos puertas de entrada canónicas sostienen decisiones incompatibles o cuando una pieza interna de coordinación queda expuesta como parte de la documentación pública principal.

Con ese criterio se inventariaron `192` documentos textuales versionados (`.md`, `.rst`, `.txt`), sin incluir archivos reservados al usuario, y `256` documentos textuales presentes bajo `Documents/`. Dentro de `Documents/01_FRENTES_ACTIVOS/` se relevaron `142` archivos Markdown. La cobertura combinó:

- lectura profunda de `README.md`, estado ejecutivo, índice global y documentos transversales de onboarding;
- contraste entre los puntos de entrada y los README/roadmaps canónicos de cada frente activo;
- búsqueda estructural de fechas, estados abiertos o cerrados, claims que todavía se presentan como vigentes y referencias internas a agentes;
- revisión de enlaces locales y rutas absolutas;
- inspección del historial versionado posterior al último corte experimental documentado.

No se reescribieron planes, preregistros, bitácoras históricas ni informes de resultados para hacerlos coincidir retrospectivamente con lo que se supo después. Cuando podían inducir una lectura presente incorrecta, se agregó una señal de alcance en la cabecera o se los retiró de la navegación pública principal.

---

## 2. Veredicto general

La documentación científica de Phideus no estaba globalmente desactualizada. Los resultados centrales de los frentes activos ya habían sido propagados a `README.md` y, en términos sustantivos, a `Proyecto_Estado_Actual.md`:

- Escalón 1 conserva el cierre causal descriptor-guided y la separación explícita entre esa mecánica y la tesis fuerte de armonía natural;
- Escalón 2 registra el null descriptorial bajo `P2/P3` y deja abierta la decisión representacional siguiente;
- Voz Expresiva incorpora el cierre `EN↔ZH`, con transferencia positiva acotada a `N-adapt` y sin réplica robusta en `N-strict`;
- Atención Armónica incorpora `Fase 0`, `0.5` y `0.6`, distingue representación de clustering y acota la ventaja del triángulo a `OOD-poly` bajo clusterers globales;
- Escalón 3 conserva `P2-flat` como baseline general, `P5-cqtshift` como mejor brazo OOD del corte y `P6` como hipótesis no ganadora bajo la receta ensayada;
- BIAS_CONTROL mantiene Gate 5B, Gate 8 y Gate 10 cerrados, y `Exp C` como única rama downstream todavía abierta.

La desalineación estaba concentrada en la arquitectura documental, no en el núcleo de resultados. Había cuatro problemas concretos:

1. `Proyecto_Estado_Actual.md` tenía un encabezado de julio, un pie de junio y una descripción ya superada del cierre de Voz Expresiva.
2. `INDICE_DOCUMENTACION.md` funcionaba simultáneamente como mapa público y como índice de protocolos privados de coordinación entre agentes.
3. `PHIDEUS_MASTER_BRIEFING.md` todavía llamaba a BIAS_CONTROL “el frente activo” y describía su código como “principal actual”, aunque el programa ya es multirfrente.
4. Algunos informes generados en abril conservaban rutas locales rotas o hallazgos ya corregidos sin advertir que eran snapshots históricos.

La segunda pasada sobre documentos enlazados como canónicos encontró además una contradicción de evidencia en `RANKING_DESCRIPTORES_UNIFICADO.md`: su cabecera todavía trataba `d4a4` como referencia eval-seed pendiente de réplica, aunque la réplica real ya había cerrado `5/5`. El mismo residuo aparecía en secciones ejecutivas de `ROADMAP_BIAS_CONTROL.md` y `ROADMAP_UNC.md`.

El historial de Git posterior al 2026-07-02 contiene cambios documentales, no nueva implementación o resultados científicos versionados. Por eso esta auditoría actualiza el estatuto y la navegación de la documentación, pero no inventa un nuevo corte experimental.

---

## 3. Capas documentales resultantes

### 3.1 Estado canónico vivo

Las puertas de entrada vigentes son:

- `README.md`, para la presentación pública y la tesis general;
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`, para el corte ejecutivo y las decisiones abiertas;
- `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`, para navegar la evidencia;
- los README y roadmaps de cada frente, para el detalle metodológico local.

Estas piezas deben cambiar cuando cambia el programa. En esta auditoría se sincronizaron sus fechas y se retiraron de la navegación principal los protocolos exclusivos de agentes. `README.md` no requirió una reescritura: su framing científico ya coincide con los cierres documentados.

### 3.2 Fundamentos transversales

Los documentos filosóficos y epistemológicos tienen otra temporalidad. No deben perseguir cada corrida, pero sí evitar que un ejemplo histórico se confunda con la totalidad actual de Phideus. `PHIDEUS_MASTER_BRIEFING.md` quedó definido como briefing conceptual y técnico del ciclo que condujo a BIAS_CONTROL, no como estado exhaustivo del programa. `PHIDEUS_NEURAL_ARCHITECTURES.md` quedó rotulado como referencia histórica hasta Gate 4.4, porque hoy no incluye las arquitecturas posteriores de voz, geometría no plana y Atención Armónica.

No se modificaron `MARCO_EPISTEMOLOGICO_PHIDEUS.md`, `Elucubraciones_Epistemologicas.md` ni `BACKPROPAGANDO_PHIDEUS.md`: los dos primeros son documentos protegidos y el tercero está reservado para ideas en discusión salvo pedido explícito.

### 3.3 Frentes e historia experimental

Los planes ejecutados, informes fechados y documentos superseded permanecen en su lugar. Que una sección diga “pendiente” dentro de un plan de febrero no constituye una contradicción si el índice actual la presenta como plan histórico y el README del frente contiene el cierre posterior. Esta separación evita una falsa limpieza retrospectiva: el repositorio conserva qué se creía antes de cada resultado.

### 3.4 Operación interna y memoria de agentes

Los protocolos de reparto de roles, handoffs y cuadernos de continuidad pueden seguir versionados cuando sirven a la trazabilidad operativa, pero ya no aparecen como documentación principal del índice público ni dentro del estado ejecutivo. La política pública describe el proyecto; `AGENTS.md` y las memorias privadas gobiernan a los agentes.

---

## 4. Enlaces y rutas

La revisión de enlaces detectó un cluster dominante de referencias rotas en dos informes de sincronización del libro HIT. No se trata de decenas de pérdidas independientes: casi todas apuntan a un antiguo checkout local de `harmonic-information-theory/`, trasladado después a otro repositorio y otra ubicación.

Reescribir esas rutas como si siempre hubieran sido otras destruiría el contexto del handoff. La corrección aplicada fue declarar ambos informes como snapshots históricos y enlazar la fuente pública vigente: [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory). Las rutas absolutas permanecen dentro del cuerpo como evidencia del entorno de abril, pero dejan de presentarse como instrucciones actuales.

El reporte numérico `AUDIT_REPORT_TRAZABILIDAD.md` recibió el mismo tratamiento: conserva el resultado generado el 2026-04-03 y advierte que su único claim `STALE` fue corregido posteriormente. El output generado bajo `tools/claims_audit/` no se modificó manualmente.

---

## 5. Cambios realizados

- Se sincronizó la fecha documental y el cierre narrativo de `Proyecto_Estado_Actual.md`, sin alterar el corte experimental.
- Se convirtió `INDICE_DOCUMENTACION.md` en un mapa público explícito y se retiraron de la navegación principal protocolos, auditorías y planes cuyo valor es exclusivamente interno o histórico de agentes.
- `PROTOCOLO_OPERATIVO_CODEX_CLAUDE.md` recibió una cabecera explícita de uso interno y quedó fuera de la navegación pública principal.
- Se agregó esta auditoría al índice transversal.
- Se actualizó la capa viva de `RANKING_DESCRIPTORES_UNIFICADO.md` al cierre homogéneo `d4a4=84.0%±2.7pp` sobre cinco training seeds, preservando la tabla eval-seed anterior como snapshot histórico.
- Se corrigieron en `ROADMAP_BIAS_CONTROL.md` y `ROADMAP_UNC.md` las instrucciones ya superadas de “correr P3” y “programar d4a4 training-seed”; el roadmap distribuido también reemplazó identidad de agente e IP local por descripciones públicas de carril y placeholders.
- Se corrigió el alcance de `PHIDEUS_MASTER_BRIEFING.md`: BIAS_CONTROL queda como línea fundacional de un ciclo, no como único frente activo.
- Se agregó una advertencia de alcance a `PHIDEUS_NEURAL_ARCHITECTURES.md` para evitar que su inventario hasta Gate 4.4 se lea como catálogo actual completo.
- Se rotularon como snapshots históricos el reporte numérico de abril y los dos informes de sincronización del libro HIT.
- La auditoría documental de 2026-04-03 quedó enlazada explícitamente como antecedente histórico de esta nueva revisión.
- Se integraron en la bitácora los mensajes recursivos `002` a `005`, con verificación local de GitHub, backup, GPU y frescura de la memoria colectiva.

---

## 6. Riesgos residuales y criterio futuro

Phideus conserva una documentación muy densa. Esa densidad es un activo si cada capa declara su estatuto y un costo si cada aparición de “pendiente” se interpreta como tarea viva. La regla práctica que surge de esta auditoría es:

> El presente se corrige en las puertas de entrada; el pasado se contextualiza, no se reescribe.

Quedan tres cuidados permanentes:

1. Todo nuevo cierre experimental debe propagarse primero al README del frente y luego, si cambia el programa, al estado ejecutivo, al índice y al README principal.
2. Los documentos públicos no deben enlazar memorias privadas, protocolos exclusivos de agentes ni rutas machine-specific como si fueran infraestructura compartible.
3. Las auditorías generadas deben incorporar fecha, alcance y condición de snapshot desde su primera emisión, para que un resultado correcto en su corte no se convierta después en una falsa fuente de estado actual.

Esta revisión no declara cerrados los frentes que permanecen abiertos ni altera decisiones `GO/NO-GO`: ordena la evidencia disponible y devuelve a cada documento la función que realmente cumple.
