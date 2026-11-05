# Phideus — instrucciones locales

## Alcance, precedencia y entrada

Este contrato gobierna Phideus. Se complementa con el global efectivo de Codex y el [contrato raíz de M2](../AGENTS.md), que conserva canales, procedimientos de máquina y política común. Dentro de esa autoridad, AGENTS gobierna los adaptadores del agente, incluido [CODEX.md](CODEX.md); un adaptador no modifica el contrato del proyecto.

Al iniciar sesión o tarea, recuperar el contexto en este orden:

1. Este AGENTS.
2. El adaptador del agente activo: para Codex, CODEX.md.
3. La memoria propia indicada en la sección de adaptadores.
4. [Estado del proyecto](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) e [índice documental](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md).
5. [Roadmap BIAS_CONTROL](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md), como contexto del frente; leerlo no lo reanuda.
6. Para el mapa de relaciones, [wiki](Documents/05_WIKI/index.md) y [contexto integral](Documents/05_WIKI/LLM_CONTEXT.md); volver a las fuentes pertinentes antes de afirmar cifras o decisiones.

CASMI tiene raíz, contratos y estado propios: entrar por [AGENTS CASMI](../Phideus-CASMI/AGENTS.md). Allí están el objetivo molecular, el roadmap aprobado, el protocolo de evaluación y las fuentes de permisos, modelos, contador y próximos pasos. Phideus no mantiene otra copia editable de esas decisiones. La retoma CASMI no reanuda otros frentes ni el goal histórico de audio. Las autorizaciones situadas de los demás frentes se consultan en el estado Phideus.

El [registro051](Documents/00_TRONCAL/RECONCILIACION_INSTRUCCIONES_20261104.md) conserva la conciliación anterior. La [entrada de restauración de instrucciones](Documents/00_TRONCAL/bitacora_desarrollo.md#restauracion-de-instrucciones-propias-20261105) identifica los estados sustituidos y el protocolo histórico del respaldo. Se recupera todo el contenido propio no redundante; los antecedentes no se convierten en permisos vigentes.

## Investigación situada y acción experimental

Antes de nuevas arquitecturas o encargos de investigación, leer las síntesis de [caja de herramientas](Documents/05_WIKI/foundations/cajas-de-herramientas.md), [lectura y transferencia](Documents/05_WIKI/foundations/lectura-y-transferencia.md) e [investigación-acción](Documents/05_WIKI/foundations/investigacion-accion.md).

La prioridad del régimen Phideus es el diseño arquitectónico y la experimentación concreta sobre nuevas olas bibliográficas. Usar primero wiki, corpus local y artefactos acumulados para proponer y falsar arquitecturas; registrar una carencia teórica no abre automáticamente una campaña. También se admite exploración conceptual acotada antes de definir un experimento: explicitar qué se busca, qué podría cambiar y cuándo revisar la búsqueda. No exigir utilidad inmediata a cada hallazgo ni reabrir campañas expansivas por inercia.

Aplicar el examen crítico también a HIT, Phideus, benchmarks y propuestas del agente o del usuario. Revisar alineación al cerrar un ciclo y ante bifurcaciones; los resultados y auditorías pueden justificar cambiar la dirección. La referencia a Foucault es una orientación formulada por Mariano: no atribuir citas o una genealogía filosófica precisa sin consultar originales.

Antes de inferir causalidad, priorizar comparabilidad estricta de receta, schedule y protocolo. Una diferencia de diseño no puede ocultarse bajo una comparación presentada como equivalente.

Cuando un plan aprobado o una autorización sostenida cubran un frente de Phideus, producir la evidencia experimental necesaria: diseñar, implementar y ejecutar autónomamente los experimentos CPU pertinentes dentro de ese alcance. Incorporar GPU cuando sea necesaria y esté autorizada, avisando previamente alcance, duración y recursos estimados para coordinar el dispositivo compartido. Ese aviso no concede recursos ni convierte una corrida ya autorizada en una nueva consulta de permiso. Producir evidencia para que Mariano decida GO/NO-GO no equivale a promover por cuenta propia una arquitectura ni clausurar la hipótesis.

Mendieta conserva una autorización situada sólo para Phideus fuera de CASMI, localizada en el estado vigente. Cada despacho debe justificarse por costo, urgencia y valor diagnóstico, cuando tolere cola larga o convenga descargar recursos locales. No usarlo ni consultarlo por rutina; no extender ese permiso a CASMI u otros proyectos ni reanudar un frente pausado.

## Criterios científicos de Phideus

- Escalón1 sustenta la mecánica descriptor-guided y la reorganización geométrica observada; no clausura por sí solo la tesis fuerte de armonía natural.
- Escalón2 formula explícitamente esa hipótesis en coordenadas físicamente naturales del fenómeno. Separar dinámica temporal del oscilador, armonía natural intra-frame, controles no-ratio y comparaciones perceptuales/logarítmicas.
- Preferir invariantes físicos como descriptores primarios de esta investigación. Las representaciones perceptuales o logarítmicas son brazos comparativos explícitos, no el default. No confundir utilidad ingenieril con validación de HIT ni tratar ratio, relacional, natural, armónico y perceptual como equivalentes.
- Comenzar nuevos dominios con baselines trazables y comparables antes de asimetrías fuertes o foundation encoders. Una excepción requiere decisión específica autorizada, no una licencia implícita del agente.

Estos criterios de la investigación Phideus no imponen una arquitectura a CASMI; su canon decide qué herramienta aporta a la identificación molecular.

## Encargos acotados y aceptación de resultados

Para inventarios y extracciones mecánicas, acotar una pregunta y una familia de archivos o fuentes. Son encargos pertinentes inventariar símbolos, referencias o artefactos; extraer manifests o métricas ya autorizadas a JSON/CSV; señalar candidatos de contradicciones, enlaces o tests; ejecutar checks CPU definidos y finitos; o aplicar cambios mecánicos pequeños con allowlist sobre archivos no congelados. No delegar un comando que rg, un parser o un linter resuelvan directamente si el agente no aporta cobertura o revisión útil.

El encargo mecánico no autoriza editar fuentes congeladas, checkpoints, splits, seeds, recibos o artefactos canónicos ni abrir truth/holdouts antes del sello previsto. Puede señalar candidatos en lectura autorizada; no le transfiere decisiones de métricas, loss, arquitectura, GO/NO-GO, interpretación de HIT, seguridad, borrados, merges o despliegues. Esta frontera corresponde a esa clase de encargo, no restringe al coordinador o auditor que tengan expresamente otra responsabilidad autorizada.

Cada despacho concreta entradas, salidas, archivos permitidos, comprobaciones y condición de cierre. Para aceptar su entrega:

- Un inventario informa path, línea y cobertura.
- Una extracción científica conserva fuente, hash de procedencia y esquema, con cotejo contra el origen. Esto no impone SHA a las comprobaciones ordinarias de backup, cuyo contrato es distinto.
- Un cambio entrega diff y salida cruda de los checks realmente ejecutados, incluido exit code. El coordinador revalida e integra; plausibilidad o autoauditoría no equivalen a aceptación.
- Un fallo material, ambigüedad de fuente, contexto insuficiente o conflicto de ownership se eleva al coordinador. No alterar silenciosamente el protocolo para hacer pasar un test. Cerrar antes si se agotaron las unidades pertinentes y pasan los checks.

En investigación, separar encargos por pregunta y fuente, abrir originales y priorizar papers, documentación y repositorios oficiales frente a comentarios secundarios. Cada claim conserva localización, condiciones y limitaciones, distinguiendo cita, extracción e inferencia. Los informes van a `Biblioteca/<tema>/`; la conservación de crudos y URLs sigue el global. Las lecturas integrales hermenéuticas, de física, geometría, arquitectura o loss y el arbitraje de contradicciones requieren lectura contextual del coordinador o auditor independiente; la coincidencia de exploradores no certifica verdad.

Cuando se evalúe una política de delegación, comparar candidato inicial, correcciones y verificador; registrar modelo, esfuerzo, alcance, contexto, tiempos observados, errores y resultado en la Biblioteca. No transferir latencias de otro proyecto ni una prueba aislada como capacidad general. Cambiar la política sólo ante evidencia material. Registrar la clase de encargo al elegir el mecanismo y no sustituir silenciosamente un modelo que el mecanismo no exponga: resolverlo dentro de la política vigente o declarar la limitación.

La selección de modelos se hereda de la política situada vigente, no del catálogo del respaldo. El antiguo protocolo Spark y sus límites numéricos, esfuerzos y rondas se conservan íntegros en la entrada histórica enlazada; no se reactivan ni se transfieren automáticamente a Sol u otros modelos. Tampoco una política escrita equivale a una validación empírica local.

## Artefactos científicos y archivo principal

Desde la regla del28/6, todo experimento nuevo conserva por defecto los estados reutilizables necesarios para reanálisis sin repetir forward o training innecesariamente. El mínimo, salvo justificación explícita, comprende:

- Checkpoints reproducibles: `last_epoch` siempre; intermedios o best según el protocolo del frente.
- Crudos de evaluación por unidad analítica, como matrices o logits por mezcla, utterance o escena; no sólo métricas agregadas o vistas reducidas.
- Configuraciones, seeds, manifests, splits y versiones de dataset/modelo suficientes para reproducir el experimento.
- Identificación de artefactos canónicos, diagnósticos y regenerables.

Si una métrica, calibración, regla de clustering, umbral, bootstrap o análisis alternativo puede necesitarse razonablemente después, guardar durante el experimento el estado que lo permita. La purga selectiva de artefactos pesados se decide al cierre formal del frente, después de que la auditoría confirme qué hace falta conservar y con la autorización aplicable.

El archivo principal de Phideus y CASMI reside en RAID1 desde la orden4/11; M.2 conserva el conjunto activo CASMI y el plano documental. Aplicar la [política de almacenamiento](../Phideus-CASMI/docs/STORAGE_POLICY.md): retirar una réplica M.2 ya respaldada y comprobada no equivale a purgar la evidencia canónica del RAID. Esta política no reanuda frentes pausados.

## Adaptadores y memoria propia

La memoria de Codex es `.codex/memory.md`. No inspeccionar la memoria privada de otro agente ni `.claude/memory.md`. Un `CLAUDE.md` preexistente sólo puede consultarse como antecedente documental, no como contexto administrativo vigente; no editarlo ni recrearlo. Leerlo no restablece el antiguo reparto por proveedor. Ese reparto y el modo de colaboración OFF/ON son antecedentes, no condiciones para conducir hoy el alcance autorizado.

La antigua versión de `phideus-doc-maintainer` está retirada del uso ordinario por orden8/9: mantener documentación desde las fuentes actuales. Releer Mandinga y su protocolo de auditorías no reinicializa ni reorganiza el workspace ni autoriza usar la skill deprecada `codex-audit-loop`.

## Documentación, wiki y tono

Mantener la [bitácora](Documents/00_TRONCAL/bitacora_desarrollo.md) y las páginas pertinentes de la wiki cuando cambien estado, roadmap, framing o arquitectura. La wiki conserva síntesis relacionales, tensiones, preguntas y enlaces a originales; no reemplaza informes ni artefactos, no es bitácora o backlog y no declara GO/NO-GO. Cada página de frente separa estado de arquitectura, experimento, evidencia y decisión según [SCHEMA.md](Documents/05_WIKI/SCHEMA.md). Al modificarla, ejecutar `python3 scripts/lint_phideus_wiki.py --write-catalog`, con Python3 y PyYAML disponibles.

Cuando cambien descriptores, roadmap o framing, sincronizar el [informe histórico](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md), el [catálogo narrativo](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md) y el marco epistemológico si queda afectado. No editar `BACKPROPAGANDO_PHIDEUS.md` sin pedido explícito.

README, estado, bitácora y `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/**` conservan prosa narrativa explicativa y el diseño visual ya adoptado. `README.md` y `MARCO_EPISTEMOLOGICO_PHIDEUS.md` son anclas de tono: continuidad argumental, estado actual y voz conceptual clara; no reducirlos a changelog o minuta salvo pedido explícito. Referencia adicional: [explicación gate4.2](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md). No forzar ese tono en documentos operativos fuera de esos objetivos. Mantener el diseño visual reforzado de README, índice, estado y roadmap BIAS_CONTROL sin extenderlo por rutina a cada archivo.

La documentación pública presenta el estado canónico actual, con rutas relativas o placeholders públicos como `$REPO`, `$HOME` y `$USER`; no un diario de negociaciones internas ni rutas privadas de máquina. Excluir memoria privada, continuidad operativa y tooling interno. `Documents/Skills/` sólo contiene skills estables, reutilizables y exportables fuera del flujo interno. Versionar scripts científicos de experiments/src/tests/scripts salvo exclusión explícita del usuario.

## Operación entre contextos

Al usar Orca, seguir primero la skill y el protocolo vivo que indica el contrato raíz. Elegir el mecanismo por responsabilidad: subagente interno para fan-out breve; worker supervisado cuando importen continuidad, preguntas y ciclo de vida verificables; handoff sólo cuando el receptor asuma ownership completo.

En trabajo entre workspaces, verificar el destino explícito incluso al reutilizar terminal. Si aparece `terminal_worktree_mismatch`, comprobar worktreeId y selector antes de matar o recrear recursos. Reutilizar contexto para continuaciones dependientes; para auditorías ciegas, réplicas o arbitrajes abrir instancia independiente y, si se realiza con Orca, terminal nueva conforme al protocolo. Los comandos concretos y su disponibilidad se verifican en el runtime.

La integración de mensajes recursivos, informes de hallazgos de Orca, cierres por handle y avisos indispensables a Mariano sigue el contrato raíz, sin duplicar sus canales aquí. La recuperación de este respaldo no abre una campaña de mensajes, reinstala modelos ni altera servicios o permisos.
