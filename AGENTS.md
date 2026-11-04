# Phideus — instrucciones locales

## Alcance y entrada

Este contrato gobierna Phideus. Se complementa con el global efectivo de Codex y el [contrato raíz de M2](../AGENTS.md); los canales y procedimientos de máquina se consultan allí. La administración mantiene esas capas. Las instrucciones locales anteriores y su conciliación son recuperables en el [registro051](Documents/00_TRONCAL/RECONCILIACION_INSTRUCCIONES_20261104.md).

Al entrar, consultar [CODEX.md](CODEX.md), la memoria privada propia `.codex/memory.md`, [estado del proyecto](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) e [índice documental](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md). Para síntesis y relaciones, leer [wiki](Documents/05_WIKI/index.md) y [contexto integral](Documents/05_WIKI/LLM_CONTEXT.md); volver a las fuentes del frente antes de actuar. Leer [roadmap BIAS_CONTROL](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) cuando se trabaje en ese frente; su existencia no lo reanuda.

CASMI tiene raíz, contratos y estado propios: [AGENTS CASMI](../Phideus-CASMI/AGENTS.md). Para ese trabajo, entrar por esa fuente; Phideus no mantiene otra copia editable de permisos, modelos, contador o próximos pasos CASMI. Los demás frentes y el goal histórico de audio conservan sus pausas y alcances propios; una retoma CASMI no los reanuda. Las autorizaciones singulares de esos frentes, incluida Mendieta fuera de CASMI, se consultan en el estado Phideus enlazado arriba.

## Fuentes y criterios científicos de Phideus

Antes de nuevas arquitecturas o encargos de investigación del proyecto, leer las síntesis de [caja de herramientas](Documents/05_WIKI/foundations/cajas-de-herramientas.md), [lectura y transferencia](Documents/05_WIKI/foundations/lectura-y-transferencia.md) e [investigación-acción](Documents/05_WIKI/foundations/investigacion-accion.md). Usar el corpus y wiki existentes para experimentos discriminantes; una consulta externa acotada debe responder una incertidumbre o pregunta situada, sin reabrir por inercia campañas bibliográficas.

- Escalón1 sustenta la mecánica descriptor-guided y la reorganización geométrica observada; no clausura la tesis fuerte de armonía natural.
- Escalón2 formula explícitamente esa hipótesis en coordenadas físicamente naturales. Separar dinámica temporal del oscilador, armonía intra-frame, controles no-ratio y comparaciones perceptuales/logarítmicas.
- Preferir invariantes físicos como descriptores primarios de esa investigación; conservar representaciones perceptuales como comparaciones explícitas. No confundir utilidad ingenieril con validación de HIT ni tratar ratio, relacional, natural, armónico y perceptual como equivalentes.
- Comenzar nuevos dominios con baselines comparables antes de asimetrías fuertes o foundation encoders, salvo decisión específica autorizada. Estas reglas del estudio Phideus no imponen una arquitectura a CASMI.

## Artefactos y memoria del proyecto

Un experimento conserva `last_epoch`, checkpoints intermedios/best según protocolo, estados crudos por unidad analítica, configuraciones, seeds, manifests y splits suficientes para reanalizar sin repetir forward/training. Distinguir artefactos canónicos, diagnósticos y regenerables. La retención debe permitir calibración y métricas posteriores; no purgar antes del cierre auditado y la autorización correspondiente.

La memoria propia de Codex es `.codex/memory.md`. No inspeccionar la memoria privada de otro agente ni `.claude/memory.md`. El antiguo modo de colaboración con Claude y el reparto fijo por proveedor son antecedentes, no requisitos para continuar el alcance autorizado. La antigua versión de `phideus-doc-maintainer` está retirada del uso ordinario por orden8/9; mantener documentación desde las fuentes actuales.

## Documentación y síntesis

Mantener la [bitácora](Documents/00_TRONCAL/bitacora_desarrollo.md) y las páginas pertinentes de la wiki cuando cambie un frente. La wiki sigue [SCHEMA.md](Documents/05_WIKI/SCHEMA.md); ejecutar `venv/bin/python scripts/lint_phideus_wiki.py --write-catalog` cuando se modifique. No usarla como backlog ni sustituir fuentes por copias.

Cuando cambien descriptores, roadmap o framing, sincronizar [informe histórico](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md), [catálogo narrativo](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md) y el marco epistemológico si queda afectado. No editar `BACKPROPAGANDO_PHIDEUS.md` sin pedido explícito.

Conservar el diseño y la prosa narrativa explicativa de README, estado, bitácora y teoría transversal; README y marco epistemológico son anclas de tono. Referencia adicional: [explicación gate4.2](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md). Los documentos operativos pueden ser sobrios. Mantener el diseño visual reforzado de README, índice, estado y roadmap BIAS_CONTROL sin extenderlo por rutina a cada archivo.

La documentación pública presenta el estado canónico, usa rutas relativas/placeholders y excluye memoria privada, continuidad operativa y tooling interno. `Documents/Skills/` sólo contiene skills estables y exportables. Versionar scripts científicos de experiments/src/tests/scripts salvo exclusión explícita del usuario.
