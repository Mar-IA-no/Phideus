# Continuación CPU con inventario de fuentes por fase

2026-09-15. Diseño previo a implementación y ejecución. No modifica el
protocolo científico, las predicciones ni el operador prospectivo activo.

## Problema reproducible

El supervisor congelado construye `sources()` desde los módulos `src`
cargados en `sys.modules`. Antes del sello no importa el lector de targets.
La evaluación lo importa localmente después de autenticar el sello, pero
el callback de cierre vuelve a exigir la misma lista de módulos que antes
del sello. Una carga legítima puede fallar esa igualdad aunque ningún archivo
haya cambiado. Las pruebas que sustituyen `sources()` por una lista constante
no reproducen ese ciclo de vida.

La simulación aislada con un módulo inerte, sin importar el lector ni abrir
datos, cambia el inventario de 84 a 85 fuentes. El freeze no contiene
`generative_evidence_supervision.py`; el perfil CPU de cierre anterior al
test sí conserva su hash y el de `observable_rival_evaluation.py`.

## Cambio mínimo y frontera

Crear un entry point separado, `run_geometric_decision_postseal.py`, sólo
para `evaluate` y `replay`. No editar el supervisor, los módulos científicos,
el protocolo, el freeze, los perfiles ni ningún resultado existente. No
eliminar entradas de `sys.modules`, sustituir verificadores por monkeypatch
ni ignorar desigualdades de fuentes.

1. Exigir CPU de una hebra, lock exclusivo y cierre COMPLETE prospectivo;
   replay exige además evaluación COMPLETE. Reutilizar la admisión existente
   y su reserva, sin imports tardíos durante admisión o reapertura del freeze.
2. Reutilizar `prepare_freeze` para comprobar todos los inputs y el freeze
   original. No crear un freeze de reemplazo ni reetiquetar los tests.
3. Autenticar las dos fuentes tardías mediante el binding del perfil de
   cierre ya validado, no mediante un hash recién tomado como única autoridad.
   Rechazar ausencia, cambio o cualquier fuente tardía adicional.
4. Publicar un contrato suplementario inmutable con freeze, perfil de origen,
   las dos fuentes tardías, este plan y el nuevo entry point. Referenciarlo
   explícitamente en los manifiestos y outputs de evaluación/replay.
5. El callback de ejecución revalida todos los hashes originales, protocolo,
   freeze, contrato suplementario y código del entry point. El inventario
   cargado debe ser exactamente el original en PRE y exactamente su unión
   con las dos fuentes tardías en POST. Verificar también sus hashes aunque
   todavía no estén cargadas. No ampliar la lista por observación del runtime.
6. Reutilizar sin cambios `evaluate_fresh` y su gate de sello: sólo ese puerto
   importa el lector y abre targets después del sello. No preleer respuestas
   para probar el entry point.
7. Mantener la recuperación observable post-sello del replay mediante las
   mismas funciones y manifiesto exigidos por el supervisor congelado. Esa
   recuperación no requiere fuentes tardías ni cambia fitting/predicciones.
   Su callback exige el inventario original exacto antes y después.
   El manifiesto posterior de replay de métricas enlaza su cierre y el
   contrato suplementario; ninguno sustituye al otro.
8. Mantener reservas acumuladas y presupuesto original: admisiones y métricas
   bajo evaluación; recuperación observable bajo fresh. El trabajo de
   bootstrap/verificación previo se carga desde el comienzo del proceso.
   No resetear el ledger ni usar este cambio para autorizar reintentos.
9. La admisión del reporte exige el mismo contrato suplementario en los
   manifiestos y outputs COMPLETE de evaluación y replay, autentica su origen
   en el perfil/freeze y revalida todos sus hashes. No exige que el proceso
   lector del reporte tenga el grafo de ejecución científica: no importa
   targets y añade legítimamente sus propios helpers descriptivos.

## Verificación y cierre

Probar con fuentes y módulos inertes las transiciones de inventario, una
fuente tardía alterada, una fuente inesperada y la pérdida de una original.
Probar los gates, la referencia suplementaria y el presupuesto compartido
sin datos experimentales, modelos, sampler, fitting, CUDA o targets reales.
Revisar independientemente la implementación antes de usarla. No ejecutar
el antiguo `--stage evaluate/replay` mientras conserve el fallo identificado.

El informe final registra esta corrección de ejecución como tal. No implica
otra arquitectura, otra métrica, ajuste por resultados ni promoción científica.
