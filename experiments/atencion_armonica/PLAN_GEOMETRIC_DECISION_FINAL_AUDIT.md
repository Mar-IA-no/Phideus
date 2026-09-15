# Auditoría final de energía geométrica para la decisión

2026-09-15. Alcance previo a leer resultados, derivado del protocolo y del
plan de informe. No modifica selección, métricas, contrastes, código congelado
ni presupuesto científico. La evaluación está en curso al redactarlo.

## Dos revisiones independientes

La revisión técnica comprueba procedencia, cobertura, aritmética y fidelidad
del informe. La revisión de horizonte lee después el informe completo y sus
fuentes: qué respondió el experimento y qué siguiente pregunta justifica.
Ninguna revisión promueve arquitectura ni declara GO/NO-GO. Las instancias
son distintas del coordinador que implementó el experimento; la revisión
de horizonte es distinta de quien redacte su interpretación.

Antes del cierre real sólo se permite revisar código, este plan y fixtures
abstractos. La auditoría ejecutable sobre datos reales exige COMPLETE de
prospectivo, evaluación, recuperación observable, replay de métricas e
informe. La presencia de archivos no reemplaza esos recibos. No se consume
ningún resultado parcial para seleccionar las comprobaciones.

El checker separa dos fases irreversibles bajo el mismo lock. En **PRECOMMIT**
lee sólo bindings, receipts terminales hasta `finish -> output`, freeze, sello,
contrato postseal y metadatos observables necesarios para la regla estructural.
No desreferencia `evaluation/complete`, targets, métricas, sidecars,
`report/complete` ni sus resúmenes; tampoco usa helpers que los abran de forma
indirecta. Publica un manifiesto inmutable con split, scene_id, razones,
referencias fuente, roster de probes y hash del plan y del checker. En **VERIFY**
reautentica ese manifiesto antes de abrir los payloads restantes. Autenticar
bytes de un sidecar sin decodificarlo no equivale a leer sus labels.

## Cobertura integral y comprobaciones independientes

1. Autenticar cadena y costes de entrenamiento, selección, archivo de estados,
   freeze, sello global, evaluación, recuperación, replay e informe. Revisar
   continuidad de cargos y versiones, no sólo los estados declarados. Mantener
   explícita la corrección operativa postseal y el origen previo de sus dos
   fuentes tardías. Validar hashes actuales sin exigir que el auditor tenga
   el grafo de imports del operador científico.
2. Comprobar roster de 72 entrenamientos, sus estados inicial/final y selección
   por brazo, y los 144 estados initial/selected evaluados. No volver a entrenar,
   cargar modelos para forward, buscar otro checkpoint ni modificar artefactos.
3. Autenticar todos los artefactos referenciados por los cierres. Verificar las
   2048 identidades, cuatro splits de 512, roster de cabezas por split,
   candidatos/soportes compartidos y escenas vacías conservadas. Distinguir
   autenticación de bytes de recomputación matemática independiente.
4. Seleccionar un corte estructural desde observaciones selladas: por split,
   unión de scene_id 0, 511, primera escena vacía si existe y primera escena
   con máximo número de candidatos. Empates por scene_id; deduplicar, sin
   sustituciones. Añadir los hasta cuatro originales exactamente enumerados en
   `roundtrip_scene_ids` y todos sus derivados sellados; comprobar que son los
   primeros min(4, elegibles), sin sustitución. Registrar cardinalidad 0..4 por
   split. Publicar selección y razones en PRECOMMIT, antes de cualquier payload
   de resultados. Es cobertura diagnóstica, no muestra representativa.
5. En ese corte, reconstruir el linaje de labels desde los sidecars sellados
   y la identidad de evento. No generar observaciones ni repetir fitting.
   Calcular entropías/VI y ARI con un método independiente de los helpers de
   métricas del experimento. Comparar valores float64 con atol 1e-12 y rtol 0
   en su dominio normalizado; registrar discrepancias, no corregir respuestas.
   La tolerancia no autoriza igualdad de óptimos: empates y elecciones se
   validan exactamente con la aritmética canónica preservada. Comprobar por
   separado u32 = float32(u64 canónico), tD = suma64(u32) y tM = suma64(u64).
6. Para todas las cabezas y referencias clásicas de esas escenas, verificar
   elecciones por energía y firma, conjuntos coóptimos, regret, ARI, soporte
   y presencia. Diferenciar score, target matemático y target entregado. Los
   componentes de la loss Decisión no se interpretan como entropías calibradas.
7. Recalcular los cuatro contrastes primarios sobre las matrices completas
   del escenario deformado: nueve celdas dentro de escena, luego escenas
   elegibles. Verificar las 10000 remuestras, índices, distribuciones e
   intervalos predeclarados; no sustituir el estimando ni elegir otra seed.
8. Comprobar todos los probes previstos: transporte y su inversión, margen,
   elecciones, roundtrip por identidad de evento, cambios de universo y
   diagnóstico de coordenadas. Auditar donors, masks y estratos con soporte
   efectivo, incluyendo singleton. Separar comprobación integral de campos
   de recomputación independiente sobre el corte estructural.
9. Contrastar las tablas y afirmaciones numéricas del informe con las fuentes
   autenticadas. Reportar exactamente el alcance comprobado y cualquier
   requisito todavía no verificado; no convertir fixtures en evidencia real.

## Raíces y clausura autenticada

La raíz de campaña es `data/atencion_armonica/geometric_decision_energy_v1`.
El checker declara una tabla de dispatch por schema y campo de referencia;
no infiere ownership por la mera presencia de una clave `path`.

| Raíz o store | Entradas y alcance exigidos |
|---|---|
| `control` | binding, fuentes de prior_charges y ledger contiguo completo de attempts/start/finish; incluir cargos de intentos no COMPLETE y reserva íntegra cuando no existe finish |
| Cierres operativos en `control` | único COMPLETE aplicable de OPEN, training, selection, archive/exclusions, perfiles que autorizan el freeze, prospectivo, observable-replay, evaluate, replay y report; manifests y outputs vinculados |
| Freeze, contrato postseal y sello | procedencia original y tardía, selección, roster, exclusiones, batches y todos los archivos inventariados; igualdad exacta del inventario actual de fresh con el sellado |
| `open`, `profiles`, `training`, `selection`, `archive` | bindings y cierres, agregados de entrada, calibraciones y estados referenciados; roots de celdas de training explícitos, no herencia implícita del store del archivo de cabezas |
| `fresh`, `evaluation`, `report` | bindings y cierres; referencias transitivas tipadas a observaciones, fuentes, inputs, predicciones, probes, targets, scores, primario y reportes por escenario |
| Fuentes y entradas externas declaradas | código/protocolo relativos al repositorio; stores históricos y perfiles sólo por roots exactas declaradas en bindings y manifests, con su clausura de referencias requerida, sin barrer árboles históricos ajenos |

Cada referencia se resuelve como `(store, path, bytes, sha256)` según schema y
campo; las transiciones entre stores se explicitan. Rechazar escapes de raíz,
schema requerido desconocido, referencia requerida no consumida y dos
identidades distintas para el mismo `(store, path)`. Publicar conteos y bytes
autenticados por store y tipo, con exclusiones o requisitos no verificados
expresos. Los estados binarios se autentican como bytes, sin unpickle ni forward;
los arrays necesarios para recomputación sí se decodifican. La lista exacta de
schemas y transiciones forma parte del checker revisado antes de la corrida real.

## Independencia y comparación

El checker implementa por separado targets, selector por firma, coóptimos,
regret, ARI, primarios scene-first, índices/bootstrap/cuantiles y cálculos
derivados de transporte y roundtrip. No llama a los helpers científicos bajo
auditoría para obtener el valor esperado. Puede reutilizar I/O autenticado,
presupuesto y parser canónico de sidecars como autoridad semántica autenticada;
esa dependencia se declara y no se presenta como reconstrucción independiente
de la ley física. Los fixtures incluyen mutaciones rechazadas por cada familia
de comprobaciones, no sólo ejemplos coincidentes.

| Cantidad | Regla de comparación |
|---|---|
| Hashes/bytes, IDs, shapes/dtypes, firmas, elecciones, coóptimos, presencia, donors, masks, soporte | Exacta; ninguna tolerancia crea empates ni cambia el selector |
| u32, tD, tM y energía desde componentes canónicos preservados | Exactamente float32(u64), suma64(float64(u32)), suma64(u64) y suma64(componentes), respectivamente |
| Entropías/VI normalizadas y ARI calculados independientemente | atol 1e-12, rtol 0; separar discrepancia algebraica de último bit de incumplimiento de aritmética canónica |
| Otros floats derivados: regret, componentes de error, agregados, márgenes, distribuciones bootstrap e IC | atol 1e-12, rtol 0, sobre sus unidades declaradas; NaN sólo en posiciones de soporte ausente previstas y null según schema |
| Bootstrap | PCG64, seed 2026091494, 10000 e índices int64 exactos; nueve celdas dentro de escena, escenas después; cuantiles lineales .00625/.99375; tolerancia anterior sólo para floats derivados |
| Transporte | atol 1e-6, rtol 1e-5 únicamente para reproducir la bandera de estabilidad numérica; errores derivados con la regla anterior y elecciones/empates exactos |

No se redondean targets independientes para fabricar coóptimos. El checker
verifica los óptimos exactos desde el target canónico preservado y evalúa por
separado su concordancia matemática con el cálculo independiente.

## Ejecución y límites

El auditor propone su checker separado y sus fixtures; el coordinador revisa
su alcance antes de la ejecución real. El checker fija fuentes, plan y entradas,
usa el lock y ledger comunes, CPU de una hebra y CUDA deshabilitada. Reserva
máxima inicial de 1500 s adicionales a los 600 s previstos para el informe:
máximo nominal conjunto 2100 s, ambos sobre el mismo ledger, sin ampliar el
tope audit de 3600 s ni reiniciar cargos. Antes de crear el intento reconstruye
`remaining = min(3600 - charged[audit], 49200 - sum(charged))`; la reserva y
proyección con margen deben caber. Publicar la proyección por archivos/bytes y
cómputo antes de ejecutar. Si no cabe, documentar antes de ejecutar; no recortar
cobertura ni relanzar silenciosamente. Conservar crudo, hashes, discrepancias y
recibos.

La revisión de horizonte debe distinguir: geometría del fitter frente a
geometría neuronal; prior inicial frente a aprendizaje; efecto conjunto de
bypass y trayectoria frente a mecanismos identificados; utilidad de la loss
frente a validación física; estabilidad numérica frente a invariancia aprendida;
controles con distinta información; evidencia sintética frente a medición real.
Debe poder recomendar cambiar de dirección. El cierre se completa sólo después
de integrar hallazgos materiales, actualizar documentación/wiki y hacer
commit/push. El siguiente goal nace de los resultados, no de este plan.
