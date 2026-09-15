# Perfil CPU de cierre del presupuesto

2026-09-14. Medir las operaciones todavía ausentes del presupuesto prospectivo,
sin producir nuevos tests ni repetir redes o ajustes. El perfil observable
0010 ya está completo y se reutiliza; no se ejecuta otra vez.

Entradas: sus16originales TRAIN y4probes,144predicciones por recorrido,
archivo/selección COMPLETE y primer shard TRAIN autorizado. Leer los bytes de
los dos archivos TRAIN completos de512filas, parsear sólo las16 primeras
entradas y reconstruir su correspondencia con las
observaciones ya conocidas. Una reconstrucción determinista del productor
sobre esas mismas16tuplas TRAIN mide muestreo/serialización; debe reproducir
exactamente tanto observación como sidecar conocido. No usar seeds de tests
ni crear frecuencias nuevas. Las coordenadas del perfil0010 se excluyen por
el helper ya auditado; no se sobrescribe su inventario original.

Medir por fases: admisión/setup, verdad TRAIN, productor/IO sobre tuplas
conocidas, evaluación16originales, evaluación4probes, replay de ambas y hash
del store observable. Medir bootstrap de512filas mediante repetición de
métricas ya conocidas, no repetición de observaciones. Conservar este último
como fixture de coste, nunca como resultado científico o intervalo de test.

CPU con CUDA oculto y una hebra. Reserva360s dentro de los410,06s de perfil
disponibles al diseñarlo; límite120s por fase, guards comunes8GiB/100GiB/30GiB.
Guardar timings/unidades, resultados completos y referencias exactas de replay.
Contabilizar desde lanzamiento; no rehacer silenciosamente intentos parciales.
El binding preliminar sólo enlaza código, protocolo y runtime. La admisión
material de entrenamiento, selección, archivo y datos ocurre dentro de la
fase medida `admission-setup`, con presupuesto y alarmas ya activos; conserva
su propio recibo y pasa el guard vivo a los helpers que lo admiten.

Proyección con25%: originales por2048/16, probes por4escenarios. Para draw/IO
contar cuatro pases por cada unidad como reserva de reapertura de índices
anteriores; no presentar esa aproximación como cota de todos los OOD. Fresh
incluye recorrido observable y recovery previo al sello, más draw/IO y un
inventario global. Evaluación incluye recovery observable posterior, métricas
y su replay, dos bootstrap, reconstrucción de verdad y cinco verificaciones
globales de hashes —entrada/salida de evaluación y replay, más la admisión
previa a la recuperación observable del replay—. El setup observado
se reserva cuatro veces en cada etapa. Añadir tails de finishes al admitir.
Conservar volumen proyectado y scope; comprobar que cada suma entra en el
presupuesto restante antes de congelar y generar tests. Si no entra, revisar
recursos de forma documentada, sin reducir controles ni muestras.
