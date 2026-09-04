# Plan CPU — fidelidad de un surrogate diferenciable de IRLS

**Fecha:** 2026-09-04  
**Estado:** protocolo congelado antes de la corrida oficial  
**Régimen:** diagnóstico numérico sobre validation; sin entrenamiento ni test  
**Autoridad:** habilita o rechaza una herramienta de pérdida; no promueve arquitectura ni decide GO/NO-GO

## Problema

El refit de `correction_head` redujo el error marginal de relación y, aun así,
degradó IRLS IID. El siguiente entrenamiento sólo tendría sentido si su pérdida
representa la dinámica residual del executor robusto. El IRLS canónico es NumPy
y externo; antes de diferenciar a través de una aproximación Torch hay que
medir dos fidelidades distintas:

1. si una trayectoria de longitud fija aproxima la solución del executor hasta
   convergencia;
2. si el gradiente automático de esa trayectoria coincide con diferencias
   finitas de una implementación NumPy independiente de la misma longitud.

## Universo

El diagnóstico usa exclusivamente validation del smoke. No abre test ni
reentrena. Incluye las `127` vistas para la relación observada y un subconjunto
determinista de `32` vistas, estratificado por tamaño, para las trayectorias de
corrección:

- relación observada con peso unitario;
- trayectorias `observada + alpha × (corregida − observada)` para
  `alpha ∈ {0.25, 0.5, 0.75, 1}`;
- cuatro brazos primarios y dos checkpoints por brazo;
- profundidades fijas `K ∈ {4, 8, 16, 32, 64, 128, 256}`.

La grilla expone tanto el régimen donde IRLS unitario converge rápido como las
correcciones heredadas que pueden volver lenta o fallida la optimización. Cada
estado conserva `view_id`, `master_id`, brazo, seed, alpha y convergencia del
executor externo.

## Tres implementaciones separadas

1. **Executor canónico:** `solve_huber_irls`, con `max_iterations=7500`,
   tolerancia `1e-6`, damping `1.0`, delta y weight floor congelados.
2. **Referencia NumPy fixed-K:** reproduce la actualización por residuos
   durante exactamente K pasos y un solve final, sin early stopping.
3. **Surrogate Torch fixed-K:** misma receta en `float64`, KKT de gauge explícito
   y ninguna lectura de targets privados. Sólo esta rama participa en autograd.

La referencia NumPy no importa helpers del surrogate Torch. Ambas pueden usar
la matriz de incidencia pública, pero construyen por separado normalización,
KKT, residual, Huber, clipping y actualización.

## Fidelidad de valores

Por estado y profundidad se registran:

- RMSE y máximo absoluto de `x_hat_torch − x_hat_numpy_fixed`;
- RMSE y máximo absoluto de pesos finales;
- RMSE y máximo absoluto frente al executor convergido cuando sea evaluable;
- objetivo Huber y cambio de solución;
- proporción y distribución de iteraciones del executor canónico.

La igualdad Torch↔NumPy fixed-K es una prueba de implementación. La cercanía a
IRLS convergido es una prueba de aproximación y se informa separadamente por
observada, alpha, brazo y seed. Los casos no convergidos no se mezclan con los
convergidos ni se convierten en valores finitos.

## Fidelidad de gradiente

Validation contiene `127/127` vistas IID y no permite estratificar por
mecanismo. Se eligen determinísticamente dieciséis vistas estratificadas por
tamaño. A cada vista se asigna una combinación `brazo × seed` de modo que las
ocho combinaciones aparezcan dos veces. Para `K ∈ {16, 64, 256}` y
`alpha ∈ {0, 0.5, 1}`, se compara
el gradiente de MSE de cociente respecto de cada relación válida contra
diferencia finita central de la referencia NumPy fixed-K.

- paso base `h=1e-5 × max(1, |y_e|)`;
- excluir sólo coordenadas cuya distancia al kink Huber sea menor que `10h`,
  dejando motivo y conteo explícitos;
- reportar coseno, error relativo L2 y máximo error absoluto;
- repetir ocho coordenadas con `h/2` para diagnosticar estabilidad del cociente.

La comparación usa `x_true` únicamente para definir la función escalar de
auditoría; nunca entra al surrogate como input.

## Regla para habilitar entrenamiento

No se elige K por rendimiento downstream. Se toma la menor profundidad que, en
validation y sobre todos los estados convergidos, satisfaga simultáneamente:

- Torch↔NumPy fixed-K: máximo error de potenciales, pesos y objetivo
  Huber `≤1e-9`;
- frente al executor convergido: percentil 99 de RMSE de potenciales `≤1e-4`
  y máximo `≤1e-3`;
- gradiente: coseno mediano `≥0.999`, percentil 95 de error relativo `≤1e-2` y
  ninguna inversión de signo en coordenadas estables con magnitud `≥1e-6`.

Estos son umbrales de conformidad numérica, no de mérito científico. Si ninguna
K los cumple, el surrogate queda rechazado para training en su forma fija. No
se relajan umbrales después de observar resultados.

## Recursos, artefactos y replay

La corrida es CPU-only, `CUDA_VISIBLE_DEVICES=''`, un thread numérico, techo de
`20 min` y `4 GiB`. Registra RSS al inicio, cada 64 estados de valor, después
de cada vista de gradiente y al cierre; la serie temporal forma parte del
paquete.

Se preservan config resuelta, hashes de fuentes/checkpoints, estados y errores
por unidad, probes y coordenadas de gradiente, exclusiones por kink, resumen,
decisión mecánica de conformidad, manifest, runtime separado y replay
byte-exacto. Las consultas web, el entrenamiento, nuevos seeds, transferencia y
toda GPU quedan fuera de alcance.
