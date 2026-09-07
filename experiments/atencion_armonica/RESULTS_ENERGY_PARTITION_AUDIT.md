# Diagnóstico histórico de partición por amplitudes

Fecha: 2026-09-07. Estado: ejecución CPU, replay y auditoría independiente de resultados completos, sin findings abiertos. Diagnóstico auxiliar, no cierre del contraste neuronal.

## Resultado y alcance

El solver que recibe solamente amplitudes permutadas recuperó una partición única y correcta en las ocho mezclas seleccionadas de dos fuentes y las ocho de tres fuentes. El resultado se conserva al reconstruir amplitudes desde el canal `float32(log_amp)` que recibe la red. Los ocho casos de una fuente también fueron únicos y correctos, pero se informan aparte porque no exigen separar fuentes.

La muestra consta de los primeros cuatro IDs de cada celda histórica `poly{1,2,3} × {easy,hard}`: 24 registros ya abiertos, no aleatorios. No estima rendimiento poblacional ni demuestra que las redes históricas utilizaran este canal. Sí muestra que esos casos pueden resolverse sin frecuencias bajo el prior público de energía unidad y entre cuatro y ocho parciales por fuente. Por ello, resolver este banco no basta para atribuir una ventaja a geometría armónica.

| Vista | Precisión | Una fuente | Dos fuentes | Tres fuentes |
|---|---|---:|---:|---:|
| Original | Float64 del pool | 8/8 únicas correctas | 8/8 únicas correctas | 8/8 únicas correctas |
| Original | Canal log-amp float32 | 8/8 únicas correctas | 8/8 únicas correctas | 8/8 únicas correctas |
| Ganancias por fuente | Ambas, por separado | 8/8 prior incompatible | 8/8 prior incompatible | 8/8 prior incompatible |

Las dos permutaciones dieron estados y particiones restauradas compatibles en cada uno de los 96 registros de evaluación. No hubo estados de límite, múltiples particiones ni soluciones únicas incorrectas. La intervención de ganancias `0.70/1.10/1.40` rompe deliberadamente el prior: los 24 rechazos por precisión ocurren en el filtro escalar de energía total, antes de buscar particiones. No es un control igualado ni prueba de que el banco intervenido quede libre de otros atajos.

## Reproducción y artefactos

Protocolo congelado: [plan, §4](PLAN_GEOMETRIC_RESEARCH_ACTION.md). Implementación previa al resultado: commit `4e72314`, [harness](run_energy_partition_audit.py), [núcleo](../../src/atencion_armonica/energy_partition.py). Quince pruebas CPU sobre fixtures y auditoría independiente del arnés precedieron la ejecución; no se modificaron tolerancias ni muestra a partir del resultado.

```sh
timeout 180s env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  venv/bin/python experiments/atencion_armonica/run_energy_partition_audit.py --campaign
```

Las rutas de salida deben ser nuevas; la campaña rechaza sobrescritura. Los artefactos locales preservados son `data/atencion_armonica/geometry_energy_audit_v1/` y `data/atencion_armonica/geometry_energy_audit_v1_replay/`. Cada directorio contiene los 24 registros originales, amplitudes entregadas, permutaciones, candidatos, particiones, evaluación por mezcla, resumen, hashes de código/datos y runtime. Los cuatro archivos científicos coinciden byte a byte entre primaria y replay; el recibo está en `geometry_energy_audit_v1/replay_comparison.json`.

El recibo registra 1.731914288 s totales; primaria y replay registran 0.772905541 s y 0.776018146 s, RSS máxima de 40,202,240 y 40,022,016 bytes, y 235,524 bytes científicos cada una. Ambas terminaron sin importar PyTorch. No se utilizó ni consultó GPU.

## Consecuencia experimental

El siguiente contraste debe impedir o controlar explícitamente la recuperación por energía unidad; cambiar sólo la arquitectura o su loss conservaría una explicación alternativa suficiente. Tampoco alcanza con multiplicar cada fuente por una ganancia y declarar resuelto el problema: habrá que definir las observaciones y comprobar los controles pertinentes. El aporte de una restricción física compartida entre parciales debe distinguirse del de una consistencia genérica de particiones y del de descriptores explícitos.

Continúan pendientes la selección final de hipótesis, el protocolo neuronal auditado y su ejecución. No se promueve ninguna arquitectura ni se declara GO/NO-GO.
