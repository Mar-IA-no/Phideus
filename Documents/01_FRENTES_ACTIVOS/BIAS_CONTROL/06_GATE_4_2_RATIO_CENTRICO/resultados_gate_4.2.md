# Resultados Gate 4.2

## Cierre formal D4 extendido (8 epocas)

Fuente operativa: corrida `D4 8ep` sobre foundation lock (`run-d`, evaluacion canonica por epoch).

### Tabla por epoch

| Epoch | S | A2M R@10 | M2A R@10 | hard_neg | Nota |
|---|---:|---:|---:|---:|---|
| e1 | 57.2% | 57.2% | 59.6% | 90.8% | warmup |
| e2 | 60.6% | 60.6% | 63.0% | 90.4% | LR aun alto |
| e3 | 61.8% | 61.8% | 63.6% | 90.0% | iguala foundation |
| e4 | 58.6% | 58.6% | 60.2% | 91.2% | dip de cosine midpoint |
| e5 | 62.8% | 62.8% | 64.4% | 90.6% | supera foundation |
| e6 | 62.2% | 62.2% | 63.8% | 91.6% | estabilizacion |
| e7 | **64.2%** | **65.0%** | **64.2%** | **91.6%** | **best** |
| e8 | 63.4% | 63.4% | 64.2% | 91.6% | plateau alto |

### Comparacion de referencia

| Referencia | Best S | hard_neg | Epoch |
|---|---:|---:|---:|
| Foundation (`D-02 e25`) | 61.8% | 90.4% | 25 |
| D4 3ep | 64.2% | 91.4% | 3 |
| D4 8ep | 64.2% | 91.6% | 7 |

### Lectura operativa

1. `D4` sostiene mejora sobre foundation y no muestra colapso tardio.
2. El techo de `S` se confirma en `64.2%` (mismo valor en 3ep y 8ep).
3. El tramo largo aporta consolidacion en `hard_neg` y confirma estabilidad.
