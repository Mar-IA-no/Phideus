# Decisiones Gate 4.2

## DEC-G42-001 (2026-02-14 UTC) - Cierre de Gate 4.2

### Decision

Se cierra Gate 4.2 con `D4` extendido a 8 epocas y se habilita Gate 4.3.

### Evidencia minima

1. `D4 8ep` alcanza `S=64.2%` (epoch 7) y `hard_neg=91.6%`.
2. `D4 8ep` iguala el mejor `S` de `D4 3ep` (`64.2%`) y mejora robustez de negativos duros.
3. No se observan patrones de degradacion irreversible en la cola larga.

### Implicacion operativa

1. Gate 4.2 queda cerrado como fase.
2. Gate 4.3 se ejecuta con 6 brazos fresh (`D0`, `D4`, `A4`, `A7`, `D4+A4`, `D4+A7`).
3. Siguiente accion inmediata: pilotos de 1 epoca/100 batches para `a4`, `a7`, `d4a4`, `d4a7`.
