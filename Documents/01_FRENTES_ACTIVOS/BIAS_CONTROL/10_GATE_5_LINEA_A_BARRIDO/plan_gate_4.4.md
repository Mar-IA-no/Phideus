# Plan Gate 4.4 (Barrido Amplio Post Gate 4.3)

Fecha de corte: 2026-02-14

## Alcance

Gate 4.4 se ejecuta despues del bloque focal de Gate 4.3, reutilizando el mismo protocolo canonico de evaluacion.

## Orden acordado - linea MIDI (temperada)

`D0` control transversal + barrido en este orden:

1. `D3`
2. `D8`
3. `D9`
4. `D10`
5. `D2`
6. `D5`
7. `D6`
8. `D7`

Nota: `D1` queda como descriptor ya evaluado en Gate 4.2, sin prioridad de rerun en el inicio de Gate 4.4.

## Orden acordado - linea Audio (armonia natural)

1. `A1`
2. `A2`
3. `A3`
4. `A5`
5. `A6`

## Regla de comparabilidad

- Correr fresh desde `foundation_locked_e25.pt`.
- Mantener `pool=256`, `queries=500`, `seed=42`.
- Metrica primaria: `S=min(A2M, M2A)` + `hard_neg`.

## Criterio de promocion

- Igual a Gate 4.3 (Carril A performance + Carril B potencial), usando `D0` del mismo bloque como referencia.
