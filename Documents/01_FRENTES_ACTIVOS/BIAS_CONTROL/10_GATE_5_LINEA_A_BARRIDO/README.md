# Gate 5 Linea A — Barrido Comprehensivo + Cross-Modal Injection

**Estado**: PENDIENTE (post Gate 4.5)  
**Fecha de actualizacion**: 2026-02-22  
**Origen**: etapa de barrido descriptor x mecanismo y combinaciones cruzadas, posterior al cierre de Gate 4.5.

---

## Contenido

Linea A de Gate 5 combina dos componentes:

### 1. Barrido de descriptores x mecanismos

Matriz factorial amplia, adaptada a los learnings de Gate 4.3, Gate 4.4 y Gate 4.5:

**Mecanismos de inyeccion**:
- Concat (Gate 4.3 ganador)
- Cross-attention (regular y reverse)
- Third Tower (combinatorios con y sin inyeccion)
- FiLM (si hay gap abierto tras Gate 4.4)

**Descriptores** (los que no se probaron en Gate 4.3):
- Linea MIDI: D3, D8, D9, D10, D2, D5, D6, D7
- Linea Audio: A1, A2, A3, A5, A6
- Control transversal: D0

### 2. Cross-Modal Injection

Inyectar descriptores de un dominio en el encoder del OTRO dominio:

| Brazo | Audio encoder recibe | MIDI encoder recibe | Testea |
|-------|---------------------|---------------------|--------|
| CM-a | — | Audio desc (A_best) | Audio ratios ayudan a MIDI |
| CM-m | MIDI desc (D_best) | — | MIDI intervals ayudan a audio |
| CM-bi | MIDI desc (D_best) | Audio desc (A_best) | Bidireccional |

## Dependencias

- **Gate 4.3**: proporciona ganadores de descriptores y mecanismos
- **Gate 4.4**: resultados de arquitecturas mayores informan que variantes probar
- **Gate 4.5**: define scheduler/ventana de entrenamiento para comparación justa en la siguiente ola

## Protocolo

- Fresh desde `foundation_locked_e25.pt`
- `pool=256`, `queries=500`, `seed=42`
- Metrica primaria: `S=min(A2M, M2A)` + `hard_neg`

---

## Brazos nuevos propuestos (pendientes de implementacion)

Segun `Documents/NOTAS_CLAUDE-CODEX.md`:

1. `t3-wt-vanilla`
   - Third tower con encoders vanilla (sin inyeccion descriptorial en audio/midi).
   - Loss weighted `70/15/15` (AM/AR/MR).
   - Pregunta: cuanto aporta la tower por si sola.
2. `t3-wt-a4r`
   - Third tower combinada con inyeccion `d4-a4r` (audio reverse + midi concat).
   - Loss weighted `70/15/15`.
   - Pregunta: si la tower suma sobre el mejor mecanismo de inyeccion observado.

Estos brazos se registran como **propuesta de plan**, no como resultados.

---

## Documentos de referencia

- Gate 4.5 (scheduler): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md`
- Plan original barrido: `plan_gate_4.4.md` (histórico)
- Cross-modal injection: `ROADMAP_INSUMOS_GPT5.2PRO.md`
