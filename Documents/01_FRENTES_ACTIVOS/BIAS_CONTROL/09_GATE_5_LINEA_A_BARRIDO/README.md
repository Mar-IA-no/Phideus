# Gate 5 Linea A — Barrido Comprehensivo + Cross-Modal Injection

**Estado**: PENDING (post Gate 4.4)
**Fecha**: 2026-02-15
**Origen**: Fusion de ex-Gate 4.4 (barrido bifurcado) + ex-Gate 4.5 (cross-modal injection) + FiLM (GPT 5.2 Pro §11)

---

## Contenido

Linea A de Gate 5 combina tres componentes:

### 1. Barrido de descriptores x mecanismos

Matriz factorial amplia, adaptada a los learnings de Gate 4.3 y Gate 4.4:

**Mecanismos de inyeccion** (4 familias):
- Concat (Gate 4.3 ganador)
- Cross-attention (regular y reverse)
- **FiLM** (Feature-wise Linear Modulation) — nueva familia, propuesta GPT 5.2 Pro §11

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

### 3. FiLM Transversal

Feature-wise Linear Modulation: el descriptor genera gamma y beta que modulan
las features del encoder por canal. Alternativa a concat y cross-attention.

---

## Dependencias

- **Gate 4.3**: proporciona ganadores de descriptores y mecanismos
- **Gate 4.4**: resultados de arquitecturas mayores informan que variantes probar

## Protocolo

- Fresh desde `foundation_locked_e25.pt`
- `pool=256`, `queries=500`, `seed=42`
- Metrica primaria: `S=min(A2M, M2A)` + `hard_neg`

---

## Documentos de referencia

- Plan original barrido: `plan_gate_4.4.md` (migrado desde ex-08_GATE_4_4)
- Cross-modal injection: `ROADMAP_INSUMOS_GPT5.2PRO.md`
- FiLM proposal: `ROADMAP_INSUMOS_GPT5.2PRO.md` §11 (Exp P2)
