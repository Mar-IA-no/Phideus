# Gate 4.4 — Arquitecturas Mayores: Third Tower + FiLM + MoE

**Estado**: CERRADO (screening 5ep + runs largos clave cerrados)  
**Fecha de corte**: 2026-02-19  
**Origen**: renumeración de roadmap (absorbe ex-Gate 4.5 + integra FiLM y MoE)

---

## Estado operativo

Gate 4.4 se ejecutó con protocolo fijo comparable al cierre de Gate 4.3:

- checkpoint: `foundation_locked_e25.pt`
- `--freeze-policy run-d`
- 5 epochs por brazo
- evaluación estructurada en epochs 3 y 5

El screening cerró con 24 brazos (21 originales + `moe-a4-v2`, `moe-a4-v3`, `moe-a4-v4`) y se completaron corridas largas scratch de `t3-wt` y `moe-dual`.

### Tabla final Gate 4.4 (structured eval 5ep)

| Brazo | Familia | Best S | Best Ep | A2M | M2A | hard_neg | vs D0 |
|-------|---------|--------|---------|-----|-----|----------|-------|
| `t3-wt` | Third Tower | 67.6% | 5 | 71.4% | 67.6% | 91.2% | +7.4pp |
| `t3-tri` | Third Tower | 65.0% | 5 | 65.4% | 65.0% | 90.6% | +4.8pp |
| `moe-a4-v2` | MoE v2 | 60.2% | 5 | 60.4% | 60.2% | 90.8% | 0.0pp |
| `film-dual` | FiLM | 59.4% | 5 | 60.2% | 59.4% | 91.4% | -0.8pp |
| `moe-a4-v4` | MoE v4 | 59.4% | 5 | 60.6% | 59.4% | 91.2% | -0.8pp |
| `film-a4` | FiLM | 59.2% | 3 | 60.8% | 59.2% | 89.8% | -1.0pp |
| `moe-dual` | MoE | 59.2% | 5 | 61.2% | 59.2% | 91.6% | -1.0pp |
| `moe-a4-v3` | MoE v3 | 59.2% | 5 | 60.6% | 59.2% | 91.2% | -1.0pp |
| `film-d4` | FiLM | 58.6% | 5 | 61.0% | 58.6% | 91.8% | -1.6pp |
| `moe-a4` | MoE | 58.2% | 3 | 58.8% | 60.2% | 89.6% | -2.0pp |
| `t3-anc` | Third Tower | 42.2% | 5 | 42.2% | 42.2% | 89.4% | -18.0pp |

Referencia de comparación corta: `D0=60.2%`, `d4a4=69.8%` (Gate 4.3, mismo protocolo 5ep).

---

## Runs largos vinculados (scratch 30ep)

| Descriptor | Best S | Best Ep | A2M | M2A | hard_neg |
|-----------|--------|---------|-----|-----|----------|
| `t3-wt` | 79.8% | 30 | 82.4% | 79.8% | 94.8% |
| `moe-dual` | 72.6% | 30 | 72.8% | 72.6% | 93.4% |

Estas dos corridas cierran la parte de validación larga dentro de Gate 4.4 y alimentan el ranking largo general del frente.

---

## Lectura técnica del cierre

1. Third Tower fue la familia con mejor señal en 5ep dentro de Gate 4.4 (`t3-wt`, `t3-tri`).
2. FiLM quedó agrupado en banda 58-59%.
3. MoE quedó en banda 58-60% en 5ep; las variantes v2/v3/v4 no superaron D0 (v2 empata D0).
4. En largo, `t3-wt` mostró crecimiento tardío fuerte y cerró en `79.8%`; `moe-dual` cerró en `72.6%`.

---

## Próxima conexión de roadmap

Gate 4.4 queda cerrado como bloque arquitectural.  
El foco operativo pasa a:

- batch 60ep comparativo (`D0`, `d4a4`, `a4r`, `d4-a4r`, `moe-dual`)
- `t3-wt` 50ep con hold de LR (`--lr-hold-fraction=0.5`)

Estos runs alimentan la decisión de ventana temporal previa al paso de ejecución de Gate 5A/5B.

---

## Referencias

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `results_unc/gate44/`
- `results_unc/gate44_t3-wt_scratch_30ep/`
- `results_unc/gate44_moe-dual_scratch_30ep/`
