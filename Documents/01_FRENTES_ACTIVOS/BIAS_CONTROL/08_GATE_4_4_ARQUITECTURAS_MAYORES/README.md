# Gate 4.4 — Arquitecturas Mayores: Third Tower + FiLM + MoE

**Estado**: CORTE PARCIAL DE SCREENING (UNC)
**Fecha**: 2026-02-17
**Origen**: Renumeracion de roadmap — absorbe ex-Gate 4.5 (third tower), integra FiLM y agrega MoE con Ratio Expert (GPT 5.2 Pro §11)

---

## Estado operativo

Screening Gate 4.4 en UNC (8 brazos):

- `t3-tri`, `t3-anc`, `t3-wt`
- `film-a4`, `film-d4`, `film-dual`
- `moe-a4`, `moe-dual`

Protocolo activo de screening (comparabilidad con Gate 4.3):

- checkpoint: `foundation_locked_e25.pt`
- `--freeze-policy run-d` (explícito)
- 5 epochs por brazo
- evaluación estructurada en epochs 3 y 5

Corte parcial validado desde `results_unc/`:

| Brazo | Familia | S@e3 | S@e5 | Notas de corte |
|-------|---------|------|------|----------------|
| `t3-wt` | Third Tower | 47.6% | 67.6% | recuperación tardía fuerte en e5 |
| `t3-tri` | Third Tower | 47.4% | 65.0% | convergencia estable |
| `t3-anc` | Third Tower | 40.2% | 42.2% | mejora marginal entre e3-e5 |
| `moe-a4` | MoE | 58.8% | 58.2% | mejor valor en e3 |
| `film-a4` | FiLM | 59.2% | pendiente | solo eval estructurada e3 |
| `film-d4` | FiLM | 58.8% | pendiente | solo eval estructurada e3 |
| `film-dual` | FiLM | pendiente | pendiente | sin eval estructurada consolidada |
| `moe-dual` | MoE | pendiente | pendiente | sin eval estructurada consolidada |

Referencia canónica de comparación corta: `d4a4=69.8%` (Gate 4.3, 5ep, foundation + `run-d`).

Objetivo inmediato: cerrar `film-a4`, `film-d4`, `film-dual`, `moe-dual` en el mismo protocolo y consolidar tabla completa `S/A2M/M2A/hard_neg`.

---

## Contenido

Gate 4.4 agrupa tres propuestas de **cambio arquitectonico mayor** del modelo:

### 1. Third Tower / Ratio Bridge

Tratar los ratios como una **modalidad propia** con su propio encoder independiente.
Tres torres convergen en el espacio latente. Los ratios dejan de ser señal auxiliar
y se convierten en ciudadanos de primera clase.

```
Audio Tower          Ratio Tower          MIDI Tower
Waveform [B,96000]   Ratio desc [B,?,K]   MIDI Events [B,N]
    |                     |                    |
CNN (4 stages)       Transformer (2-4L)   Event Embedding
    |                d=256                     |
Transformer (4L)          |              Transformer (4L)
d=1024                    |              d=512
    |                     |                    |
Pool + Proj          Pool + Proj         Pool + Proj
    |                     |                    |
audio_emb [B,256]   ratio_emb [B,256]   midi_emb [B,256]
    \                     |                    /
     ========= Convergencia en latente ========
```

Input recomendado para la torre de ratios: **ambos descriptores combinados** (audio A_best + MIDI D4),
con source embeddings para distinguir procedencia. Es la opcion mas alineada con Phideus:
la torre de ratios ES el puente explicito.

Variantes de loss:
- **Triangular**: VICReg(audio,midi) + VICReg(audio,ratio) + VICReg(midi,ratio)
- **Ratio como ancla** (la mas audaz): solo VICReg(audio,ratio) + VICReg(midi,ratio) — sin loss directo audio↔midi
- **Combinado ponderado**: VICReg(audio,midi) + alpha * (VICReg(audio,ratio) + VICReg(midi,ratio))

### 2. FiLM estructural (audio / midi / dual)

Feature-wise Linear Modulation aplicada dentro de los encoders (no como post-procesado).
El descriptor global genera `(gamma, beta)` por capa para modular activaciones internas.

Objetivo:
- condicionar dinámicamente la representación sin expandir fuertemente los parámetros;
- evaluar si la señal ratio funciona mejor como modulación que como concatenación fija.

### 3. MoE con Ratio Expert

Mixture of Experts donde uno de los expertos se especializa en ratio information.
Cambia la arquitectura interna del encoder (no solo la inyeccion).

Propuesta de GPT 5.2 Pro (§11): router learned que asigna frames a expertos,
con uno dedicado a ratio processing. Permite al modelo decidir dinámicamente
cuándo y cómo usar ratio info a nivel de frame.

---

## Dependencias

- **Gate 4.3** (cerrado): proporciona mejores descriptores + mecanismos de inyeccion
- Usa ganadores de Gate 4.3 para diseñar torre, FiLM y input del MoE

## Criterios de referencia (protocolo)

| Criterio | Umbral | Significado |
|----------|--------|-------------|
| S(third-tower) > S(best Gate 4.3) | +2pp minimo | Third tower aporta sobre inyeccion |
| S(ratio-ancla) > 50% | absoluto | Ratios como puente viable |
| S(FiLM) > S(best Gate 4.3) | +2pp minimo | FiLM aporta como modulación estructural |
| S(MoE) > S(best Gate 4.3) | +2pp minimo | MoE routing aporta |
| Convergencia estable | no NaN/colapso en 5ep | Arquitectura viable |

---

## Documentos de referencia

- Plan original third tower: este README (migrado desde ex-09_GATE_4_5)
- FiLM proposal: `ROADMAP_INSUMOS_GPT5.2PRO.md` §11 (Exp P2)
- MoE proposal: `ROADMAP_INSUMOS_GPT5.2PRO.md` §11 (Exp P3)
- Resultados Gate 4.3: `07_GATE_4_3_RATIO_RE_CENTRICO/README.md`
