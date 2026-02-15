# Gate 4.4 — Arquitecturas Mayores: Third Tower + MoE

**Estado**: PENDING (post Gate 4.3)
**Fecha**: 2026-02-15
**Origen**: Renumeracion de roadmap — absorbe ex-Gate 4.5 (third tower) + MoE con Ratio Expert (GPT 5.2 Pro §11)

---

## Contenido

Gate 4.4 agrupa las dos propuestas de **cambio arquitectonico mayor** del modelo:

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

### 2. MoE con Ratio Expert

Mixture of Experts donde uno de los expertos se especializa en ratio information.
Cambia la arquitectura interna del encoder (no solo la inyeccion).

Propuesta de GPT 5.2 Pro (§11): router learned que asigna frames a expertos,
con uno dedicado a ratio processing. Permite al modelo decidir dinámicamente
cuándo y cómo usar ratio info a nivel de frame.

---

## Dependencias

- **Gate 4.3** (cerrando): proporciona mejores descriptores + mecanismos de inyeccion
- Usa ganadores de Gate 4.3 para diseñar torre optima e input del MoE

## Criterios GO/NO-GO

| Criterio | Umbral | Significado |
|----------|--------|-------------|
| S(third-tower) > S(best Gate 4.3) | +2pp minimo | Third tower aporta sobre inyeccion |
| S(ratio-ancla) > 50% | absoluto | Ratios como puente viable |
| S(MoE) > S(best Gate 4.3) | +2pp minimo | MoE routing aporta |
| Convergencia estable | no NaN/colapso en 5ep | Arquitectura viable |

---

## Documentos de referencia

- Plan original third tower: este README (migrado desde ex-09_GATE_4_5)
- MoE proposal: `ROADMAP_INSUMOS_GPT5.2PRO.md` §11 (Exp P3)
- Resultados Gate 4.3: `07_GATE_4_3_RATIO_RE_CENTRICO/README.md`
