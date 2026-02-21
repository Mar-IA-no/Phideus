<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_1--C-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Gate_4.4_Cerrado_%2B_Batch_60ep-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-02-21  
> **Estado**: Gate 4.4 permanece cerrado en screening (24 brazos) y el bloque largo 30ep permanece cerrado (`d4a4=83.6`, `a4r=82.0`, `d4-a4r=79.8`, `t3-wt=79.8`, `d4a4r=74.4`, `moe-dual=72.6`).  
> **Decisión operativa vigente**: sostener extensión temporal con runs 50ep/60ep y abrir comparación de scheduler con batch `cosine-tail` 60ep (`--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02`).  
> **Infraestructura**: estrategia distribuida LOCAL+UNC activa; foundation lock publicado (`v0.1.0-foundation`).

## Navegación rápida

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Estado por Gate](#estado-por-gate)
- [Hallazgos Causales del Corte](#hallazgos-causales-del-corte)
- [Plan Operativo Vigente](#plan-operativo-vigente)
- [Frentes y Documentos](#frentes-y-documentos)

---

## Resumen Ejecutivo

Gate 4.3 dejó una base fuerte (`d4a4=69.8%` a 5ep; `d4a4=83.6%` a 30ep), y Gate 4.4 completó el filtro arquitectural con evidencia comparable en toda la grilla corta. El cierre largo de 30ep mostró trayectorias heterogéneas: `t3-wt` recuperó tarde hasta `79.8%` y `moe-dual` cerró en `72.6%`.

En paralelo, apareció un hallazgo de dinámica de entrenamiento: el scheduler cosine en 60ep retrasa la fase de explotación respecto de 30ep. Por eso el frente abrió dos líneas simultáneas: (1) 60ep/50ep para validar dinámica temporal y (2) `cosine-tail` para replicar fase útil de 30ep y sostener gradiente en cola.

### Baseline oficial de comparación (histórico)

`Gate 2 - checkpoint_epoch45`

| Métrica | Valor |
|--------|-------|
| A2M R@10 (pool 256/500/seed42) | 34.4% |
| M2A R@10 (pool 256/500/seed42) | 37.6% |
| Hard negative accuracy | 80.4% |
| S=min(A2M,M2A) | 34.4% |

### Screening @5ep (ranking unificado, top del frente)

| Rank | Brazo | Gate/Familia | Best S | A2M | M2A | hard_neg | vs D0 |
|------|-------|--------------|--------|-----|-----|----------|-------|
| 1 | d4a4 | 4.3 Dual concat | 69.8% | 69.8% | 70.6% | 91.6% | +9.6pp |
| 2 | a4r | 4.3-F5 reverse | 68.6% | 68.6% | 69.0% | 91.6% | +8.4pp |
| 3 | t3-wt | 4.4 Third Tower | 67.6% | 71.4% | 67.6% | 91.2% | +7.4pp |
| 4 | t3-tri | 4.4 Third Tower | 65.0% | 65.4% | 65.0% | 90.6% | +4.8pp |
| 10 | D0 | 4.3 baseline | 60.2% | 60.4% | 60.2% | 90.0% | — |
| 11 | moe-a4-v2 | 4.4-MoE | 60.2% | 60.4% | 60.2% | 90.8% | 0.0pp |

Notas de cierre 4.4:
- `film-dual` cerró en `59.4%` (e5), `moe-dual` en `59.2%` (e5).
- `moe-a4-v2/v3/v4` no superan D0 (v2 empata D0).

### Runs largos (30ep, scratch) — todos cerrados

| Descriptor | Best S | Best Ep | A2M | M2A | hard_neg |
|-----------|--------|---------|-----|-----|----------|
| d4a4 | 83.6% | 30 | 83.6% | 84.2% | 95.2% |
| a4r | 82.0% | 29 | 82.6% | 82.0% | 94.4% |
| d4-a4r | 79.8% | 30 | 81.4% | 79.8% | 94.2% |
| t3-wt | 79.8% | 30 | 82.4% | 79.8% | 94.8% |
| d4a4r | 74.4% | 30 | 74.4% | 74.8% | 92.0% |
| moe-dual | 72.6% | 30 | 72.8% | 72.6% | 93.4% |

Multi-seed e30 (5 seeds): `d4a4 = 84.1% +/- 2.3pp`.

### Corridas activas (UNC, corte operativo 2026-02-21)

| Bloque | Corridas | Estado |
|--------|----------|--------|
| Batch 60ep (cosine estándar) | `a4r` | **completado** (`S=79.4%` en e60) |
| Batch 60ep (cosine estándar) | `D0`, `d4a4` | **en curso** (`D0 S@e40=72.4%`, `d4a4 S@e40=82.6%`) |
| Batch 60ep (cosine estándar) | `d4-a4r`, `moe-dual` | pendientes en cola |
| Hold scheduler 50ep | `t3-wt` (`--lr-hold-fraction=0.5`) | **en curso** (`S@e40=80.6%`) |
| Batch 60ep (cosine-tail) | `D0`, `d4a4`, `a4r`, `d4-a4r` | enviados / pendientes según cola |

---

## Estado por Gate

| Gate / Etapa | Estado | Resultado |
|--------------|--------|-----------|
| Gate 0 | Completado | GO |
| Gate 1 | Completado | GO (sanity intra-modal) |
| Gate 2 | Completado | GO (baseline canónico) |
| Gate 2.5 | Completado | Diagnóstico de separabilidad |
| Gate 3 (DANN) | Cerrado | NO-GO |
| Gate 4.1 | Cerrado | NO-GO (`R1-rescue` insuficiente) |
| Gate 6 (diagnóstico) | Completado | Causa raíz confirmada |
| Bloque A v1.1 | Cerrado | `D-02 e25` como foundation lock |
| Gate 4.2 ratio-céntrico | Cerrado | `D4 8ep` (`S=64.2%`) |
| Gate 4.3 ratio re-céntrico | Cerrado | 13 brazos + scratch; record `S=83.6%` |
| Gate 4.4 arquitecturas mayores | **Cerrado** | Screening 24 brazos + 30ep (`t3-wt`, `moe-dual`) |
| Extensión temporal (post 4.4) | **Abierta** | batch 60ep + `t3-wt` 50ep hold |
| Gate 5A barrido | Pendiente | Barrido descriptor x mecanismo + cross-modal injection |
| Gate 5B showcase científico | Pendiente | 13 tests de validación |

---

## Hallazgos Causales del Corte

1. **Dual same-modality es superaditivo**  
D4 y A4 por separado dan `+3.4pp`; juntos (`d4a4`) dan `+9.6pp`.

2. **Reverse cross-attention supera al cross-attention regular**  
Se observó en audio y MIDI (`A4r>A4x`, `D4r>D4x`).

3. **Cross-modal injection temprana no fue mecanismo ganador**  
`d4a4cm` quedó por debajo del baseline (`-7.8pp` vs D0).

4. **El mejor espacio no apareció por accidente**  
`d4a4-scratch` superó a `D-02` por más de 20pp en el mismo marco de evaluación.

5. **Third Tower weighted (`t3-wt`) mostró convergencia tardía real**  
Pasó de `S=40.0%` (e5 en 30ep scratch) a `S=79.8%` (e30).

6. **MoE mejoró con más tiempo, pero no lideró el bloque largo**  
`moe-dual` llegó a `72.6%` a 30ep: crecimiento sostenido, techo por debajo de d4a4/a4r.

7. **En 5ep, FiLM/MoE quedaron en banda 58-60%**  
La familia 4.4 no desplazó a los ganadores de Gate 4.3 en screening corto.

8. **Scheduler como variable causal de segundo orden**  
En 30ep, el cosine llega rápido a zona de explotación; en 60ep esa transición se retrasa. Quedaron habilitados `--lr-hold-fraction`, `--lr-cosine-ref-epochs`, `--lr-floor`, `--lr-tail-end` y logging `lr_mult`.

9. **A4r mantiene ventaja costo/calidad**  
En runs largos, `a4r` conserva ventaja de velocidad (~13 min/epoch) con desempeño alto (`S=79.4%` en 60ep), reforzando su perfil de descriptor/mecanismo eficiente.

---

## Plan Operativo Vigente

Secuencia inmediata:

1. Monitorear y consolidar las 6 corridas nuevas (`batch_60ep_*` + `t3-wt_50ep_hold`).
2. Cerrar `D0` y `d4a4` 60ep, y mantener trazabilidad explícita de `d4-a4r` y `moe-dual` en cola.
3. Comparar `e30` y `e40/e60` contra el bloque 30ep para separar efecto de presupuesto temporal vs descriptor.
4. Ejecutar lote `cosine-tail` y contrastarlo contra cosine estándar con mismas métricas (`S`, `A2M`, `M2A`, `hard_neg`).
5. Sincronizar ranking + roadmap + transversales en cada corte verificable.

---

## Frentes y Documentos

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/00_TRONCAL/HANDOFF.md` | Continuidad operativa |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro vigente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del frente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md` | Tabla canónica corta+larga |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` | Estrategia distribuida LOCAL+UNC |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | Evolución histórica de representaciones |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | Catálogo vivo de descriptores |

Nota operativa:
- Foundation lock publicado en GitHub Release: `v0.1.0-foundation` (`foundation_locked_e25.pt`, MD5 `ddb2ebf7075eec4dcec1628341ec4942`).

---

*Documento actualizado al corte operativo 2026-02-21 (runs 50ep/60ep en curso + batch cosine-tail lanzado).* 
