<div align="center">

# Escalón 2
### Speech ↔ EGG Cross-Modal Alignment

![Status](https://img.shields.io/badge/Status-S2--P1_Complete-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Speech↔EGG-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--06-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: Escalón 2 ya tiene `S2-P0` y `S2-P1` completos. Sobre French Lombard `v1.1` (`38` speakers, `9,120` clips, ~`20h`), el baseline lineal ya mostró una señal cross-modal muy fuerte en condición limpia `noise0`: `CCA S=64.4%` contra `7.8%` random, con `CI grouped [57.8%, 70.2%]`.
> **Próximo paso único**: `S2-P2-control` (`D0` neural con dos encoders simétricos trainables), usando exactamente la población congelada en `data/lombard/segment_index.json`.

## Qué es este frente

Escalón 2 es la primera prueba de generalización fuera de música dentro de Triplescaloneta. La pareja ya no es Audio↔MIDI, sino dos sensores físicos distintos del mismo fenómeno vocal:

- `Speech`: micrófono, fuente glotal más filtro del tracto vocal.
- `EGG`: electroglotógrafo, oscilación glotal medida por electrodos.

La hipótesis no es que el descriptor musical `A4` se trasplante intacto. La hipótesis es más disciplinada: si la idea ratio-céntrica toca algo real, debería sobrevivir cuando el acople entre modalidades siga siendo físico pero cambie radicalmente de dominio.

## Estado verificado de `S2-P0`

| Elemento | Estado |
|----------|--------|
| Dataset local | French Lombard `v1.1` |
| Speakers | `38` (`20F/18M`) |
| Clips | `9,120` |
| Duración real | ~`20h` |
| Split speaker | `28/5/5` (`train/validation/test`) |
| Manifest | `data/lombard/manifest.json` |
| Segment index | `data/lombard/segment_index.json` |
| Segmentos totales | `108,536` |
| Piloto limpio `noise0` | `19,910` train, `3,624` val, `3,629` test |
| Alignment audit | `data/lombard/alignment_audit.json` |
| P1 results | `data/lombard/p1_results/p1_results_noise0.json` |
| Lag correction | `0` samples |
| Voiced threshold | `0.1494` |
| Clipping auditado | `0` |

## Protocolo canónico congelado

Estos parámetros ya no están en discusión mientras corran `S2-P1` y `S2-P2`:

| Parámetro | Valor |
|-----------|-------|
| Sample rate | `16 kHz` |
| Ventana | `2.0 s` |
| Hop | `0.5 s` |
| Positivo | misma ventana temporal del mismo clip |
| Split | por speaker |
| Piloto inicial | condición limpia `noise0` |
| Pool size | `128` |
| Métrica primaria | `S = min(Speech2EGG@10, EGG2Speech@10)` |
| CI | grouped bootstrap por speaker (o mínimo por clip) |

El `segment_index.json` es parte del protocolo. `P1` y `P2` no deben regenerar ventanas ni redefinir población.

## Resultado de `S2-P1`

El baseline lineal ya dejó una respuesta fuerte a la pregunta inicial del escalón: Speech↔EGG sí tiene una señal cross-modal muy clara incluso con features simples y métodos lineales.

| Método | Speech2EGG@10 | EGG2Speech@10 | S | CI grouped |
|--------|---------------|---------------|---|------------|
| Raw cosine | `50.4%` | `46.8%` | `46.8%` | `[38.0%, 54.5%]` |
| **CCA** | **`68.4%`** | **`64.4%`** | **`64.4%`** | **`[57.8%, 70.2%]`** |
| Ridge R² | `0.851` Speech→EGG | `0.694` EGG→Speech | — | — |

Lectura prudente:
- observación: la señal lineal supera el azar (`7.8%`) por un margen muy amplio;
- hipótesis compatible: Speech y EGG comparten estructura suficiente como para que un objetivo contrastivo no arranque desde cero;
- inferencia válida hoy: Escalón 2 ya superó el filtro de “posibilidad básica” y quedó habilitado para un `D0` neural serio.

## Artefactos disponibles

| Artefacto | Ruta | Rol |
|-----------|------|-----|
| Manifest clip-level | `data/lombard/manifest.json` | población base y split |
| Segment index window-level | `data/lombard/segment_index.json` | población canónica para evaluación |
| Alignment audit | `data/lombard/alignment_audit.json` | sincronía, clipping, voiced threshold |
| Script P0 | `experiments/bias_control/escalon2/s2_p0_manifest.py` | ingestión, split, segmentación y audit |
| Script P1 | `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` | baseline lineal sobre protocolo congelado |
| Resultados P1 | `data/lombard/p1_results/p1_results_noise0.json` | métricas lineales, CIs grouped y correlaciones CCA |

## Lectura actual

Observación:
- Escalón 2 ya tiene datos, protocolo y baseline lineal muy por encima del azar, pero todavía no tiene baseline neural ni descriptor vocal canonizado.

Hipótesis:
- Speech↔EGG debería dejar una señal lineal usable antes de pedirle a un descriptor nuevo que explique nada.

Inferencia válida hoy:
- el siguiente experimento correcto es `S2-P2-control`, no otra ronda de discusión sobre si Speech↔EGG “tiene o no tiene” señal compartida.

## Próximos pasos

1. Correr `S2-P2-control` (`D0` neural) sobre `noise0`.
2. Usar `manifest.json` y `segment_index.json` tal como quedaron congelados.
3. Mantener pool canónico y CI grouped idénticos a `S2-P1`.
4. Recién después abrir la competencia de descriptores vocales (`V4`, `A4-16k`, `V4+A4`).

## Relación con el resto del programa

- Escalón 1 queda cerrado en su argumento principal; lo que sigue activo ahí ya no bloquea esta apertura.
- Gate 6 AMT sigue como validación downstream musical.
- Gate 5A continúa como línea oportunista.
- Gate 7.1a ya devolvió un resultado útil: agrandar el encoder de audio en modo frozen no mejoró el retrieval. Escalón 2 no necesita esperar más a ese frente.
