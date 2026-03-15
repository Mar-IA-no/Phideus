# Escalón 1 — Índice Completo

**Dominio**: Audio ↔ MIDI (MAESTRO v3.0.0)
**Estado**: ✅ CERRADO como cierre principal del escalón; Gate 6 AMT sigue activo como validación downstream paralela, Gate 9 / `A10` ya dejaron datos retrospectivos y Gate 10 quedó abierto como barrido causal en curso parcial en UNC
**Última actualización**: 2026-03-15

> Este documento es el punto de entrada único para todo el trabajo del Escalón 1.
> El Escalón 1 se distribuye en **dos directorios físicos** por razones históricas:
> - `Documents/01_FRENTES_ACTIVOS/ESCALON_1/` — brazo Shazam (subfases 1-A y proto-1-B)
> - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/` — brazo neural (subfases 1-B DANN + 1-C VICReg/descriptores)
>
> Ambos directorios conforman un único Escalón 1 dentro de la Triplescaloneta.
> **Decisión documental vigente (2026-03-05):** mantener esta separación sin mover ni renombrar directorios; este índice cumple el rol de unión.

---

## Contexto: Triplescaloneta

| Escalón | Dominio | Dataset | Estado |
|---------|---------|---------|--------|
| **1** | Audio ↔ MIDI | MAESTRO v3 (~200h, 1276 piezas) | ✅ CERRADO |
| **2** | Speech ↔ EGG | French Lombard (38 speakers, 9120 clips) | Activo |
| **3** | Audio XY ↔ Lissajous | Generado (determinista) | Conceptual |
| **4** | ECG ↔ PPG | BIDMC / MIMIC-III | Futuro |

Plan maestro: `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md`
Definición de escalones: `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`

---

## Subfases del Escalón 1

### Escalón 1-A — Brazo Shazam: ratio tokens sin aprendizaje

**Directorio**: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/` (este directorio)
**Estado**: ✅ CERRADO — límite estructural confirmado
**Período**: 2026-02-04

**Objetivo**: Demostrar cross-modal Audio↔MIDI mediante matching directo de constelaciones de ratios (estilo Shazam), sin entrenar ningún parámetro.

**Resultado final**:
- Token compatibility: cosine=0.96 ✅
- Cross-modal retrieval (Route A, N=20): **26.6%** (5.4× random)
- Causa raíz del límite: resolución temporal del onset detector (~50-100ms) incompatible con timing exacto MIDI
- Cierre formal: `CIERRE_ESCALON1_SHAZAM.md`

**Documentos clave**:

| Documento | Descripción |
|-----------|-------------|
| `CIERRE_ESCALON1_SHAZAM.md` | **★ Cierre formal** — análisis completo del brazo |
| `RESULTADOS_ESCALON_1.md` | Cronología completa (fases 1-11) |
| `03_INFORMES_EXPERIMENTOS/INFORME_FASES_A_B.md` | Auditoría (bug 80%→32.9%) + replicación N=20 |
| `03_INFORMES_EXPERIMENTOS/INFORME_ANALISIS_ERRORES.md` | Análisis causa raíz + ablation por tipo de token |
| `03_INFORMES_EXPERIMENTOS/RESULTADOS_NUEVOS_ENFOQUES.md` | Route A/B — piloto original (con caveat bug) |

**Código**:
- `src/extractors/event_based_extractor.py` — Route A
- `src/extractors/improved_tf_extractor.py` — Route B
- `experiments/un_audio_un_midi/` — todos los scripts de prueba

---

### Escalón 1-B — DANN: domain adversarial training

**Directorio**: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/02_GATE_3_DANN/`
**Estado**: ✅ CERRADO — resultado negativo informativo
**Período**: 2026-02-07

**Objetivo**: Investigar si forzar invarianza de dominio (DANN) mejora la representación cross-modal.

**Resultado**: 4 runs de DANN completados. DANN destruye información útil; ningún run mejora sobre Gate 2 baseline. Resultado negativo pero informativo: la confusión audio/MIDI no es el problema.

**Documentos clave**:
- `../BIAS_CONTROL/02_GATE_3_DANN/` — resultados de las 4 runs
- `../BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md`

---

### Escalón 1-C — BIAS_CONTROL: representaciones densas + descriptores

**Directorio**: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/`
**Estado**: ✅ CERRADO (Gate 5B); Gate 6 AMT sigue abierto como validación downstream paralela y la rama retrospectiva `Gate 9 / A10 / Gate 10` queda activa como limpieza causal de segundo orden dentro del escalón
**Período**: 2026-02-12 — 2026-03-05+

**Objetivo**: Aprendizaje cross-modal con encoders densos (MERT audio + Transformer MIDI + VICReg) y descriptores relacionales (A4, D4) como señal auxiliar. Análisis causal de qué parte de la estructura de ratios aporta la ventaja.

**Resultado principal (Gate 5B)**:
- d4a4: S=**84.1% ±2.3pp** (multi-seed, RECORD)
- a4r: S=80.7% ±1.9pp
- D0: S=75.2% ±2.3pp
- Hallazgo central: los descriptores reorganizan geometría de embeddings (+82% CKA) pero no enriquecen la decodificabilidad de features individuales — ventaja geométrica, no de feature richness.

**Documentos clave**:

| Documento | Descripción |
|-----------|-------------|
| `../BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Roadmap completo v2.2 |
| `../BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md` | **★ Cierre científico Escalón 1-C** |
| `../BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md` | Ranking unificado de todos los descriptores |
| `../BIAS_CONTROL/12_GATE_6_AMT/` | Gate 6 AMT (validación downstream: `Exp B` negativo útil, `Exp A` screening mínimo) |
| `../BIAS_CONTROL/16_GATE_9_NAT_HARM_DESCRIPTOR/PLAN_GATE9.md` | Gate 9 / `A10` como reapertura retrospectiva ya con datos |
| `../BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md` | Gate 10 como barrido causal descriptor × mecanismo ya en curso parcial |

---

## Resumen de Cierre del Escalón 1

| Subfase | Pregunta | Resultado | Veredicto |
|---------|----------|-----------|-----------|
| 1-A Shazam | ¿Funciona cross-modal sin aprendizaje? | 26.6% (5.4× random), límite estructural | CERRADO — NO (el mecanismo directo no alcanza) |
| 1-B DANN | ¿Ayuda forzar invarianza de dominio? | DANN destruye info útil | CERRADO — NO |
| 1-C Neural | ¿Funciona con representaciones densas + descriptores? | S=84.1%, causalidad confirmada, más ramas downstream/retrospectivas ya abiertas | CERRADO — SÍ (ventaja geométrica) |

**H3a (Escalón 1: Audio↔MIDI)**: Parcialmente validada. Los descriptores relacionales (A4, D4) capturan estructura cross-modal de manera causal. La ventaja es geométrica: reorganizan la geometría de embeddings sin enriquecer la decodificabilidad de features individuales. El matching directo sin aprendizaje (1-A) no es suficiente; se requiere optimización.

---

## Sobre la Separación en Dos Directorios

El brazo Shazam (1-A/proto-1-B) y el brazo neural (1-B DANN + 1-C BIAS_CONTROL) quedaron en directorios separados por razones históricas: el Shazam se empezó antes de que se definiera BIAS_CONTROL como su sucesor. Conceptualmente son el mismo Escalón 1 de la Triplescaloneta.

**Decisión tomada (2026-03-05): Opción A — solo documentación, sin mover archivos.**

Esto implica:
- este `INDICE_ESCALON1_COMPLETO.md` es el punto de entrada unificado;
- `ROADMAP_BIAS_CONTROL.md` sigue siendo la referencia correcta para Escalón 1-C;
- `ESCALON_1/` queda como brazo Shazam cerrado;
- `BIAS_CONTROL/` concentra la evidencia científica principal del escalón;
- no se renombra ni se mueve ninguno de los dos directorios.

La separación física tiene sentido conceptual: el Shazam fue el intento "sin aprendizaje" y BIAS_CONTROL fue el intento "con aprendizaje". Que vivan en directorios distintos no es una inconsistencia a corregir en el árbol, sino una distinción histórica que ahora queda resuelta por índice y narrativa.
