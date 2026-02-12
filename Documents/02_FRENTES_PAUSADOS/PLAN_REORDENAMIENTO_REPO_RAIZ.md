# Plan De Reordenamiento Del Repo (Raiz)

## 1) Objetivo

Dejar el repositorio con una estructura clara, navegable por roadmap y sostenible para desarrollo diario, separando de forma estricta:

- codigo fuente,
- scripts de experimento,
- documentacion,
- datos/checkpoints/artefactos operativos.

Este plan es de **ordenamiento estructural**. No cambia logica de modelos ni resultados experimentales.

---

## 2) Diagnostico Actual (Resumen)

### Hallazgos principales

1. Hay mezcla de codigo y artefactos pesados en zonas de codigo:
   - `src/RNA/vae_checkpoints` (~1.1G) dentro de `src/`.
2. Hay duplicacion semantica en raiz:
   - `test/` (assets de prueba) vs `tests/` (pytest de codigo).
   - `train/`, `models/`, `artifacts/`, `visualizations/` superpuestos con `data/`.
3. `experiments/un_audio_un_midi` contiene medios masivos (~3.0G) adentro de carpeta de scripts.
4. `Documents/` ya esta en reestructuracion activa (muchos renames), por lo que conviene estabilizar por fases y no mezclar demasiados cambios simultaneos.

### Estado de versionado (alto nivel)

- Lo mayormente trackeado: `src/`, `experiments/`, `Documents/`, `tools/`.
- Lo mayormente no-trackeado (bien): `data/`, `models/`, `train/`, `test/`, `visualizations/`, `artifacts/`.

---

## 3) Principios De Ordenamiento

1. `src/` solo codigo.
2. `experiments/` solo scripts y configuraciones de ejecucion.
3. Todo dato, checkpoint, visualizacion y salida operacional en `data/`.
4. `tests/` unico lugar para tests de codigo.
5. `Documents/` como capa narrativa y de decision (roadmaps, informes, estado).
6. Evitar carpetas redundantes en raiz si su contenido vive naturalmente bajo `data/`.

---

## 4) Arbol Objetivo (Raiz)

```text
Phideus/
  README.md
  config/
  src/
  experiments/
    bias_control/
    escalon_1/
    archived/
  tests/
  tools/
  data/
    datasets/
    fixtures/
    runs/
      bias_control/
      escalon_1/
    checkpoints/
    evaluations/
    visualizations/
    artifacts/
  Documents/
    00_TRONCAL/
    01_FRENTES_ACTIVOS/
    02_FRENTES_PAUSADOS/
    03_FRENTES_CERRADOS/
    04_TRANSVERSAL/
    90_ARCHIVO_GLOBAL/
```

---

## 5) Plan De Ejecucion Por Fases

## Fase 0 - Congelamiento y Base Segura

Objetivo: evitar mezclar una migracion con otra.

1. Cerrar el estado pendiente de `Documents/` (renames/moves actuales).
2. Tomar snapshot (commit de estabilizacion).
3. Confirmar que no hay procesos de entrenamiento dependiendo de paths legacy a mover.

## Fase 1 - Higiene Critica Del Codigo

Objetivo: sacar artefactos de carpetas de codigo.

1. Mover `src/RNA/vae_checkpoints` -> `data/checkpoints/rna/vae/`.
2. Mover `src/RNA/vae_validation` -> `data/visualizations/rna/vae_validation/`.
3. Verificar imports/rutas para que ningun script espere checkpoints dentro de `src/`.

## Fase 2 - Consolidacion De Datos Operativos

Objetivo: unificar datos y outputs bajo `data/`.

1. `train/` -> `data/datasets/legacy_train/`.
2. `models/` -> `data/checkpoints/legacy_models/`.
3. `visualizations/` -> `data/visualizations/global/`.
4. `artifacts/` -> `data/artifacts/`.
5. `test/` (assets) -> `data/fixtures/test_assets/`.
6. Mantener `tests/` para pruebas de codigo.

## Fase 3 - Experimentos Limpios y Roadmap-Friendly

Objetivo: `experiments/` enfocado solo a ejecucion.

1. Mantener scripts python y configs de experimento en `experiments/*`.
2. Mover medios pesados de `experiments/un_audio_un_midi/*` hacia:
   - `data/datasets/UOEMD/` o
   - `data/runs/escalon_1/un_audio_un_midi/`.
3. Renombrar subcarpetas de `experiments/` para reflejar frentes activos y `archived/`.

## Fase 4 - Contratos De Carpeta y Guardrails

Objetivo: prevenir recaidas.

1. Crear `README.md` corto en cada raiz (`src`, `experiments`, `data`, `tests`, `tools`, `Documents`) con:
   - que entra,
   - que no entra,
   - ejemplos.
2. Ajustar `.gitignore` a estructura final.
3. Agregar chequeo automatico (script simple) que falle si detecta:
   - binarios pesados dentro de `src/`,
   - datasets medios dentro de `experiments/`.

---

## 6) Mapa Keep / Move / Archive

| Zona actual | Accion | Destino propuesto |
|---|---|---|
| `src/RNA/vae_checkpoints` | Move | `data/checkpoints/rna/vae/` |
| `src/RNA/vae_validation` | Move | `data/visualizations/rna/vae_validation/` |
| `test/` | Move | `data/fixtures/test_assets/` |
| `tests/` | Keep | `tests/` |
| `train/` | Move | `data/datasets/legacy_train/` |
| `models/` | Move | `data/checkpoints/legacy_models/` |
| `visualizations/` | Move | `data/visualizations/global/` |
| `artifacts/` | Move | `data/artifacts/` |
| `experiments/un_audio_un_midi` medios | Move | `data/datasets/UOEMD/` o `data/runs/escalon_1/` |
| `config/`, `tools/`, `src/`, `tests/` | Keep | sin cambios de rol |

---

## 7) Riesgos y Mitigaciones

1. **Riesgo**: scripts rotos por cambio de path.  
   **Mitigacion**: fase por fase + smoke test de scripts clave al terminar cada fase.

2. **Riesgo**: mezclar migracion de docs con migracion de raiz.  
   **Mitigacion**: cerrar primero `Documents/` y recien despues mover raiz.

3. **Riesgo**: confusion temporal por coexistencia de paths viejo/nuevo.  
   **Mitigacion**: usar migracion corta por lote y eliminar path viejo al cerrar cada lote.

---

## 8) Criterios De Aceptacion

1. `src/` y `experiments/` sin artefactos pesados operativos.
2. `tests/` unico directorio de tests de codigo.
3. `data/` como unico hub de datasets/checkpoints/evals/visualizaciones.
4. Estructura navegable por frente/roadmap sin ambiguedad.
5. `.gitignore` y contratos de carpeta alineados con la estructura final.

---

## 9) Orden Recomendado De Implementacion

1. Cerrar migracion actual de `Documents/`.
2. Ejecutar Fase 1 (higiene critica en `src/`).
3. Ejecutar Fase 2 (consolidacion data).
4. Ejecutar Fase 3 (limpieza de `experiments/`).
5. Ejecutar Fase 4 (guardrails y contratos).

Con este orden minimizamos riesgo de romper ejecuciones y dejamos el repo listo para iteraciones de roadmap mas rapidas.
