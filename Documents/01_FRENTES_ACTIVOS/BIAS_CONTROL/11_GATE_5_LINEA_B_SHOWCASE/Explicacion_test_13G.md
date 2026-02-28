# Explicacion Test 13G

**Estado**: EN CURSO (`Phase A`)
**Fecha de corte**: 2026-02-28
**Rol en Gate 5B**: probar si el encoder puede preservar mas informacion musical reentrenandolo con un objetivo dual (`VICReg + reconstruction`).

---

## Pregunta cientifica

Test 11 mostro que los embeddings compartidos (`z=256`) son utiles para retrieval pero muy pobres para reconstruccion (`frame F1 ~4-5%`).

Test 13G pregunta:

- si obligamos al encoder a reconstruir un piano-roll ademas de optimizar retrieval,
- ¿el embedding compartido puede preservar mas informacion musical decodificable?
- ¿esa mejora aparece solo en `z_midi`, o tambien cruza a `z_audio`?

---

## Que hace concretamente

Test 13G re-entrena el modelo completo desde foundation con una loss combinada:

`L_total = L_vicreg(z_audio, z_midi) + λ * BCE(MiniPRDecoder(z_midi), piano_roll_target)`

Componentes:

- `L_vicreg`: la loss normal de retrieval.
- `MiniPRDecoder`: decoder auxiliar liviano (~1.92M params) que toma `z_midi` y reconstruye un piano-roll `[188 x 88]`.
- `λ`: peso de la reconstruction loss; es el hiperparametro que se barre en `Phase A`.

---

## Por que este test existe despues de Pre-Proj A/B

La secuencia logica es:

1. **Test 11 baseline**: diagnostico del problema (`frame F1 ~4-5%`).
2. **Pre-Proj A/B**: cuantificacion del bottleneck de proyeccion (`81-88%` de info destruida en MIDI).
3. **Test 13G**: intento de resolver el problema desde el encoder, no desde la proyeccion.
4. **Gate 5A C1**: intento de resolver el mismo problema desde la projection head.

Test 13G no contradice Pre-Proj A/B. Lo complementa.

---

## Fases del experimento

### Phase A — λ Sweep

- Descriptor: `D0` primero
- λ ∈ `{0.03, 0.1, 0.3}`
- 15 epochs por brazo
- seleccion robusta por promedio de las ultimas evaluaciones, no por pico aislado

### Phase B — Confirmatoria

- mejor `λ*`
- `gen`: 30 epochs x 2 seeds
- `ctrl`: 30 epochs x 1 seed
- doble criterio de checkpoint: `best_S` y `best_recon`

### Phase C — Post-hoc

- event decoder mas pesado sobre los mejores checkpoints
- generacion de `.mid` para escucha humana

---

## Metricas que importan

- `S`: retrieval. No debe colapsar.
- `midi_pr_f1`: que tan bien `z_midi` preserva estructura reconstuctiva.
- `audio_pr_f1`: metrica crucial. Si sube, significa que la alineacion audio<->MIDI se volvio mas rica musicalmente.
- `gap = midi_f1 - audio_f1`: mide la calidad del cruce de modalidad en terminos reconstructivos.

Regla metodologica del corte actual:

- **No leer resultados de Test 13G como hallazgo** mientras siga en `Phase A`.
- Solo documentar estado operativo y motivacion hasta cerrar al menos `Phase B`.

---

## Estado operativo al ultimo corte documentado

- `tmux`: `test13g`
- Descriptor actual: `D0`
- Fase actual: `Phase A` (`λ sweep`)
- Primer brazo corriendo al ultimo corte: `λ = 0.03`
- ETA de `Phase A`: varias horas; se ejecuta secuencialmente sobre los tres valores de `λ`

Lectura prudente:

- losses bajando y retrieval subiendo temprano no significan exito;
- el test sigue siendo exploratorio hasta elegir `λ*` y correr confirmatoria.

---

## Que conclusion SI se puede sostener hoy

1. Test 13G ya no es solo un plan: **esta lanzado y corriendo**.
2. Es el primer test de Gate 5B que modifica el entrenamiento del encoder.
3. Su funcion no es reemplazar Gate 5A, sino medir si el problema se resuelve mejor desde el encoder que desde la proyeccion.

---

## Artefactos clave

- `experiments/bias_control/gate5b/test13g_generative_encoder.py`
- `data/gate5b_results/d0/test13g/`
- `data/gate5b_results/d0/test13g/pr_validation_gate.json`
- `Documents/NOTAS_CLAUDE-CODEX.md` (secciones `11.31`, `11.32`, `13.1-13.4`)
