# Explicacion Test 13G

**Estado**: `Phase A` CERRADA
**Fecha de corte**: 2026-03-01
**Rol en Gate 5B**: probar si el encoder puede preservar mas informacion musical reentrenandolo con un objetivo dual (`VICReg + reconstruction`), y decidir si el problema de generación se resuelve desde el entrenamiento del encoder o desde representaciones menos comprimidas.

> [!NOTE]
> La continuación operativa de esta línea ya no vive en este archivo sino en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G_faseB.md`, donde se documenta la nueva `Phase B` post-hoc sobre features pre-pooling.

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

### Phase A — λ Sweep (ejecutada)

- Descriptor: `D0` primero
- λ ∈ `{0.03, 0.1, 0.3}`
- 15 epochs por brazo
- seleccion robusta por promedio de las ultimas evaluaciones, no por pico aislado

Resultado observado:

| λ | best_S | last3_S | audio_f1 | midi_f1 |
|---|--------|---------|----------|---------|
| 0.03 | 64.6% | 63.2% | 0.1139 | 0.1183 |
| 0.10 | 64.4% | 62.8% | 0.1137 | 0.1172 |
| 0.30 | 64.4% | 63.6% | 0.1140 | 0.1187 |

Comparación clave:
- baseline `D0` sin decoder auxiliar: `73.4%`
- Test13G `Phase A`: `64.4-64.6%`

### Phase B — Confirmatoria (cancelada en su forma original)

El diseño original preveía elegir `λ*` y correr confirmatoria multi-seed. Esa fase se cancela porque `Phase A` mostró que el problema no es la selección de `λ`: la limitación dominante está en la compresión extrema hacia `z=256`.

### Phase C — Post-hoc original (cancelada)

El post-hoc original, también pensado sobre `z=256`, se cancela por la misma razón.

### Nueva hipótesis operativa

En lugar de decodificar desde `z=256`, la siguiente pregunta pasa a ser:

- ¿qué ocurre si el decoder se entrena sobre features pre-pooling `[B,188,1024]`?
- ¿los arms con descriptores (`a4r`, `d4a4`) retienen más estructura musical que `D0` en ese espacio intermedio?

---

## Metricas que importan

- `S`: retrieval. No debe colapsar.
- `midi_pr_f1`: que tan bien `z_midi` preserva estructura reconstuctiva.
- `audio_pr_f1`: metrica crucial. Si sube, significa que la alineacion audio<->MIDI se volvio mas rica musicalmente.
- `gap = midi_f1 - audio_f1`: mide la calidad del cruce de modalidad en terminos reconstructivos.

Regla metodologica del corte actual:

- **Sí** se puede leer `Phase A` como resultado negativo sobre una hipótesis precisa: `z=256` no alcanza para reconstrucción fiel, aunque siga siendo útil para retrieval.
- **No** se debe sobregeneralizar esa lectura como "el encoder no sirve para generación". Lo que quedó falsado es una ruta de decodificación demasiado comprimida.

---

## Estado operativo al corte actual

- Descriptor ejecutado: `D0`
- Fase cerrada: `Phase A` (`λ sweep`)
- Artefacto resumen: `data/gate5b_results/d0/test13g/test13g_sweep_summary.json`
- Generaciones visuales: `data/gate5b_results/d0/test13g/generation_samples/`

Lectura prudente:

- el barrido de `λ` no resolvió nada relevante;
- `z_audio` y `z_midi` reconstruyen casi igual de mal, lo que indica buena alineación pero poca capacidad reconstructiva;
- el cuello de botella dominante queda en la compresión, no en el ajuste fino del loss.

---

## Que conclusion SI se puede sostener hoy

1. Test 13G ya no es solo un plan: **Phase A quedó ejecutada y cerrada**.
2. Es el primer test de Gate 5B que modifica el entrenamiento del encoder.
3. La lectura principal es negativa pero útil: reentrenar el encoder con reconstrucción auxiliar no rescata la generación mientras la decodificación dependa de `z=256`.
4. Su función no es reemplazar Gate 5A, sino aclarar que el problema generativo no parece resolverse solo "desde el loss"; hay un límite estructural en la compresión.

---

## Artefactos clave

- `experiments/bias_control/gate5b/test13g_generative_encoder.py`
- `data/gate5b_results/d0/test13g/`
- `data/gate5b_results/d0/test13g/pr_validation_gate.json`
- `Documents/NOTAS_CLAUDE-CODEX.md` (secciones `15` y `16`)
