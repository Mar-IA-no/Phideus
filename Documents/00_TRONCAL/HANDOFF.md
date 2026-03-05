# HANDOFF Operativo - Proyecto Phideus

---

## Proposito

Este documento es el puente de continuidad entre sesiones, agentes e instancias.
No reemplaza roadmaps ni decisiones formales: sintetiza estado operativo real y proximo paso ejecutable.

---

## Precedencia de fuentes

Cuando `collab_mode=on`:
1. `COLLAB/STATUS.md`: snapshot operativo "ahora" del ciclo collab.
2. `COLLAB/DECISIONS.md`: decisiones formales vigentes del protocolo.
3. `Documents/00_TRONCAL/HANDOFF.md` (este archivo): continuidad entre sesiones/instancias.
4. `COLLAB/HANDOFFS.md`: historial de traspasos entre agentes.

Cuando `collab_mode=off`:
1. `COLLAB/DECISIONS.md` (decisiones históricas válidas).
2. `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` + `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` (estado operativo real).
3. `Documents/00_TRONCAL/HANDOFF.md` (este archivo).
4. `COLLAB/STATUS.md` y `COLLAB/HANDOFFS.md` como referencia histórica (pueden quedar stale).

---

## Como usar este documento

1. Actualizar solo al cierre de un hito operativo o ante corte de contexto.
2. Mantener formato breve, verificable y con rutas concretas.
3. Registrar una sola "ultima decision valida" por entrada.
4. No duplicar contenido de roadmap: solo referenciarlo.

---

## Plantilla de entrada

```md
## YYYY-MM-DD HH:MM (UTC) - Handoff

### Metadata
- as_of_commit: <hash>
- collab_mode: on|off
- from_to (opcional): <origen -> destino>
- turn_summary_ref (opcional): <ruta o id>

### Estado real verificado
- item

### Ultima decision valida
- item

### Proximo paso unico recomendado
- item

### Bloqueantes / riesgos
- item

### Evidencia y archivos clave
- path
```

## 2026-03-05 21:10 (UTC) - Handoff

### Metadata
- as_of_commit: 2860dfe
- collab_mode: off

### Estado real verificado
- `Documents/NOTAS_CLAUDE-CODEX.md` sigue siendo la fuente operativa mas fresca; para reentrada importan especialmente:
  - `Sección 21`: Gate 5B v2, tesis de ventaja geométrica vs `feature richness`, Gate 6 con `a4r` local y `job 1144560` en UNC.
  - `Sección 22`: cierre formal del brazo Shazam de Escalón 1, documentos canónicos nuevos (`CIERRE_ESCALON1_SHAZAM.md`, `INDICE_ESCALON1_COMPLETO.md`) y decisión explícita de no mover `ESCALON_1/` ni `BIAS_CONTROL/`.
- Hay una sincronización documental en progreso y sin commit:
  - gran parte de `Sección 21` ya quedó propagada a troncales, BIAS_CONTROL y transversales;
  - la propagación de `Sección 22` todavía no está cerrada, sobre todo en la capa de navegación/índices y en `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_ESCALON_1.md`.
- El usuario pidió cortar esta conversación para liberar ventana de contexto y retomar en un chat nuevo con Codex.

### Ultima decision valida
- Reingresar en una nueva conversación con bootstrap corto, usando `HANDOFF` + `NOTAS` como ancla, y continuar la sincronización documental selectiva desde el estado ya editado del worktree.

### Proximo paso unico recomendado
- En el próximo chat: releer `AGENTS.md`, `CODEX.md`, `.codex/memory.md`, `Documents/00_TRONCAL/HANDOFF.md`, `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`, `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` y `Documents/NOTAS_CLAUDE-CODEX.md` (`Sección 21` y `Sección 22`), luego inspeccionar el diff actual antes de hacer más cambios.

### Bloqueantes / riesgos
- No asumir que la sync documental ya está terminada: el worktree sigue abierto y sin commit.
- Riesgo de dejar inconsistente Escalón 1 si se mezcla el cierre formal nuevo con textos viejos que todavía lo presentan como “en progreso”.
- No tocar `PENDIENTES.md`, `CLAUDE.md` ni `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Reconstruccion_final_claude.md` salvo pedido explícito del usuario.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/00_TRONCAL/HANDOFF.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/CIERRE_ESCALON1_SHAZAM.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_ESCALON_1.md`

## 2026-03-05 21:08 (UTC) - Handoff

### Metadata
- as_of_commit: 2860dfe
- collab_mode: off

### Estado real verificado
- `Documents/NOTAS_CLAUDE-CODEX.md` ya tiene dos bloques nuevos relevantes para reentrada:
  - `Sección 21`: Gate 5B v2, tesis de ventaja geométrica vs `feature richness`, estado real de Gate 6 (`a4r` local + `1144560` en UNC).
  - `Sección 22`: cierre formal del brazo Shazam de Escalón 1, nuevos documentos canónicos (`CIERRE_ESCALON1_SHAZAM.md`, `INDICE_ESCALON1_COMPLETO.md`) y decisión explícita de no mover `ESCALON_1/` ni `BIAS_CONTROL/`.
- El worktree quedó con una sincronización documental **en progreso y sin commit**:
  - `16` archivos trackeados modificados en troncal, BIAS_CONTROL, transversales y skill documental;
  - `Documents/Skills/phideus-doc-maintainer/` sigue como mirror untracked.
- La pasada ya integró gran parte de `Sección 21` en los entrypoints del repo:
  - Gate 5B completo,
  - Gate 6 activo con `job 1144560`,
  - fix `1da73fb`,
  - tesis geométrica propagada a docs clave.
- La pasada **no** quedó cerrada todavía para `Sección 22`:
  - falta terminar de propagar el cierre formal de Escalón 1-A y el índice maestro de Escalón 1 a la capa troncal/navegacional;
  - `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_ESCALON_1.md` sigue siendo un candidato claro a actualización o reencuadre como histórico.

### Ultima decision valida
- Continuar con una actualización documental amplia pero selectiva: no reescribir todo `Documents/**`, sino alinear entrypoints, docs canónicos e índices con `Sección 21` y `Sección 22`.

### Proximo paso unico recomendado
- En la nueva conversación, releer bootstrap + `NOTAS` (`Sección 21` y `Sección 22`), revisar el diff actual del worktree, y completar la integración de Escalón 1 antes de pensar en commit/push.

### Bloqueantes / riesgos
- Hay worktree sucio y no committeado; no asumir que el estado documental ya está cerrado.
- Riesgo de mezclar documentos canónicos nuevos de Escalón 1 con textos viejos que todavía lo presentan como “en progreso”.
- No tocar `PENDIENTES.md`, no tocar `CLAUDE.md`, y no mover `ESCALON_1/` ni `BIAS_CONTROL/` (decisión explícita de `NOTAS`).

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/CIERRE_ESCALON1_SHAZAM.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_ESCALON_1.md`
- `Documents/00_TRONCAL/HANDOFF.md`

## 2026-03-05 20:03 (UTC) - Handoff

### Metadata
- as_of_commit: 3c1b5a9
- collab_mode: off

### Estado real verificado
- Gate 5B ya quedó completamente cerrado: `Test05` `15/15`, `Test02` `4/4`, `Test11` `2/2` y `Test13G-B` `4/4` están incorporados a la lectura canónica.
- La lectura mecanística final de Gate 5B quedó más precisa:
  - retención pre-proyección: `d4a4=0.770`, `d4-a4r=0.748`, `a4r=0.712`, `D0=0.597`;
  - decodificabilidad pre-pooling: `D0(pool-188)=0.1089`, `d4a4=0.1037`, `a4r=0.1024`, `d4-a4r=0.1021`.
- Gate 6 pasó de “submitted” a estado operativo real:
  - el primer array UNC (`1144325`) falló por path absoluto de MAESTRO;
  - los `3` scripts SLURM Gate 6 ya quedaron corregidos y `Exp C` fue reenviado como `1144560`;
  - además hubo un fix en `main` para `build_pr_targets()` (`1da73fb`), necesario antes de que el job salga de la cola;
  - en local corre `a4r` con decoder AMT grande (`epoch ~35/80`, `F1=0.1485`, `onset_F1=0.0988`), ya claramente por encima del decoder de `13G-B`.
- `transkun`, `pretty_midi`, `midi2audio` y `mir_eval` ya están instalados en UNC; `Exp A` queda listo para submitir cuando haya slot y `Exp B` sigue condicionado por `Exp A`.

### Ultima decision valida
- Tratar Gate 5B como bloque definitivamente cerrado y leer Gate 6 como validación downstream ya activa en dos planos: corrida local de sanity/progreso y array UNC reenviado.

### Proximo paso unico recomendado
- Confirmar que UNC tenga el último `main` antes de que arranque `1144560`, monitorear `Exp C`, y abrir `Exp A` apenas haya disponibilidad de GPU.

### Bloqueantes / riesgos
- Si `1144560` arranca sin el `main` que incluye `1da73fb`, puede volver a fallar por targets en CPU.
- La mejora local de `a4r` en `Exp C` todavía no autoriza una lectura descriptor-guided: falta cerrar `80` epochs y comparar contra `D0`, `d4a4` y `d4-a4r`.
- Riesgo narrativo si se presenta Gate 6 como reemplazo de Escalón 2: sigue siendo validación downstream paralela.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Briefing_para_claude_unc.md`
- `experiments/bias_control/slurm/gate6_vicreg_decoder.sh`

## 2026-03-02 23:40 (UTC) - Handoff

### Metadata
- as_of_commit: c3ce32e
- collab_mode: off

### Estado real verificado
- Gate 5B permanece cerrado como bloque canónico: `Test05` `15/15`, `Test02` `4/4`, `Test13G-B` completo sin ventaja descriptor-guided.
- Gate 6 ya abrió una línea nueva de validación downstream por AMT:
  - `Exp 0` (`Transkun` baseline) quedó completo en local sobre segmentos de `4s` y `16s`.
  - `Exp C` quedó enviado a UNC como job `1144325` (`D0`, `d4a4`, `a4r`, `d4-a4r`).
  - `Exp A/B` siguen pendientes de habilitar `pip install transkun` en servidor.
- El antiguo `Gate 6` diagnóstico no debe reutilizarse como estado activo: quedó absorbido por Gate 5B `Test06`.

### Ultima decision valida
- Tratar Gate 6 AMT como validación downstream en paralelo: no reabre Gate 5B ni desplaza el foco principal de Escalón 2.

### Proximo paso unico recomendado
- Monitorear el cierre de `Exp C` en UNC y, en paralelo, preparar el entorno `Transkun` para destrabar `Exp A`.

### Bloqueantes / riesgos
- Riesgo narrativo si se mezcla `Gate 6 (diagnóstico histórico)` con `Gate 6 AMT`.
- Riesgo metodológico si `Exp A/B` se describen como activos antes de que UNC tenga realmente instalado `transkun`.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md`
- `experiments/bias_control/gate6/README.md`
- `experiments/bias_control/slurm/gate6_vicreg_decoder.sh`

## 2026-03-02 18:25 (UTC) - Handoff

### Metadata
- as_of_commit: 3a340eb
- collab_mode: off

### Estado real verificado
- `main` y `origin/main` quedaron alineados en `3a340eb`.
- Gate 5B ya quedó cerrado en documentación canónica y en artefactos sincronizados:
  - `Test05` multi-seed `15/15`,
  - `Test02` `4/4` (`real=83.0%`, `zero=75.0%`, `random=73.6%`, `shuffled=73.6%*`),
  - `Test13G-B` completo (`D0 pool-188=0.1089`, `d4a4=0.1037`, `a4r=0.1024`).
- La lectura estratégica vigente es: Escalón 1-C cerrado; Escalón 2 puede abrirse; Gate 5A sigue en paralelo oportunista.
- La homepage de `viz/` ya quedó reordenada con 12 rutas activas por gate/arm.

### Ultima decision valida
- Tratar Gate 5B como bloque cerrado y no seguir documentándolo como si `Test02` o `13G-B` permanecieran abiertos.

### Proximo paso unico recomendado
- Abrir planificación/ejecución de Escalón 2, manteniendo Gate 5A solo cuando haya recursos libres.

### Bloqueantes / riesgos
- `shuffled` se cerró por convergencia clara en `e20`; si apareciera un rerun completo a `e30`, la métrica podría moverse levemente, pero no cambia la lectura causal actual.

### Evidencia y archivos clave
- `results_unc/gate5b_param_matched/`
- `results_unc/gate5b_test13g/`
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md`

## 2026-03-01 20:04 (UTC) - Handoff

### Metadata
- as_of_commit: worktree local
- collab_mode: off

### Estado real verificado
- `results_unc/gate5b_multiseed/` ya contiene `15` corridas cerradas de `Test05` (`D0`, `a4r`, `d4-a4r`; `5` seeds por descriptor).
- La lectura multi-seed vigente queda en: `d4a4=84.1% +/- 2.3pp` (referencia multi-seed ya cerrada), `d4-a4r=81.2% +/- 2.5pp`, `a4r=80.7% +/- 1.9pp`, `D0=75.2% +/- 2.3pp`.
- `Test02` sigue parcial por reporte operativo de UNC: `real=83.0%` completo, `random≈73.0%`, `zero≈74.4%`, `shuffled` relanzado; esos artefactos aun no estan sincronizados localmente.
- `Test13G` Phase A quedo completa sobre `D0`: el barrido `λ={0.03,0.1,0.3}` no movio retrieval ni reconstruccion de forma relevante (`best_S≈64.4-64.6%`, `audio_f1≈0.114`, `midi_f1≈0.118`).
- La interpretacion vigente es negativa para la ruta original: decodificar desde `z=256` no alcanza; la siguiente hipotesis propuesta pasa a decoder post-hoc sobre features pre-pooling `[B,188,1024]`.

### Ultima decision valida
- Tratar `Test05` como cierre estadistico ya consolidado del bloque UNC y reencuadrar `Test13G` como falsacion del camino `z=256 -> piano-roll`, no como linea confirmatoria en progreso.

### Proximo paso unico recomendado
- Esperar/sincronizar el cierre completo de `Test02` y, en paralelo, decidir si se lanza la nueva `Phase B` generativa sobre features pre-pooling congeladas (`D0`, `a4r`, `d4a4`).

### Bloqueantes / riesgos
- Riesgo metodologico si se presenta `Test02` como cerrado antes de que entren los artefactos locales.
- Riesgo narrativo si se describe `Test13G` como fracaso general del encoder y no como limite especifico de la compresion a `z=256`.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `results_unc/gate5b_multiseed/`
- `data/gate5b_results/d0/test13g/test13g_sweep_summary.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G.md`

## 2026-03-01 22:16 (UTC) - Handoff

### Metadata
- as_of_commit: worktree local
- collab_mode: off

### Estado real verificado
- `Test13G` ya no está solo reencuadrado: `Phase B` post-hoc sobre features pre-pooling quedó implementada en `experiments/bias_control/gate5b/test13g_posthoc_decoder.py` y corre en local (`tmux test13g_b`).
- El pipeline generativo actual usa encoder congelado + `PostHocPRDecoder` sobre la salida pre-pooling del `audio_encoder.transformer`, con comparación planeada entre `D0`, `a4r`, `d4a4` y control `D0 pool-to-188`.
- `Test10` quedó efectivamente cerrado con paquete visual propio (`10 PNG + metadata`) y copia compartible consolidada en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/test10_visualizations/`.
- Se creó `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md` como informe exhaustivo del corte, y `Explicacion_test_13G_faseB.md` como explicación específica de la nueva fase generativa.

### Ultima decision valida
- Tratar `13G-A` como falsación del camino `z=256 -> piano-roll` y `13G-B` como probing exploratorio en curso sobre representaciones pre-pooling; no esperar `Test02` para documentar ese giro metodológico.

### Proximo paso unico recomendado
- Mantener sincronizada la capa documental con el estado real de `13G-B` y, cuando cierren `a4r/d4a4`, consolidar una lectura comparativa sin sobreatribuir causalidad al descriptor.

### Bloqueantes / riesgos
- Riesgo documental si `INFORME_COMPLETO_GATE5B.md` mantiene estados mezclados (`Test10` “no ejecutado” vs “completo”, `13G-B` citado pero no propagado al troncal).
- Riesgo metodológico si `13G-B` se lee como cierre paper-ready: el protocolo sigue siendo exploratorio `train/val`, no clausura final.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G_faseB.md`
- `experiments/bias_control/gate5b/test13g_posthoc_decoder.py`

## 2026-02-28 04:27 (UTC) - Handoff

### Metadata
- as_of_commit: worktree local
- collab_mode: off

### Estado real verificado
- `Test11` Pre-Proj A/B quedó completo para `D0` y `a4r`.
- Hallazgo fuerte verificado: `information retention ratio` `D0=0.597` vs `a4r=0.712`; la proyeccion MIDI 512→256 destruye aproximadamente `81-88%` de la informacion condicionante.
- `Test13G` ya fue lanzado en local (`tmux test13g`) y corre `Phase A` sobre `D0` con λ sweep; no hay lectura concluyente todavia.
- En `main`, `results_unc/` sigue en `9/15` corridas cerradas de Test05; en runtime UNC, el ultimo reporte ya marca `10/15` completadas (`a4r` 5/5, `d4-a4r` 5/5) y los 5 seeds de `D0` corriendo.
- Test02 en UNC sigue pending (`4/4`) y entrara a medida que se liberen slots.

### Ultima decision valida
- Documentar Test 11 Pre-Proj A/B como hallazgo ya cerrado y mantener Test13G solo como fase exploratoria en curso hasta cerrar `Phase B`.

### Proximo paso unico recomendado
- Cerrar `Phase A` de Test13G en `D0`, seleccionar `λ*` y mantener el seguimiento UNC distinguiendo siempre `sync local` de `runtime reportado`.

### Bloqueantes / riesgos
- Riesgo narrativo si se mezclan medias multi-seed reportadas por runtime UNC con artefactos aun no sincronizados a `results_unc/`.
- Riesgo metodologico si se presentan metricas tempranas de Test13G como resultado y no como monitoreo de `Phase A`.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `data/gate5b_results/D0/test11_preproj_ab.json`
- `data/gate5b_results/a4r/test11_preproj_ab.json`
- `data/gate5b_results/test11_preproj_ab_summary.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicaccion_pre-projection_test.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G.md`

## 2026-02-27 18:34 (UTC) - Handoff

### Metadata
- as_of_commit: worktree local
- collab_mode: off

### Estado real verificado
- En `main` siguen sincronizadas `9/15` corridas cerradas de Test05 en `results_unc/gate5b_multiseed/` (a4r `42/123/456/789/1337`, d4-a4r `42/123/456/789`).
- Estado UNC reportado al corte operativo (`2026-02-27 03:26 -03`): bloque `D0` de Test05 activo (`seed42` e9, `seed123` e8, `seed456/789` recién iniciados, `seed1337` pending).
- Test02 en UNC sigue en cola como array de 4 modos (`real/random/shuffled/zero`, job `1143844`, `nice=1000`).
- En local, `tmux preproj_ab` sigue activo: `D0 preproj_midi2events` cerró (CE `2.9449`, token_acc `0.3108`, frame F1 `0.1250`, `shuffle_gap=1.1498`) y `D0 preproj_audio2events` está entrenando (último hito visible: e9).

### Ultima decision valida
- Mantener la secuencia operativa `preproj_ab -> Test13G Phase A (D0)` y no bloquearla por el cierre total de UNC.

### Proximo paso unico recomendado
- Cerrar `preproj_ab` (terminar tramo `D0`, ejecutar tramo `a4r`) y luego liberar GPU para Test13G Phase A; en paralelo, sostener seguimiento UNC hasta completar bloque `D0` de Test05 y destrabar Test02.

### Bloqueantes / riesgos
- Riesgo de deriva narrativa si la documentación mezcla “sync local 9/15” con estado runtime UNC sin explicitar fuente temporal.
- Riesgo estadístico: Gate 5B sigue sin cierre multi-seed completo hasta completar `D0` en Test05 y correr Test02.

### Evidencia y archivos clave
- `data/gate5b_results/test11_preproj_ab.log`
- `results_unc/gate5b_multiseed/a4r_seed1337/final_results.json`
- `results_unc/gate5b_multiseed/d4-a4r_seed789/final_results.json`
- `Documents/NOTAS_CLAUDE-CODEX.md`

## 2026-02-27 04:10 (UTC) - Handoff

### Metadata
- as_of_commit: 6ab46e8
- collab_mode: off

### Estado real verificado
- Se incorporaron en `main` cinco corridas adicionales de Test05 desde UNC: `a4r_seed456/789/1337` y `d4-a4r_seed456/789`, con sus `eval_epoch25..30` y logs `g5b-ms_1143414_{7,8,10,11,13}.{out,err}`.
- Estado UNC actualizado de Test05: `9/15` cerradas, `1` running (`d4-a4r_seed1337`), `5` pending (`D0` seeds).
- Test13G quedó definido como alias estable (evita colisión con Test13 de demo retrieval) y con plan por fases `A->B->C`.

### Ultima decision valida
- Priorizar secuencia local `A/B pre-projection -> Test13G Phase A (D0)` mientras UNC completa Test05/Test02 en paralelo.

### Proximo paso unico recomendado
- Cerrar `d4-a4r_seed1337`, lanzar bloque `D0` de Test05 y luego ejecutar Test02 para cerrar robustez estadística Gate 5B.

### Bloqueantes / riesgos
- Riesgo de sobrelectura temprana: con `9/15` cerradas todavía no hay cierre estadístico completo del bloque UNC.
- Algunos estados `FAILED` de wrapper en logs SLURM no invalidan resultados si `final_results.json` está presente.

### Evidencia y archivos clave
- `results_unc/gate5b_multiseed/a4r_seed456/final_results.json`
- `results_unc/gate5b_multiseed/a4r_seed789/final_results.json`
- `results_unc/gate5b_multiseed/a4r_seed1337/final_results.json`
- `results_unc/gate5b_multiseed/d4-a4r_seed456/final_results.json`
- `results_unc/gate5b_multiseed/d4-a4r_seed789/final_results.json`
- `results_unc/logs/g5b-ms_1143414_7.out`
- `results_unc/logs/g5b-ms_1143414_8.out`
- `results_unc/logs/g5b-ms_1143414_10.out`
- `results_unc/logs/g5b-ms_1143414_11.out`
- `results_unc/logs/g5b-ms_1143414_13.out`

## 2026-02-27 01:48 (UTC) - Handoff

### Metadata
- as_of_commit: 7783b7e
- collab_mode: off

### Estado real verificado
- Se sincronizaron a `main` artefactos UNC cerrados de Gate 5B Test05 en `results_unc/gate5b_multiseed/` para `a4r` y `d4-a4r` (seeds `42/123`).
- Se importaron logs asociados de SLURM en `results_unc/logs/` (`g5b-ms_1143414_{1,2,4,5}.{out,err}`).
- Estado operativo UNC al corte: Test05 `4/15` cerradas, `6/15` running, `5/15` pending; Test02 `3/3` pending.
- Se mantiene política de artefactos livianos en repo: sin checkpoints `.pt` en `results_unc`.

### Ultima decision valida
- Consolidar robustez estadística UNC en modo incremental (import por run cerrado) sin esperar el cierre completo de Test05.

### Proximo paso unico recomendado
- Cerrar las `11` corridas restantes de Test05 y lanzar Test02 en UNC; luego consolidar tabla final multi-seed + lectura parameter-matched.

### Bloqueantes / riesgos
- Algunos jobs pueden reportar `FAILED` en SLURM por wrapper aunque `python` cierre y deje `final_results.json`; validar por artefacto, no solo por estado SLURM.
- Riesgo de deriva documental si no se refleja explícitamente que Test05 está en progreso parcial (no cerrado).

### Evidencia y archivos clave
- `results_unc/gate5b_multiseed/a4r_seed42/final_results.json`
- `results_unc/gate5b_multiseed/a4r_seed123/final_results.json`
- `results_unc/gate5b_multiseed/d4-a4r_seed42/final_results.json`
- `results_unc/gate5b_multiseed/d4-a4r_seed123/final_results.json`
- `results_unc/logs/g5b-ms_1143414_1.out`
- `results_unc/logs/g5b-ms_1143414_2.out`
- `results_unc/logs/g5b-ms_1143414_4.out`
- `results_unc/logs/g5b-ms_1143414_5.out`

## 2026-02-25 23:30 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 5B mantiene cierre local de `Test12/01/04/03/06/08/10` y ahora también `Test09` cerrado en `D0`, `d4a4`, `a4r`, `d4-a4r`.
- Resultados de `Test09` ya están consolidados en JSON canónico para los 4 arms.
- Persisten pendientes UNC de robustez estadística: `Test02` (parameter-matched) y `Test05` (multi-seed).
- El bloque generativo no lineal se mantiene como línea interna reservada (sin difusión pública por ahora).

### Ultima decision valida
- Tratar `Test09` como evidencia local cerrada y desplazar la ruta crítica a robustez estadística UNC.

### Proximo paso unico recomendado
- Ejecutar y consolidar `Test02/05` en UNC.

### Bloqueantes / riesgos
- Riesgo de sobrerreclamo sin CIs/multi-seed (`Test05`) y sin control parameter-matched (`Test02`).
- Riesgo de inconsistencias narrativas si quedan documentos con estado antiguo de `Test09` parcial.

### Evidencia y archivos clave
- `data/gate5b_results/D0/test09_invariance_suite.json`
- `data/gate5b_results/d4a4/test09_invariance_suite.json`
- `data/gate5b_results/a4r/test09_invariance_suite.json`
- `data/gate5b_results/d4-a4r/test09_invariance_suite.json`
- `Documents/NOTAS_CLAUDE-CODEX.md`

## 2026-02-25 11:30 (UTC) - Handoff

### Metadata
- as_of_commit: e7ecd7e
- collab_mode: off

### Estado real verificado
- Gate 5B mantiene cierre local de `Test12/01/04/03/06/08/10`.
- `Test09` queda en cierre parcial verificable: `D0` y `d4a4` completos en JSON canónico; `a4r` y `d4-a4r` pendientes.
- Se detectó deriva documental menor entre notas narrativas y resultados canónicos; queda priorizado usar JSON como fuente de verdad.

### Ultima decision valida
- Publicar estatus de Test09 como parcial (no “cerrado”) hasta completar los dos arms faltantes y consolidar lectura comparativa final.

### Proximo paso unico recomendado
- Completar `Test09` en `a4r` y `d4-a4r`, integrar resumen final de invariancia y luego ejecutar fase UNC pendiente (`Test02/05`).

### Bloqueantes / riesgos
- Riesgo de sobregeneralizar robustez de invariancia con solo 2 de 4 arms.
- Riesgo de inconsistencias narrativas si se citan tablas históricas sin validar contra JSON canónico.

### Evidencia y archivos clave
- `data/gate5b_results/D0/test09_invariance_suite.json`
- `data/gate5b_results/d4a4/test09_invariance_suite.json`
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md`

## 2026-02-25 06:30 (UTC) - Handoff

### Metadata
- as_of_commit: 8343c06
- collab_mode: off

### Estado real verificado
- Gate 5B dejó de estar en “cierre parcial”: el paquete local quedó consolidado con `Test12`, `Test01`, `Test04`, `Test03`, `Test06`, `Test08` y `Test10` cerrados.
- `Test09` (invariance suite) continúa en ejecución local como próximo cierre técnico del bloque.
- La fase UNC de validación estadística queda acotada a `Test02` (parameter-matched ablations) y `Test05` (multi-seed replication).
- El paquete visual de referencia Gate 5B quedó explícito para documentación/showcase (`24 PNG` + `6 GIF`).

### Ultima decision valida
- Tratar Gate 5B como cierre local robusto de mecanismo/rendimiento, manteniendo abiertas solo las validaciones pendientes (`Test09` local y `Test02/05` en UNC) antes de cierre científico final.

### Proximo paso unico recomendado
- Cerrar `Test09` y anexar su lectura al resumen científico unificado de Gate 5B; luego planificar ejecución UNC de `Test02/05`.

### Bloqueantes / riesgos
- Riesgo de sobreextender conclusiones sin bloquear la variabilidad entre seeds (`Test05`) ni el control de confound paramétrico (`Test02`).
- Riesgo narrativo si se mezcla en documentos “cierre local” con “cierre total” sin etiquetar explícitamente.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/`

## 2026-02-25 02:59 (UTC) - Handoff

### Metadata
- as_of_commit: a59d24e
- collab_mode: off

### Estado real verificado
- Gate 5B quedó destrabado con loader universal y evaluación reproducible para checkpoints no `eval_compatible`.
- Test12 (scoreboard canónico) quedó cerrado y consistente con históricos: `D0=73.4%`, `d4a4=83.8%`, `a4r=82.0%`, `d4-a4r=79.8%`.
- Test01 (causal ablation) quedó cerrado para 5 arms (`D0`, `d4`, `d4a4`, `a4r`, `d4-a4r`); la degradación fuerte aparece al ablacionar audio (`A4/A4r`) y el efecto D4 en modelos duales top es marginal.
- Se actualizaron documentos de cierre y notas para paper con tabla consolidada de resultados Test01/Test12.

### Ultima decision valida
- Tratar Gate 5B Test01/Test12 como cierre verificable de validez causal inicial (scoreboard + ablation) y no extrapolar “techo” sin completar Test04/Test06+.

### Proximo paso unico recomendado
- Ejecutar y cerrar Test04 (transposition invariance) usando cache de embeddings, luego integrar resultado al paquete de evidencia Gate 5B.

### Bloqueantes / riesgos
- Test04 sigue incompleto (`d4-a4r` pendiente), por lo que la lectura de invariancia aún es parcial.
- Riesgo metodológico si se concluye “D4 no sirve” sin separar: (1) señal marginal histórica D4> D0 y (2) dominancia de audio en arquitecturas duales top.

### Evidencia y archivos clave
- `experiments/bias_control/gate5b/test12_scoreboard.py`
- `experiments/bias_control/gate5b/test01_causal_ablation.py`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md`
- `Paper/notas_para_paper.md`

## 2026-02-24 23:26 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Se reforzó la persistencia de rol de Codex en `CODEX.md` y `.codex/memory.md` con guardrail bloqueante.
- Quedó explícito que Codex opera por defecto como auditor/documentador y no implementa ni ejecuta runs sin orden explícita del usuario.

### Ultima decision valida
- Tratar el guardrail de rol como regla operativa dura post-compactación y post-reentrada de contexto.

### Proximo paso unico recomendado
- Mantener Gate 4.5/Gate 5B en modo auditoría documental/técnica para Codex, salvo instrucción explícita de cambio de rol.

### Bloqueantes / riesgos
- Riesgo de desvío de rol tras compactación si no se revalida `CODEX.md` + `.codex/memory.md` al inicio del turno.

### Evidencia y archivos clave
- `CODEX.md`
- `.codex/memory.md`
- `Documents/NOTAS_CLAUDE-CODEX.md`

## 2026-02-23 22:00 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.5 queda en cierre parcial verificable: bloque stretched/hold cerrado y bloque `cosine-tail` en finalización.
- `d4-a4r 60ep` (cosine stretched) completó con `S=79.8%` (e55), empatando su 30ep.
- `moe-dual 60ep` quedó **dead** por time limit; best `S=73.0%` en e30 (no sostenido).
- `a4r ctail 60ep` completó con `S=80.6%`; `D0 ctail` y `d4a4 ctail` siguen en curso; `d4-a4r ctail` re-submitted (`Job 1143330`).

### Ultima decision valida
- Mantener Gate 4.5 abierto hasta cerrar la comparativa final 30ep vs stretched vs `cosine-tail` con métricas canónicas alineadas.

### Proximo paso unico recomendado
- Cerrar `d4a4/D0/d4-a4r` en bloque `cosine-tail` y publicar tabla final única de Gate 4.5 (`S`, `A2M`, `M2A`, `hard_neg`).

### Bloqueantes / riesgos
- Riesgo de mezclar resultados cerrados con corridas todavía en curso y adelantar conclusiones de scheduler sin corte final homogéneo.
- Riesgo de abrir ejecución plena de Gate 5A con Gate 4.5 todavía parcialmente abierto.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md`
- `results_unc/batch_60ep_ctail_a4r/final_results.json`
- `results_unc/batch_60ep_ctail_d0/eval_per_epoch/eval_epoch55.json`
- `results_unc/batch_60ep_ctail_d4a4/eval_per_epoch/eval_epoch50.json`
- `results_unc/batch_60ep_d4-a4r/final_results.json`

## 2026-02-22 12:00 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Se formaliza **Gate 4.5 (LR Schedule Optimization)** entre Gate 4.4 y Gate 5A/5B.
- Corridas cerradas en Gate 4.5: `d4a4 60ep` (`S=83.8%`), `a4r 60ep` (`S=79.4%`), `D0 60ep` (`S=72.8%`), `t3-wt 50ep hold` (`S=81.2%`).
- Pendientes en cola UNC: `d4-a4r 60ep`, `moe-dual 60ep`, y batch `cosine-tail` (`D0`, `d4a4`, `a4r`, `d4-a4r`).
- Árbol documental de BIAS_CONTROL reordenado para secuencia canónica: `08_GATE_4_4 -> 09_GATE_4_5 -> 10_GATE_5A -> 11_GATE_5B`.

### Ultima decision valida
- Tratar el bloque 50ep/60ep como Gate propio de optimizacion de scheduler y no como apéndice informal post-4.4.

### Proximo paso unico recomendado
- Cerrar pendientes stretched y ejecutar batch `cosine-tail` para comparación alineada contra 30ep/stretched (`S`, `A2M`, `M2A`, `hard_neg`).

### Bloqueantes / riesgos
- Riesgo de comparar resultados de documentos con distintos cortes temporales sin etiquetar claramente fecha/epoch.
- Riesgo de adelantar decisiones de Gate 5A con Gate 4.5 todavía abierto.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`

## 2026-02-21 14:40 (UTC) - Handoff

### Metadata
- as_of_commit: ce26296
- collab_mode: off

### Estado real verificado
- `results_unc/` quedó en 182 JSON al último import consolidado en `main`; incluye `batch_60ep_a4r` completo y cortes parciales de `batch_60ep_d0`, `batch_60ep_d4a4`, `gate44_t3-wt_scratch_50ep_hold`.
- En extendidos: `a4r 60ep` cerró en `S=79.4%`; `D0 60ep` va en `S@e40=72.4%`; `d4a4 60ep` va en `S@e40=82.6%`; `t3-wt 50ep hold` va en `S@e40=80.6%`.
- `d4-a4r 60ep` y `moe-dual 60ep` permanecen en cola.
- Quedó implementado el scheduler `cosine-tail` y enviados scripts `batch_60ep_ctail_{d0,d4a4,a4r,d4-a4r}.sh` para contraste de scheduler.

### Ultima decision valida
- Tratar 60ep/50ep y cosine-tail como validación de dinámica temporal/scheduler, sin reescribir el ranking cerrado de 30ep hasta tener cortes equivalentes.

### Proximo paso unico recomendado
- Cerrar primero `D0` y `d4a4` 60ep, luego ejecutar lote `cosine-tail` y comparar `S/A2M/M2A/hard_neg` en epochs alineados (`e30`, `e40`, `e60`).

### Bloqueantes / riesgos
- Mezclar en una misma tabla valores de monitoreo de cola (UNC) con artefactos todavía no importados en `results_unc/`.
- Sobreinterpretar mejoras/pérdidas por scheduler sin control explícito de epoch de comparación.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `results_unc/batch_60ep_a4r/final_results.json`
- `results_unc/batch_60ep_d4a4/eval_per_epoch/eval_epoch40.json`
- `results_unc/batch_60ep_d0/eval_per_epoch/eval_epoch40.json`
- `results_unc/gate44_t3-wt_scratch_50ep_hold/eval_per_epoch/eval_epoch40.json`
- `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`

## 2026-02-19 15:30 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.4 quedó cerrado en screening corto: 24 brazos consolidados (21 originales + `moe-a4-v2/v3/v4`) con tabla completa `S/A2M/M2A/hard_neg`.
- Se cerraron todos los runs largos scratch 30ep del bloque activo: `d4a4=83.6`, `a4r=82.0`, `d4-a4r=79.8`, `t3-wt=79.8`, `d4a4r=74.4`, `moe-dual=72.6`.
- Se documentó hallazgo de scheduler: en 30ep cosine deja LR casi nulo en tramo final; quedó implementado `--lr-hold-fraction` + logging `lr_mult`.
- Quedó preparado el bloque siguiente en UNC: batch 60ep (`D0`, `d4a4`, `a4r`, `d4-a4r`, `moe-dual`) + `t3-wt` 50ep hold.

### Ultima decision valida
- Tratar el bloque de 60ep/hold como validación causal de dinámica temporal (no como reemplazo de ranking 30ep ya cerrado).

### Proximo paso unico recomendado
- Consolidar el primer corte verificable de las corridas 60ep/hold y compararlo contra el bloque cerrado 30ep usando las mismas métricas canónicas.

### Bloqueantes / riesgos
- Riesgo de comparar corridas en distintos puntos temporales sin alinear epochs de referencia (`e30`, `e50`, `e60`).
- Riesgo de confundir mejora por mayor presupuesto de entrenamiento con mejora por descriptor/arquitectura.

### Evidencia y archivos clave
- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`
- `results_unc/gate44_t3-wt_scratch_30ep/final_results.json`
- `results_unc/gate44_moe-dual_scratch_30ep/final_results.json`
- `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`

## 2026-02-18 18:20 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.4 pasa a corte parcial avanzado con 6 brazos ya cerrados en e5: `t3-wt` (67.6), `t3-tri` (65.0), `t3-anc` (42.2), `moe-a4` (58.2), `film-a4` (59.2), `film-d4` (58.6).
- `film-dual` y `moe-dual` tienen e3 provisional (`58.2` y `59.2`) y siguen con e5 pendiente.
- Tabla larga de 30ep sigue con `d4-a4r` y `t3-wt` en curso; se agrega `moe-dual` scratch 30ep (job UNC `1142665`, en cola/arranque al envío).
- Se sincronizó documentación troncal/frente/transversal con corte `2026-02-18`.

### Ultima decision valida
- Mantener ranking oficial separado entre resultados cerrados e5 y resultados provisionales e3 hasta completar los 8 brazos en la misma ventana.

### Proximo paso unico recomendado
- Cerrar `film-dual` y `moe-dual` en e5 y publicar tabla Gate 4.4 completa (`S`, `A2M`, `M2A`, `hard_neg`) sin filas provisionales.

### Bloqueantes / riesgos
- Riesgo de mezclar en la misma tabla filas cerradas (e5) con provisionales (e3) y degradar comparabilidad.
- Riesgo operativo si `d4-a4r`/`t3-wt`/`moe-dual` 30ep quedan sin trazabilidad homogénea con los runs largos ya cerrados.

### Evidencia y archivos clave
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md`

## 2026-02-17 23:27 (UTC) - Handoff

### Metadata
- as_of_commit: bd73402
- collab_mode: off

### Estado real verificado
- Se importaron a `main` los artefactos UNC en `results_unc/` (114 archivos: JSON + logs) mediante commit `bd73402`.
- Gate 4.4 queda en corte parcial verificable por artefacto:
  - `t3-wt` (`S@e5=67.6%`), `t3-tri` (`S@e5=65.0%`), `t3-anc` (`S@e5=42.2%`), `moe-a4` (`S@e5=58.2%`, best `S@e3=58.8%`).
  - `film-a4` (`S@e3=59.2%`) y `film-d4` (`S@e3=58.8%`) con e5 pendiente.
  - `film-dual` y `moe-dual` aún sin eval estructurada consolidada.
- Runs largos ya consolidados en artefactos UNC: `a4r-scratch` (`S=82.0%`) y `d4a4r-scratch` (`S=74.4%`); `d4-a4r` scratch relanzado y en curso.

### Ultima decision valida
- Mantener reporte orientado a datos comparables (`S`, `A2M`, `M2A`, `hard_neg`) en e3/e5 y diferir decisión de continuidad hasta completar los 8 brazos Gate 4.4.

### Proximo paso unico recomendado
- Cerrar `film-a4`, `film-d4`, `film-dual`, `moe-dual` y publicar tabla final única Gate 4.4 con artefactos estructurados.

### Bloqueantes / riesgos
- Riesgo de cierre asimétrico (unos brazos con e5 y otros sin eval estructurada) que impida comparación final en una sola tabla.
- Riesgo de mezclar señales `quick_val` con structured pool canónico.

### Evidencia y archivos clave
- `results_unc/gate44/t3-wt/final_results.json`
- `results_unc/gate44/t3-tri/final_results.json`
- `results_unc/gate44/t3-anc/final_results.json`
- `results_unc/gate44/moe-a4/final_results.json`
- `results_unc/gate44/film-a4/eval_per_epoch/eval_epoch3.json`
- `results_unc/gate44/film-d4/eval_per_epoch/eval_epoch3.json`
- `results_unc/gate43_a4r_scratch_30ep/final_results.json`
- `results_unc/gate43_d4a4r_scratch_30ep/final_results.json`

## 2026-02-17 14:42 (UTC) - Handoff

### Metadata
- as_of_commit: 84da048
- collab_mode: off

### Estado real verificado
- Gate 4.4 quedó implementado en `main` y los jobs de screening UNC fueron enviados (`8 brazos x 5ep`).
- Protocolo operativo de screening activo: `foundation_locked_e25.pt` + `--freeze-policy run-d` + eval estructurada en epochs 3 y 5.
- Documentación troncal y de frente se alinea a estado "Gate 4.4 screening en curso".

### Ultima decision valida
- Mantener comparabilidad estricta del screening Gate 4.4 contra referencias de Gate 4.3 (`d4a4@5ep=69.8%`, `D0@5ep=60.2%`) antes de decidir Fase 2 (30ep).

### Proximo paso unico recomendado
- Consolidar tabla única `S@e3/S@e5` para los 8 brazos y emitir decisión GO/NO-GO de pase a 30ep.

### Bloqueantes / riesgos
- Cola/variabilidad SLURM puede desfasar cierre simultáneo de los 8 brazos.
- Cualquier corrida sin `run-d` rompe comparabilidad con baseline corto de referencia.

### Evidencia y archivos clave
- `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md`

## 2026-02-17 06:46 (UTC) - Handoff

### Metadata
- as_of_commit: 72c818d
- collab_mode: off

### Estado real verificado
- En UNC se envió `d4a4r-scratch` 30ep y quedó en estado `PENDING`.
- `a4r-scratch` permanece en cola y ambos jobs quedan como contraste scratch vs scratch (single reverse vs dual reverse) frente al baseline `d4a4-scratch=83.6%`.
- Código de soporte `d4a4r` ya existe en `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`.

### Ultima decision valida
- Mantener `a4r-scratch` y `d4a4r-scratch` como último bloque comparativo de mecanismo dentro de continuidad Gate 4.3 antes de abrir ejecución arquitectural de Gate 4.4.

### Proximo paso unico recomendado
- Monitorear transición `PENDING -> RUNNING` y validar primeros checks (estabilidad, NaN, evolución de loss) antes de declarar corrida larga estable.

### Bloqueantes / riesgos
- Demora de cola UNC puede aplazar la decisión de mecanismo previa a Gate 4.4.
- Si falla infraestructura SLURM al arranque, se pierde ventana de comparación directa entre runs scratch.

### Evidencia y archivos clave
- `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`

## 2026-02-17 06:15 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.3 quedó cerrado con 13 brazos de 5 epocas (`D0/D4/A4/A7/A4x/A7x/D4x/d4a4/d4a4cm/A4r/D4r/A8/A9`).
- Mejor brazo corto: `d4a4` con `S=69.8%` (`+9.6pp` vs D0).
- `d4a4-scratch` 30ep completado con `S=83.6%` (record), `hard_neg=95.2%`.
- Multi-seed en `d4a4-scratch e30`: `S=84.1% +/- 2.3pp` (5 seeds).
- Gate 4.3 Fase 5 cerrada en UNC; `A4r` emerge como mejor single-descriptor (`S=68.6%`).

### Ultima decision valida
- No abrir nuevas variantes dentro de Gate 4.3 salvo `a4r-scratch` 30ep en UNC (ya en cola) como comparación scratch vs scratch.
- Transición formal a Gate 4.4 (Third Tower + FiLM + MoE), manteniendo Gate 5A/5B como etapas posteriores.

### Proximo paso unico recomendado
- Completar `a4r-scratch` 30ep en UNC y usar ese resultado para fijar baseline interno antes del arranque de Gate 4.4.

### Bloqueantes / riesgos
- Si `a4r-scratch` falla por issues de infraestructura SLURM, se retrasa decisión de mecanismo para Gate 4.4.
- Saltar a Gate 4.4 sin ese contraste podría mezclar efectos de arquitectura con efectos de mecanismo.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000/`
- `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`

## 2026-02-14 06:40 (UTC) - Handoff

## 2026-02-14 14:45 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.3 pasó de "arranque" a ejecución efectiva en `gate43_20260214_1000`.
- `D0` quedó cerrado en 5ep (best `S=60.2%`, e3).
- `D4` quedó cerrado en 5ep (best `S=63.6%`, e5), con mejora `+3.4pp` vs `D0`.
- `A4` completó e1-e3 (`S=35.4% -> 51.2% -> 61.0%`) y continúa e4-e5.
- `A7`, `D4+A4`, `D4+A7` permanecen en cola de ejecución, pero con ajuste de orden acordado.

### Ultima decision valida
- Mantener evaluación canónica por cada epoch (criterio científico, sin reducción de frecuencia).
- Al terminar `A4`, cortar el loop actual y relanzar desde `A7` con orden:
  `A7 -> A4x -> A7x -> D4+A4 -> D4+A7`.

### Proximo paso unico recomendado
- Terminar Gate 4.3 en secuencia (`A4` cierre -> `A7` -> `A4x` -> `A7x` -> `D4+A4` -> `D4+A7`) y consolidar tabla comparativa final para decisión Gate 4.4.

### Bloqueantes / riesgos
- Conclusiones tempranas sobre `A4` antes de e5 pueden sesgar la lectura (recovery no lineal).
- Si no se corta el script tras `A4`, el orden viejo (`A7 -> duales`) rompería la comparación directa `concat vs cross` antes de duales.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000/`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

## 2026-02-14 06:40 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.2 queda cerrado con run `D4` extendido a 8 epocas.
- Mejor punto del run: `epoch7` con `S=64.2%`, `A2M=65.0%`, `M2A=64.2%`, `hard_neg=91.6%`.
- `D4 8ep` confirma el techo observado en `D4 3ep` (`S=64.2%`) y mejora robustez en `hard_neg`.
- Roadmap operativo pasa a Gate 4.3 (bloque causal bifurcado) con etapa piloto previa.

### Ultima decision valida
- Gate 4.2 cerrado formalmente; no se requieren mas extensiones para `D4` dentro de esta fase.
- Gate 4.3 se inicia con pilotos (`a4`, `a7`, `d4a4`, `d4a7`) antes del barrido 5ep de los 6 brazos.

### Proximo paso unico recomendado
- Ejecutar pilotos 1 epoca/100 batches para `a4`, `a7`, `d4a4`, `d4a7`; si pasan checks de estabilidad, lanzar barrido Gate 4.3 completo (5ep fresh por brazo).

### Bloqueantes / riesgos
- Saltar pilotos puede ocultar problemas de VRAM/NaN/evaluacion en descriptores de audio.
- Mezclar runs resume/fresh en comparacion factorial invalida inferencia causal.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/resultados_gate_4.2.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/decisiones_gate_4.2.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`

## 2026-02-14 02:15 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Bloque A v1.1 permanece cerrado con foundation lock en `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`.
- Gate 4.2 queda en cierre con extension `D4` a 8 epocas (no se reetiqueta como Gate 4.3).
- Roadmap actualizado con bifurcacion formal:
  - linea MIDI temperada,
  - linea Audio armonia natural,
  - linea Dual.
- Estructura documental creada para:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/`
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_BIFURCACION_RATIO/`

### Ultima decision valida
- Gate 4.3 se ejecuta como bloque causal corto (`D0`, `D4-only`, `A4-only`, `A7-only`, `D4+A4`, `D4+A7`), todo fresh.
- Gate 4.4 absorbe el barrido amplio (MIDI: `D3/D8/D9/D10/D2/D5/D6/D7`; Audio: `A1/A2/A3/A5/A6`).

### Proximo paso unico recomendado
- Finalizar Gate 4.2 (`D4` 8 ep), verificar persistencia de mejora y abrir ejecucion de Gate 4.3.

### Bloqueantes / riesgos
- Mezclar resultados reanudados (`--resume`) con fresh en comparativas factoriales puede sesgar conclusion.
- Perder separacion de paradigma (MIDI temperado vs audio no temperado) invalida la lectura cientifica del nuevo diseño.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_BIFURCACION_RATIO/plan_gate_4.4.md`

## 2026-02-13 16:50 (UTC) - Handoff

### Metadata
- as_of_commit: ae40717
- collab_mode: off

### Estado real verificado
- `Run D-02` cerró 30 épocas; mejor single-seed en `epoch25` (`S=61.8%`, `A2M=61.8%`, `M2A=62.4%`, `hard_neg=90.4%`) y empate de `S` con `epoch26`.
- Re-evaluación multi-seed (`42/123/456/789`) entre `e25` y `e26` completada; se prioriza `e25` por estabilidad operativa.
- Foundation lock formal definido en `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`.
- `explore_foundation.py` ejecutado con checkpoint bloqueado y artefactos guardados en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/`.

### Ultima decision valida
- Bloque A v1.1 queda cerrado y la etapa activa pasa a screening Gate 4.2 sobre foundation bloqueado.
- `Gate2R-lite` se mantiene en backlog post Gate 4.2 (higiene metodológica, no bloqueante).

### Proximo paso unico recomendado
- Iniciar screening canónico de Gate 4.2 (D0 -> D1/D4) usando `foundation_locked_e25.pt`.

### Bloqueantes / riesgos
- Usar checkpoints mutables (`best_model_base.pt`) para decisiones de Gate 4.2 rompe trazabilidad.
- Desviarse del protocolo canónico (`pool=256`, `queries=500`, `seed=42`) invalida comparabilidad causal entre descriptores.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/multiseed_reeval.json`
- `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/explore_summary.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/Informe_final_bloqueA_Claude.md`

## 2026-02-12 15:55 (UTC) - Handoff

### Metadata
- as_of_commit: 4417542
- collab_mode: off

### Estado real verificado
- `Run D-02` sigue activo y alcanzó nuevo best parcial en `epoch18`: `S=59.6%`, `A2M=60.8%`, `M2A=59.6%`, `hard_neg=91.0%`.
- Foundation provisional se mantiene en `Run D ep5` hasta cierre formal de `Run D-02`.
- Gate 4.2 mantiene codigo implementado en paralelo (`gate42_training.py`, `ratio_descriptors.py`, ajustes de dataset/preflight) y screening bloqueado hasta foundation lock definitivo.
- Visualizaciones 3D de arquitectura publicadas y operativas en `https://altermundi.github.io/Phideus/`.

### Ultima decision valida
- Mantener secuencia cientifica en serie: primero cierre/lock de foundation (`C5 vs D5 vs D-02(best)`), luego screening Gate 4.2.
- La exploracion cualitativa (`explore_foundation.py`) se ejecuta solo con checkpoint inmutable post-lock.

### Proximo paso unico recomendado
- Cerrar `Run D-02`, consolidar tabla canonica C/D/D-02 y fijar foundation lock definitivo.

### Bloqueantes / riesgos
- Ejecutar screening Gate 4.2 o exploracion final sobre `best_model_base.pt` mutable rompe trazabilidad/reproducibilidad.
- Promover `D-02` antes de cierre completo por pico parcial puede sesgar decision.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/eval_per_epoch/eval_epoch18.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `experiments/bias_control/explore_foundation.py`
- `README.md`

## 2026-02-12 08:20 (UTC) - Handoff

### Metadata
- as_of_commit: fe64b6c
- collab_mode: off

### Estado real verificado
- `Run D-02` activo en `data/bias_control_medium/training_outputs/bloqueA_runD-02` (modo `run-d`, 30 epocas, base `gate2/checkpoint_epoch45.pt`).
- `Run D ep5` se mantiene como foundation provisional hasta cierre de `Run D-02`.
- Screening de Gate 4.2 sigue bloqueado hasta foundation lock definitivo.

### Ultima decision valida
- Foundation lock final queda diferido a comparativa robusta `C5 vs D5 vs D-02(best)`.
- Paralelizacion permitida se mantiene: codigo Gate 4.2 en paralelo, decision cientifica en serie.

### Proximo paso unico recomendado
- Cerrar `Run D-02`, consolidar tabla canonica C/D/D-02 y fijar foundation lock definitivo.

### Bloqueantes / riesgos
- Iniciar screening Gate 4.2 antes del lock final invalida comparabilidad causal `D0 vs Dx`.
- Si `D-02` no supera/empata robustamente, no debe desplazar foundation provisional por inercia de corrida larga.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/config.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## 2026-02-12 07:46 (UTC) - Handoff

### Metadata
- as_of_commit: e57e2fc
- collab_mode: off

### Estado real verificado
- Run D cerrado en epoch 5 con metricas canonicas: `S=51.0%`, `A2M=51.0%`, `M2A=51.8%`, `hard_neg=89.2%`.
- Tabla A/B/C/D consolidada en single-seed: D > C > B > A.
- Gate 4.2 mantiene implementacion de codigo lista para continuar, pero screening sigue bloqueado hasta foundation lock definitivo.

### Ultima decision valida
- Se mantiene la secuencia acordada: foundation lock definitivo antes de screening Gate 4.2.
- Estado de lock: `Run D ep5` queda como foundation provisional; cierre final pendiente de desempate robusto `C5 vs D5`.

### Proximo paso unico recomendado
- Ejecutar desempate robusto `C5 vs D5` (reevaluacion multi-seed) y cerrar foundation lock definitivo.

### Bloqueantes / riesgos
- Iniciar screening Gate 4.2 sin lock definitivo rompe comparabilidad causal `D0 vs Dx`.
- Diferencia single-seed `D-C` en `S` es positiva pero acotada (`+1.6pp`), por lo que conviene cierre robusto antes de promover.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `data/bias_control_medium/training_outputs/bloqueA_runC/final_results.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## 2026-02-12 03:16 (UTC) - Handoff

### Metadata
- as_of_commit: b50e446
- collab_mode: off

### Estado real verificado
- Run B cerrado con mejor epoch 3 (`S=43.2%`, `A2M=43.2%`, `M2A=43.4%`, `hard_neg=85.2%`).
- Run C en ejecucion, con evaluacion cerrada al menos hasta epoch 2 (`S=35.0%`, `hard_neg=79.6%`) y checkpoint epoch 3 guardado.
- Gate 4.2 sigue sin screening activo; implementacion de codigo aun pendiente.

### Ultima decision valida
- Secuencia acordada: cerrar Run C -> comparativa A/B/C -> Run D condicional (DEC-007) -> foundation lock definitivo -> screening Gate 4.2.
- Gate 4.2 codigo puede avanzar en paralelo; Gate2R-lite queda en backlog post Gate 4.2 (higiene, no bloqueante).

### Proximo paso unico recomendado
- Cerrar Run C y resolver foundation lock A/B/C(/D) antes de habilitar screening de Gate 4.2.

### Bloqueantes / riesgos
- Si se corre screening Gate 4.2 sin foundation lock, se rompe comparabilidad causal.
- `experiments/bias_control/bloqueA_training.py` mantiene cambios locales sin commit y requiere auditoria antes de relanzes.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runB/eval_per_epoch/eval_epoch3.json`
- `data/bias_control_medium/training_outputs/bloqueA_runC/eval_per_epoch/eval_epoch2.json`
- `data/bias_control_medium/training_outputs/bloqueA_runC_log.txt`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## 2026-02-12 00:00 (UTC) - Handoff inicial

### Metadata
- as_of_commit: 593a11d
- collab_mode: off

### Estado real verificado
- Bloque A v1.1 activo en BIAS_CONTROL, con S0 y Run A cerrados, Run B cerrado y Run C en curso (evaluado al menos hasta epoch 2).
- El plan vigente de Gate 4.2 ratio-centrico esta consolidado en:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md`
- El repositorio usa `COLLAB OFF` por defecto salvo activacion explicita del usuario.

### Ultima decision valida
- Gate 4.2 mantiene protocolo canonico y guardrails anti-variable-fantasma; su implementacion de codigo puede correr en paralelo a decisiones de foundation lock segun DEC-007 (sin bloquear trabajo no-GPU).

### Proximo paso unico recomendado
- Cerrar Run C y formalizar foundation lock A/B/C; en paralelo, mantener habilitada implementacion de Gate 4.2 sin ejecutar screening hasta tener foundation definitivo.

### Bloqueantes / riesgos
- Desalineacion temporal entre estado documental troncal y estado experimental real.
- Cambios locales sin commit en scripts de entrenamiento pueden afectar reproducibilidad si no se auditan antes de relanzes.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md`
- `data/bias_control_medium/training_outputs/bloqueA_runA/`
- `data/bias_control_medium/training_outputs/bloqueA_runB/`
- `data/bias_control_medium/training_outputs/bloqueA_runC/`
