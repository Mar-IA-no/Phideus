## Auditoría técnica independiente R466

**Veredicto: REVISE — 2 HIGH, 2 MEDIUM, 0 LOW**

Commit auditado: `fe03fa7ce4a1ff3d896d4980ec690818384e5542`  
Parent directo: `2b4df86b45d5de30bc2cbe160a9259dab65d6c4f`

### Findings

1. **HIGH — La cadena Git exigida para recovery v2 es imposible.**

   Para toda config, `validate_wave60_final_config_authority()` exige que el commit config sea hijo directo de la auditoría source-law en [prepare_wave56_fresh.py:446](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:446)–[453](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:453).

   Simultáneamente, el recovery exige:

   - amendment audit como hijo directo del amendment commit en [prepare_wave56_fresh.py:3781](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3781)–[3792](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3792);
   - amendment audit como ancestro del HEAD de ejecución en [prepare_wave56_fresh.py:3793](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3793)–[3797](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3797).

   Para un recovery nacido después del aborto v1, la cadena necesaria sería:

   ```text
   v1 config audit -> amendment -> amendment audit -> v2 config -> v2 config audit
   ```

   Pero el código exige a la vez:

   ```text
   source-law audit -> v2 config -> v2 config audit
   ```

   sin ningún commit intermedio. Por tanto, el amendment audit no puede ser ancestro de HEAD.

   La prueba que pretende demostrar v1→v2 desactiva precisamente ambas restricciones con monkeypatches en [test_wave60_frozen_policy_transport.py:1617](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1617)–[1620](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1620), y tampoco recorre `preparation_preflight()`. Es una confianza falsa sobre la ejecutabilidad real.

   Corrección: para `attempt.version >= 2`, exigir que el config commit sea hijo directo del amendment audit; ligar éste a la auditoría final del intento anterior, manteniendo source-law e implementación como ancestros inmutables.

2. **HIGH — Puede publicarse y reutilizarse un aborto pair-level sin dos terminales root físicas válidas.**

   `publish_pair_failure()` sólo comprueba el string del terminal pair-level en [run_wave60_frozen_policy_transport.py:2913](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2913)–[2924](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2924). No verifica:

   - existencia de `primary/` y `replay/`;
   - terminales root y schemas cerrados;
   - firmas e inventories root;
   - que ambos hashes de `pair_status.json` correspondan a artefactos físicos.

   Probe mínimo observado:

   ```text
   {'pair_published': True,
    'primary_exists': False,
    'replay_exists': False,
    'bound_primary': '1111...1111',
    'bound_replay': '2222...2222'}
   ```

   El recovery tampoco cierra el hueco. `_validate_wave60_recovery_origin()` valida sólo el binding de `primary` en [prepare_wave56_fresh.py:3677](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3677)–[3699](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3699); no valida el terminal replay ni los schemas/inventories/manifests completos del paquete pair.

   La propia prueba v2 deja `replay/` vacío, usa un digest ficticio `"5"*64` y publica el pair igualmente en [test_wave60_frozen_policy_transport.py:1425](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1425)–[1426](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1426) y [1508](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1508)–[1519](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1519). También usa un `failure_inventory.json` deliberadamente inválido en [línea 1493](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1493), que el recovery acepta.

   Esto rompe la autoridad durable del aborto que habilita reutilizar el draw.

   Corrección: implementar un validador closed-world de ambos root terminals y del paquete pair, ejecutarlo antes de firmar/publicar y repetirlo durante recovery.

3. **MEDIUM — `prior_attempt_container` permite traversal fuera del namespace canónico.**

   `validate_pre_draw_config()` sólo aplica `startswith(...)` al contenedor previo en [wave60_frozen_policy_transport.py:1005](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave60_frozen_policy_transport.py:1005)–[1015](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave60_frozen_policy_transport.py:1015). Luego `_validate_wave60_recovery_origin()` compara únicamente el basename resuelto en [prepare_wave56_fresh.py:3600](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3600)–[3602](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3602).

   Probe:

   ```text
   prior_attempt_container =
     data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1/
     ../../../escaped/wave60_frozen_policy_transport_attempt_v9

   TRAVERSAL_CONFIG_ACCEPTED
   ```

   Con suficientes `..`, la resolución puede salir incluso del repositorio.

   Corrección: exigir un path relativo exacto de un solo componente bajo `data/geometria_proporcional`, con regex de versión, resolución confinada y versión previa menor que la actual.

4. **MEDIUM — El presupuesto de finalize deja de ser durable al reanudar staging.**

   Cuando `runtime.json` ya existe, se valida su total previo en [run_wave60_frozen_policy_transport.py:3509](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3509)–[3547](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3547), pero ese consumo no se incorpora al `before_finalize` de la reanudación. La barrera previa al rename usa sólo el tiempo de la invocación actual en [línea 3607](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3607).

   Probe con staging durable que ya registraba 899 s:

   ```text
   {'published': True,
    'recorded_total': 899.0,
    'actual_with_resumed_finalize': 901.0}
   ```

   Corrección: cargar el consumo previo del staging en el presupuesto restante mediante un ledger append-only o acumulativo, y firmar el total de todas las invocaciones antes del rename.

### Estado de los findings anteriores

| Finding R465 | Estado R466 |
|---|---|
| Reapertura de truth/model state durante finalize | Corregido |
| Recovery v2 superficial/no ejecutable | Sigue material: implementación local existe, pero cadena Git imposible y autoridad terminal incompleta |
| Semántica terminal y ocho asimetrías | Semántica root corregida; queda el HIGH pair-level |
| Cadena config/HEAD/blobs | Corregida para v1; incompatible con recovery v2 |
| Ledger combinado y direct finalize | Parcial: ejecución simple cubierta; reanudación no acumulativa |

Los findings R464 de cadena científica, identidad física por archivo y staging transitorio están corregidos. La suite amplió sustancialmente cobertura, pero los fixtures de recovery ocultan los dos defectos altos descritos.

### Verificaciones ejecutadas

- Focal Wave 60: `43 passed in 65.16s`; RSS máximo `851216 KiB`.
- Regresión exacta Wave 56–60, diez archivos, con basetemp dentro del repositorio: `379 passed, 1 skipped in 528.46s`; RSS máximo `1040132 KiB`.
- `py_compile` de los cinco paths: PASS.
- `git diff --check`: PASS.
- Worktree inicial y final: limpio.
- Temporales y bytecode propios: eliminados.
- Config, source authority y attempt canónicos Wave 60: ausentes.
- No se usó ni consultó GPU, web, Colab o Mendieta.
- Todas las ejecuciones llevaron `CUDA_VISIBLE_DEVICES=''` y cuatro threads.

Los cinco blobs físicos coinciden byte a byte con `HEAD`; SHA-256:

```text
681340c2cda65fe7dcd23d7e1f46295a547b5cad8bf42487472d9f87361f64c4  src/geometria_proporcional/wave60_frozen_policy_transport.py
dbedaedb782f5fe46ea3115025b9b1a1a30cd48f424e3058ea48a4d90a95d1d7  experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
c2ffafbda6234e2d7c92cc829b8f5634c9f5085a7b965f8e280243c31ef5591b  experiments/geometria_proporcional/_wave60_phase_worker.py
67391f88e915c3317295f13b85881c306761a96fe2d558f20e6ecc1fd7d7d51e  experiments/geometria_proporcional/prepare_wave56_fresh.py
afcddc65a8dd104b8bb52f13c7cb4964d6d01c4050e02e5f5e6e94e93d887827  tests/test_wave60_frozen_policy_transport.py
```

No incluyo bloque de autoridad PASS porque el veredicto es `REVISE`.
