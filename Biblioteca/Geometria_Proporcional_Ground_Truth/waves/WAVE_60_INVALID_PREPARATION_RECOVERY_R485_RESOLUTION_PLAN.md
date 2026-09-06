# Ola 60 — resolución probatoria del sufijo Git observada por R485

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Implementación auditada:** commit
> `f32ba2bb5a6f38b4a3e9e9afbafb9393430f6719`
> **Auditoría R485:** commit
> `b038cf91fffe66689b103e1b3a28503b6975d645`, informe
> `485_wave60_invalid_preparation_recovery_implementation_final_reaudit.md`,
> SHA-256
> `d58148ce731c3d06e61101f44dd7db5165b9932ce47c8acc85b45383ffe15650`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Alcance

R485 confirmó el cierre sustantivo de las defensas físicas y del ledger. La
suite ya cubre las mutaciones del draw y sus autoridades firmadas, los casos de
frontera receipt/attestation y la cadena completa:

```text
v1 unsigned = 60
  -> primary v2 firmado = 61
  -> replay v2 firmado = 63
  -> par pre-truth durable = 63.5
  -> recovery v3 prior = 63.5
```

El finding restante es exclusivamente probatorio y de lineage. La fixture Git
no materializa todavía todos los negativos que el plan R483 exigió; además, una
positiva de transaction/provenance conserva `audit_id=R483` aunque R483 es una
autoridad `REVISE`. R485 no encontró un bypass equivalente en producción: el
flujo real obtiene la provenance del objeto final `recovery_implementation`.

Esta resolución autoriza únicamente:

1. completar los negativos Git omitidos;
2. corregir la fixture positiva para representar la autoridad final;
3. añadir a la amendment la historia `f32ba2b -> R485 REVISE` y esta
   resolución;
4. modificar producción sólo en los guards de lineage necesarios para
   autenticar esa historia o si un nuevo negativo reproduce un defecto real.

No se modifica el draw, la source law, el modelo, el roster, los estados HGB,
los thresholds, el estimando, el runner científico ni el worker.

## 2. Historia que debe conservarse

`f32ba2b` queda preservado como implementación incompleta; no puede ocupar el
objeto positivo `recovery_implementation`. La cadena futura será:

```text
R480 PASS
  -> e617e15 implementación rechazada
  -> R481 REVISE
  -> resolución R481
  -> R482 PASS
  -> 5aee5fb implementación incompleta
  -> R483 REVISE
  -> resolución R483
  -> R484 PASS
  -> f32ba2b implementación probatoriamente incompleta
  -> R485 REVISE
  -> esta resolución
  -> R486 PASS
  -> implementación sucesora
  -> R487 PASS
  -> amendment
  -> R488 PASS
  -> config v2
  -> R489 PASS / HEAD de ejecución
```

La amendment conservará las 27 claves previstas por R483 y añadirá exactamente:

```text
r483_resolution_implementation
r483_resolution_implementation_audit
r485_resolution_plan
r485_resolution_plan_audit
```

El keyset top-level tendrá, por tanto, 31 claves. Los cuatro objetos nuevos se
definen así:

- `r483_resolution_implementation`: `commit`, `parent` y `changed_sources`;
  commit `f32ba2bb5a6f38b4a3e9e9afbafb9393430f6719`, parent
  `f26404044cd87cc14deea22cb0d14fa54b6134ae` y deltas desde R475;
- `r483_resolution_implementation_audit`: `commit`, `path`, `sha256`,
  `audit_id`, `scope`, `verdict` y `findings`; R485, `REVISE`, `0/1/0`;
- `r485_resolution_plan`: `commit`, `path` y `sha256` de este documento;
- `r485_resolution_plan_audit`: `commit`, `path`, `sha256`, `audit_id`,
  `verdict` y `findings`; R486, `PASS`, `0/0/0`.

Los deltas de `f32ba2b`, calculados siempre contra R475 y no contra la
implementación intermedia, son:

```text
preparer
  old = 7d7ead44f6d0e64802dafa585a59a20ae78f43f5e975e03c60e6bd8a1de33d66
  new = b21d89af10021904347563997b3ec1e13292558cfc30519bd2c5f4d8cc8d7f32

test
  old = 328c934c63f2cb402633b72b52699e2d48433ff7e94966169a5b1428e6f63519
  new = 979054582ca1a61e6834f19d65e8d39954b431e1482f2c72a14af830c0363fdd
```

`recovery_implementation` queda reservado para el sucesor auditado por R487.
Los tres sources científicos continúan bajo R475 y sus hashes no cambian.

## 3. Corrección de la positiva de provenance

La fixture
`test_invalid_preparation_transaction_wires_tau_and_signed_provenance` debe
construir un contexto sintético coherente con la autoridad final:

```text
implementation_audit.audit_id = R487
```

La aserción debe seguir comprobando que la provenance completa aparece
idénticamente en generation receipt, preparation freeze, preparation receipt y
attestation. Este cambio no debilita el test ni introduce un fallback. El ID
R483 sólo permanece en objetos explícitamente etiquetados como `REVISE`.

## 4. Matriz de campos de auditoría

La fixture sintética del sufijo deberá materializar y validar la cadena completa
hasta R487. Para cada auditoría R481, R482, R483, R484, R485 y R487 se probarán
por separado todos los campos semánticos aplicables:

```text
audit_id
scope
target
technical_verdict
findings
```

También se probarán por separado los bindings externos aplicables:

```text
commit / audit_commit
path / audit_path
sha256 / audit_sha256
```

No basta mutar el diccionario entregado al helper cuando el campo real vive
dentro del reporte. Para `scope`, `target`, `technical_verdict` y `findings`, la
suite deberá construir un reporte alternativo, commitearlo como único path con
el parent correcto, actualizar coherentemente su binding externo y demostrar
que el parser semántico lo rechaza. El test verificará primero como
precondiciones:

- que el commit malo cambia exactamente el path de reporte esperado;
- que es hijo directo del parent que esa clase requiere;
- que el SHA declarado coincide con el archivo físico alternativo;
- que sólo el campo objetivo difiere de la autoridad positiva.

Así cada rechazo acredita el guard específico y no un error anterior de path,
hash o parent.

## 5. Divergencia blob Git frente a archivo físico

Debe existir al menos un negativo propio para documento y uno para auditoría en
el que:

1. el commit ligado contiene el path correcto y es exclusivo;
2. su blob Git contiene bytes A;
3. el archivo físico contiene bytes B;
4. el binding declara el SHA de B;
5. parent, path y resto del objeto permanecen válidos.

La suite comprobará explícitamente antes de llamar al validador que:

```text
sha256(blob Git) != sha256(archivo físico) == binding.sha256
```

El rechazo esperado deberá provenir de la comparación blob↔binding y no de un
path inexistente, un commit no exclusivo o un hash físico incorrecto.

## 6. Exclusividad de commits con un solo path extra

Para cada clase exclusiva del sufijo —implementaciones, planes y auditorías— se
construirá un commit malo desde su parent correcto que contenga exactamente:

```text
paths esperados de esa clase + unexpected-exclusive-path.txt
```

Antes del rechazo se verificará con `git_changed_paths()` que ése sea el delta
exacto. No se reutilizará como tree fuente una revisión posterior que arrastre
cambios de otras etapas. El caso deberá fallar por exclusividad y no por un
target, parent o blob incoherente.

## 7. Mutación real de sources y partición R475/recuperación

La fixture incorporará físicamente los tres sources bajo autoridad R475:

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
```

El helper de lineage deberá verificar para cada uno que su blob en la
implementación final sea idéntico al blob R475. Se ejecutarán tres negativos
independientes, uno por source. Cada negativo construirá una cadena secundaria
en la que el source se altera realmente después de R475 y antes del ancla R480;
desde ese ancla, todos los commits del sufijo conservarán parents directos y
paths exclusivos válidos. De este modo, el único hecho incorrecto al llegar a
la implementación final será:

```text
git_blob(final_implementation, scientific_source)
  != git_blob(R475, scientific_source)
```

La prueba no se sustituye por acortar
`unchanged_source_law_sources`. Esa mutación de lista se conserva como negativo
de schema, mientras la alteración real prueba la frontera de blobs.

Para preparer y test se mantienen negativos de `old_sha256`, `new_sha256`, path
y cruce entre las tres implementaciones rechazadas y la final. Los hashes old
continúan anclados en R475.

## 8. Autoridad del sucesor

Después de R486 PASS se publicará un commit hijo directo que modificará
exactamente:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El preparador sólo podrá:

- autenticar los cuatro objetos históricos nuevos;
- ampliar a 31 el keyset de amendment;
- verificar los tres blobs científicos en el helper de sufijo;
- renumerar las autoridades futuras a R487–R490.

La suite sólo podrá completar las matrices de §§3–7. Si un negativo reproduce
un defecto adicional, deberá documentarse antes de ampliar producción.

La auditoría R487 tendrá:

```text
scope = INVALID_PREPARATION_RECOVERY_IMPLEMENTATION
target = {implementation_commit: <commit sucesor>}
technical_verdict = PASS
findings = {high: 0, medium: 0, low: 0}
```

R487 deberá leer la implementación completa y ejecutar en CPU:

- la suite Wave 60 completa;
- la regresión explícita Waves 56–59;
- `git diff --check` y comprobación de blobs científicos.

Una salida verde no sustituye la comprobación caso por caso del finding R485.

## 9. Numeración y paths futuros

La numeración no reutiliza IDs ya emitidos:

- R486: auditoría de este plan;
- R487: auditoría PASS de la implementación sucesora;
- R488: auditoría de amendment;
- R489: auditoría de config v2;
- R490: auditoría de resultados si el par llega a un terminal evaluable.

Los paths serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  486_wave60_invalid_preparation_recovery_r485_resolution_plan_audit.md
  487_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md
  488_wave60_invalid_preparation_recovery_amendment_audit.md
  489_wave60_frozen_policy_transport_v2_config_audit.md
  490_wave60_frozen_policy_transport_result_audit.md
```

El schema de amendment conserva el identificador
`wave60-invalid-preparation-recovery-amendment-v1`: sólo amplía su historia. El
objeto `attempt.recovery` conserva exactamente las doce claves ya aprobadas. La
config canónica sigue sin `hard_set_tau`; `0.5` se inyecta únicamente en la
copia efímera entregada al materializador.

## 10. Criterio de autorización

Este plan no autoriza todavía una nueva implementación. Requiere una auditoría
independiente R486 con target exacto `plan_commit + plan_sha256`, scope
`INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN`, PASS y cero findings.

La ejecución canónica permanece prohibida hasta encadenar por parents directos
y commits exclusivos: R487 PASS, amendment, R488 PASS, config v2 y R489 PASS
como HEAD limpio. Todo el trabajo es CPU-only. Ninguna autoridad de esta cadena
promueve una arquitectura ni decide `GO/NO-GO` científico.
