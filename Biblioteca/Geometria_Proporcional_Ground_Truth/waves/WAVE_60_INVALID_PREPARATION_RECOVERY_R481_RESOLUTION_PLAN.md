# Ola 60 — resolución de la auditoría de implementación R481

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Implementación auditada:** commit
> `e617e15be290a62e5b0027c3748f1f5e85abd083`
> **Auditoría R481:** commit
> `50c1054a92fa6404b27c55a8f650690854a215ae`, SHA-256
> `5491467c5abdb50c0c14972d44ec4afbc7e41e5a93d363bf27253a8c2e428e2f`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Alcance de la resolución

R481 no reprodujo un defecto funcional en el adaptador de recuperación. Confirmó
por inspección la separación de autoridades R475/recuperación, la autenticación
del draw anidado, la cadena dura que termina en `hard_set_tau=0.5`, el débito
conservador one-shot y la compatibilidad legacy. El finding es probatorio: la
positiva añadida llama por separado al validador del hard set y al materializador,
de modo que podría seguir pasando aunque el ramal productivo que conecta ambos
estuviera ausente o mal cableado.

Esta resolución conserva el código productivo de `e617e15` salvo los cambios
necesarios para autenticar la historia adicional y para exponer una frontera de
lineage comprobable. La corrección principal será ampliar la suite hasta que
recorra la transacción real de preparación y observe sus artefactos. Si esas
pruebas revelan un defecto productivo, se lo corregirá dentro del mismo alcance;
no se presumirá de antemano que el código es correcto.

## 2. Historia que debe preservarse

La futura autoridad no puede tratar R481 como PASS. La amendment definitiva
incorporará cuatro objetos nuevos, además del keyset aprobado por R480:

```text
rejected_recovery_implementation
rejected_recovery_implementation_audit
r481_resolution_plan
r481_resolution_plan_audit
```

`rejected_recovery_implementation` tendrá exactamente:

```text
commit
parent
changed_sources
```

con `commit=e617e15be290a62e5b0027c3748f1f5e85abd083`,
`parent=abb8fd2e8c9119e46fabee2aa15405ceb146d4b2` y dos deltas desde R475:

```text
preparer
  path = experiments/geometria_proporcional/prepare_wave56_fresh.py
  old_sha256 = 7d7ead44f6d0e64802dafa585a59a20ae78f43f5e975e03c60e6bd8a1de33d66
  new_sha256 = 05571d22f2f406b07e89132482cb39715128e4b9c3a5abda1c85aac0b7143dcb

test
  path = tests/test_wave60_frozen_policy_transport.py
  old_sha256 = 328c934c63f2cb402633b72b52699e2d48433ff7e94966169a5b1428e6f63519
  new_sha256 = 369f1970c19b9fab2f80fd744bebadc6d0822d23b2593ab70d7f26f3e9dc5b17
```

`rejected_recovery_implementation_audit` tendrá exactamente `commit`, `path`,
`sha256`, `audit_id`, `scope`, `verdict` y `findings`. Sus valores fijos serán:

```text
commit = 50c1054a92fa6404b27c55a8f650690854a215ae
path = Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
       481_wave60_invalid_preparation_recovery_implementation_audit.md
sha256 = 5491467c5abdb50c0c14972d44ec4afbc7e41e5a93d363bf27253a8c2e428e2f
audit_id = R481
scope = INVALID_PREPARATION_RECOVERY_IMPLEMENTATION
verdict = REVISE
findings = {high: 0, medium: 1, low: 0}
```

El validador exigirá que `e617e15` sea hijo directo y exclusivo de R480 y que
R481 sea hijo directo y exclusivo de `e617e15`, con target, scope, verdict y
conteos exactos. No se reutilizará el parser PASS para autenticar un REVISE.

`r481_resolution_plan` ligará este documento mediante `commit`, `path` y
`sha256`. `r481_resolution_plan_audit` tendrá exactamente `commit`, `path`,
`sha256`, `audit_id`, `verdict` y `findings`, con `audit_id=R482`, scope
`INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN`, PASS y cero findings.

## 3. Autoridad de implementación sucesora

Después de R482 PASS habrá un nuevo commit de implementación que modificará
exactamente:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El preparador sólo cambiará para:

1. autenticar la historia `e617e15 → R481 REVISE → resolución → R482 PASS`;
2. desacoplar en una función testeable la verificación del sufijo Git de
   implementación, sin relajar el anchor R480 usado en producción;
3. renumerar las futuras autoridades y sus scopes exactos.

La suite incorporará la evidencia faltante. Ningún cambio alcanzará módulo,
runner o worker, que seguirán byte-exactos bajo R475. El commit sucesor será
auditado como R483 con scope
`INVALID_PREPARATION_RECOVERY_IMPLEMENTATION`, target exacto
`{implementation_commit: <commit sucesor>}`, PASS y cero findings. La futura
amendment apuntará sólo a esta autoridad positiva en `recovery_implementation`,
pero conservará por separado la implementación rechazada y R481.

Los deltas de `recovery_implementation.changed_sources` continuarán comparando
R475 contra el blob final, no `e617e15` contra el blob final. Así la config v2
acreditará el contenido completo que ejecuta y no sólo la corrección incremental.

## 4. Prueba integrada de la transacción productiva

La positiva nueva deberá atravesar `run_preparation_transaction()` —o
`execute_preparation()` más la finalización firmada cuando esa separación sea
necesaria para aislar el test— con:

- el draw canónico v1 copiado a un root temporal del mismo filesystem;
- el escrow anidado real y su contrato;
- el `recovery_context` del schema
  `wave60-invalid-preparation-recovery-amendment-v1`;
- `validate_wave60_invalid_preparation_hard_set_contract()` real;
- source law v2, manifest, request, alias y snapshot físicos reales;
- `materialize_prepared_bundles()` real;
- `CUDA_VISIBLE_DEVICES=''` y presupuesto/RSS observado.

Se permite sustituir únicamente la inferencia costosa por una frontera que copie
los logits pre-truth ya inventariados del intento v1 y devuelva la misma forma de
receipt que espera la transacción. Esa sustitución no puede construir bundles,
inyectar `hard_set_tau`, reemplazar el validador duro, omitir la copia del
benchmark ni fabricar los artefactos de provenance.

La prueba deberá demostrar simultáneamente:

1. `hard_set_tau=0.5` llega al materializador desde el snapshot autenticado;
2. la config canónica y el objeto config entregado al flujo permanecen sin esa
   clave antes y después;
3. escrow, freeze y todos los archivos del benchmark conservan bytes respecto
   del draw v1, pero cada archivo copiado tiene inode distinto;
4. `recovery_amendment.json` reproduce byte a byte la amendment autorizada;
5. `generation_receipt.json`, `preparation_freeze.json` y
   `preparation_receipt.json` contienen exactamente la misma
   `recovery_provenance.contract_extensions`;
6. la attestation de preparación firma el receipt que contiene esa provenance;
7. no aparece oracle, label autorizado, redraw ni mutación del origen.

La prueba fallará si se elimina o altera el ramal productivo que construye
`materializer_config`.

## 5. Lineage comprobable sin autoridad circular

La implementación precede necesariamente a su auditoría y a la amendment. Para
probar el validador sin inventar una autoridad futura, el preparador extraerá una
función de verificación de sufijo que reciba:

```text
repo_root
r480_anchor
rejected_implementation
r481_audit
resolution_plan
r482_audit
final_implementation
r483_audit
```

En producción, `r480_anchor` será el commit hardcodeado y ya autenticado
`abb8fd2…`. En la prueba sintética será el primer commit de un repositorio Git
temporal. La función aplicará idénticas reglas en ambos casos: parent directo,
commit exclusivo, blob físico/Git, scope, target, verdict, conteos y deltas de
sources. La inyección del anchor no podrá sustituir parsers, `git_changed_paths`,
`git_blob_sha256` ni las reglas de ancestry.

La fixture construirá el sufijo completo y comprobará una positiva y rechazos
independientes de:

- R481 tratado como PASS o con findings distintos;
- parent saltado en resolución, R482, implementación final o R483;
- path adicional en cualquiera de los commits exclusivos;
- cruce entre blobs R475 y blobs de recuperación;
- scope, target, audit id, hash físico o blob Git alterados.

Después de publicar amendment y config, el preflight canónico deberá recorrer de
nuevo `validate_recovery_amendment()` sobre la historia real antes de crear o
archivar ningún output. La fixture sintética no reemplaza esa comprobación final.

## 6. Matriz negativa física y del ledger

Sobre copias temporales del origen v1 se añadirán rechazos independientes para:

- bytes de escrow, freeze, manifest o benchmark alterados;
- modo u owner alterado del draw o de sus archivos sensibles;
- symlink, hardlink o nodo especial;
- inventario primario, firma de primary/replay o binding del par alterados;
- archivo adicional y mapa `preserved_draw_sha256` no closed-world.

El débito unsigned deberá rechazarse por separado cuando:

- `seconds` sea cero, negativo o distinto de `60.0`;
- `applied_once` sea falso;
- cambien régimen, wall observado o autoridad del registro externo;
- exista sólo uno de receipt/attestation;
- exista cualquier autoridad firmada en primary o replay;
- se intente aplicar desde otra versión, container o source.

Una positiva enlazará `primary_prior=60`, `replay_prior=primary_cumulative` y una
recuperación posterior desde el par firmado, comprobando que los 60 segundos no
se suman por segunda vez.

## 7. Schema y numeración definitivos

El schema de amendment se mantiene:

```text
wave60-invalid-preparation-recovery-amendment-v1
```

Su keyset top-level definitivo pasa a 23 claves:

```text
schema_version
status
recovery_kind
prior_attempt_container
prior_pair_failure_sha256
prior_config_audit
rejected_plan
rejected_plan_audit
r478_resolution_plan
r478_resolution_plan_audit
r479_resolution_plan
r479_resolution_plan_audit
rejected_recovery_implementation
rejected_recovery_implementation_audit
r481_resolution_plan
r481_resolution_plan_audit
recovery_implementation
hard_set_contract
unledgered_preparation_debit
escrow_origin
preserved_draw_sha256
population_contract
origin_inventory
```

`attempt.recovery` conserva exactamente las doce claves ya aprobadas. Los
campos físicos, `hard_set_contract`, débito, población y draw permanecen sin
cambios. Sólo se amplía la historia que la amendment autentica.

La numeración futura será:

- R482: auditoría de este plan;
- R483: reauditoría PASS de la implementación sucesora;
- R484: auditoría de amendment;
- R485: auditoría de config v2;
- R486: auditoría de resultados si el par llega a terminal evaluable.

Los paths serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  482_wave60_invalid_preparation_recovery_r481_resolution_plan_audit.md
  483_wave60_invalid_preparation_recovery_implementation_reaudit.md
  484_wave60_invalid_preparation_recovery_amendment_audit.md
  485_wave60_frozen_policy_transport_v2_config_audit.md
  486_wave60_frozen_policy_transport_result_audit.md
```

## 8. Lineage futuro

```text
R480 PASS abb8fd2
  → implementación rechazada e617e15
  → R481 REVISE 50c1054
  → esta resolución
  → R482 PASS
  → implementación sucesora preparer+test
  → R483 PASS
  → amendment
  → R484 PASS
  → config v2
  → R485 PASS / HEAD de ejecución
```

Cada commit será hijo directo y exclusivo. La ejecución canónica continúa
prohibida hasta que la config v2 y R485 sean HEAD, el worktree esté limpio y el
preflight real valide toda la cadena. Ninguna de estas autoridades declara
`GO/NO-GO` científico.

## 9. Criterio de autorización

Este documento no autoriza todavía la corrección. Requiere una auditoría
independiente R482 con target exacto `plan_commit + plan_sha256`, PASS y cero
findings. La auditoría deberá decidir si la prueba integrada propuesta cubre el
wiring productivo sin reemplazar precisamente los componentes bajo prueba, si
el sufijo Git sintético evita circularidad sin debilitar el anchor productivo y
si la matriz negativa es suficiente para cerrar el MEDIUM R481.
