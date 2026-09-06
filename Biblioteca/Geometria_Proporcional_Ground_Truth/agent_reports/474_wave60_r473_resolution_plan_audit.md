# R474 — Auditoría independiente del plan de resolución de R473 para Wave 60

## Identidad auditada

- El identificador solicitado `1ca64d1f` no resuelve como revisión Git. El commit real cuyo contenido coincide con el SHA-256 esperado es `1ca64d169a565e3be4e284f0f0104af98675f361`.
- Plan: `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md`.
- SHA-256 verificado: `a7d4dea482c9062213eddfddd14a21dbf2e2bf484a37481d4522a14a9c0ee6f1`.
- Parent directo: `50b07dcfb8509ec5b235da44a6fa4328fd4db737`.
- El commit modifica exclusivamente el plan de resolución.
- Veredicto técnico: **PASS — 0 HIGH, 0 MEDIUM, 0 LOW**.

## Findings

### HIGH: 0

Ninguno.

### MEDIUM: 0

Ninguno.

### LOW: 0

Ninguno.

## Preservación de la historia y autoridad

La cadena física observada coincide exactamente con la historia declarada:

- plan base `a8e8932e38fc5b38d5a568f8dbe8fe8bc62b0ae9`, exclusivo;
- R472 `346e9fc3ec2dbefd5a560c47cd30d35caa98e3f2`, exclusivo y child directo del plan;
- implementación rechazada `53c383284e222cd60890a87857d66a8245f1a148`, child directo de R472 y limitada a cuatro de los cinco paths autorizados;
- R473 `50b07dcfb8509ec5b235da44a6fa4328fd4db737`, exclusivo y child directo de la implementación rechazada;
- plan de resolución `1ca64d169a565e3be4e284f0f0104af98675f361`, exclusivo y child directo de R473.

La nueva secuencia preserva `53c3832` y su `REVISE`; no los convierte retrospectivamente en autoridad positiva. El plan exige autenticar R473 con verdict, scope, target y conteos negativos exactos, y luego construir una nueva cadena `plan de resolución → auditoría PASS → implementación corregida → reauditoría PASS` (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:36-99`).

La sustitución de la única arista `parent(implementación)=R472` del plan base está acotada al caso de un `REVISE` autenticado (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:65-73`). Las aristas posteriores —auditoría source law, config y auditoría de config— permanecen como fueron aprobadas en el plan base (`WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:342-370`). No hay reescritura, bifurcación ni autoridad circular.

## Resolución de los tres findings R473

### HIGH — scope incompatible

Queda cerrado por diseño.

El código rechazado fija `scope="IMPLEMENTATION"` dentro de `validate_implementation_audit_authority()` y el preparador repite ese literal (`run_wave60_frozen_policy_transport.py:362-429`; `prepare_wave56_fresh.py:1120-1126`). La resolución exige convertir el scope en argumento explícito sin default ambiguo y asigna:

- `IMPLEMENTATION` exclusivamente a v1;
- `SOURCE_LAW_RECOVERY_IMPLEMENTATION` a publisher v2, config futura y validación de la implementación corregida.

También exige cobertura positiva y negativa sobre scope, target, verdict, conteos y parents (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:105-124`). La modificación es local y realizable dentro del runner, preparador y tests ya autorizados.

### MEDIUM — publicación recovery fuera del namespace v2

Queda cerrado por diseño.

La implementación rechazada crea staging antes de decidir si la invocación es recovery y deriva esa identidad de la igualdad del output (`run_wave60_frozen_policy_transport.py:1917-1949`). La resolución separa:

- `publish_source_law_authority`, exclusivamente legacy;
- `publish_source_law_recovery`, exclusivamente recovery;
- el CLI `verify-source-law`, ligado únicamente al segundo.

El entrypoint recovery debe validar schema, output v2 exacto, traversal, aliases, symlinks, target y staging antes de escribir. El legacy debe rechazar el schema recovery antes de reservar namespace (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:126-154`). Los rechazos relativo y absoluto y la ausencia posterior de target/staging quedan requeridos en tests (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:182-188`).

Esta separación evita que la selección de semántica dependa del output proporcionado y elimina la ruta que produjo un terminal legacy con `recovery_allowed=true`.

### MEDIUM — ausencia de prueba positiva integrada

Queda cerrado por un contrato de prueba suficiente y no circular.

La prueba requerida atraviesa el mismo publisher recovery usado por el CLI y cubre conjuntamente API relativa, API absoluta, dispatch CLI, worker real, presupuesto acumulativo, firma, manifest closed-world y rename final (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:156-180`).

El plan permite dos construcciones legítimas:

1. una cadena Git temporal completa;
2. inyección exclusiva de la frontera de autoridad, siempre que esa frontera esté probada separadamente contra Git.

En el segundo caso quedan prohibidos monkeypatches del worker, presupuesto, journal, attestation, manifest y rename. Por ello la inyección permitida no sustituye el comportamiento que debía integrar la prueba ni usa el componente bajo prueba para concederse su propio PASS.

Las variantes relativa, absoluta y CLI pueden ejecutarse en fixtures aislados con roots temporales de una sola publicación; no requieren reutilizar ni borrar el namespace canónico.

## Realizabilidad de entrypoints y scopes

Los cinco paths del plan base son suficientes:

- el módulo `src` puede conservar las constantes de plan/R472 y agregar las de `53c3832`, R473, resolución y R474;
- el runner contiene el validator Git, publishers, canonicalización y dispatch CLI;
- el preparador ya recibe un scope explícito en `validate_wave60_audit_commit()`, por lo que sólo necesita seleccionar el scope recovery en la rama v2;
- el worker ya distingue los schemas v1 y recovery sin acceder a Git ni al terminal v1 (`_wave60_phase_worker.py:145-210`);
- la suite existente concentra fixtures de request, publisher, worker, config y namespaces.

La frase “worker cuando corresponda” no obliga a introducir acceso Git dentro del sandbox: el worker sólo necesita conservar la validación tipada del request; la autoridad Git permanece en el coordinador.

La implementación corregida podrá ligar su propio commit mediante los campos del request y su auditoría directa. No necesita hardcodear anticipadamente su SHA, por lo que tampoco aparece una circularidad de commits.

## Estimando, draw y alcance científico

El plan limita la corrección a lineage, scopes, namespace y cobertura del publisher (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:29-34`). Conserva íntegramente los invariantes científicos del plan base (`WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:54-73`) y exige reproducción exacta de los outputs fuente.

La prueba positiva consume únicamente namespaces temporales. La ejecución canónica continúa prohibida hasta después de la reauditoría positiva de implementación (`WAVE_60_SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN.md:190-193,214-231`).

El estado físico observado confirma:

- terminal v1 presente;
- source law v2 ausente;
- staging v2 ausente;
- attempt Wave 60 ausente;
- staging de attempt ausente.

Por tanto, el plan no cambia el estimando, no materializa config final, no consulta truth y no consume draw.

## Condiciones para la auditoría de implementación sucesora

Sin constituir findings contra el plan, la auditoría siguiente deberá comprobar:

- que ningún caller recovery conserva el scope histórico;
- que ambos publishers rechazan el schema ajeno antes de reservar staging;
- que output recovery alternativo no deja archivos ni `recovery_allowed=true`;
- que la prueba positiva integrada usa el publisher real y no sustituye worker, presupuesto, journal, firma, manifest o rename;
- que las variantes API relativa, API absoluta y CLI usan namespaces temporales independientes;
- que los blobs y parents de `53c3832`, R473, resolución y R474 se validan físicamente;
- que v1 permanece byte-exacto y que v2/attempt canónicos continúan ausentes durante toda la regresión.

## Verificaciones realizadas

- Lectura completa del plan de resolución: `242` líneas.
- Lectura completa del plan base: `383` líneas.
- Lectura completa de R472: `69` líneas.
- Lectura completa de R473: `95` líneas.
- Inspección focal del código vigente en los cinco paths autorizados.
- SHA-256 físicos y blobs Git coincidentes para plan base, R472, R473 y plan de resolución.
- Parents directos y paths exclusivos verificados para toda la cadena `a8e8932 → 346e9fc → 53c3832 → 50b07dc → 1ca64d1`.
- `git diff --check` limpio.
- Worktree sin cambios.
- No se ejecutaron tests largos ni publisher/recovery.
- No se modificaron archivos.
- No se usó ni consultó GPU, Colab o Mendieta.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R474",
  "scope": "SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "1ca64d169a565e3be4e284f0f0104af98675f361",
    "plan_sha256": "a7d4dea482c9062213eddfddd14a21dbf2e2bf484a37481d4522a14a9c0ee6f1"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
